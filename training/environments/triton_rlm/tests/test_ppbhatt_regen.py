import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from triton_rlm.trace_translate import SUBMIT_IDIOM, ParsedTrace
from triton_rlm.verifier import format_verify_report

from rlm_train.repl.base import ExecResult

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str):
    # `python scripts/<name>.py` puts scripts/ on sys.path; the sibling import needs it here
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = "import torch\n\n\nclass Model(torch.nn.Module):\n    def forward(self, x):\n        return x * 2\n\n\ndef get_inputs():\n    return [torch.randn(4)]\n\n\ndef get_init_inputs():\n    return []\n"
FAILED = "import triton\n\n\ndef triton_forward(x):\n    return k[grid, 4](x)"
FIXED = FAILED.replace("k[grid, 4](x)", "k[grid](x, num_warps=4)")
BROKEN = FAILED.replace("k[grid, 4](x)", "k[grid](x, num_warps=tl.constexpr)")


def trace() -> ParsedTrace:
    return ParsedTrace.from_json(
        {
            "sample_key": "kernelbook_1",
            "source": "kernelbook",
            "pytorch_code": REF,
            "stop_reason": "correct",
            "ops": ["mul"],
            "root_class": "Model",
            "cluster": 0,
            "split": "train",
            "turns": [
                {"index": 0, "code": FAILED, "recorded": {"kind": "error", "exc_type": "TypeError"}}
            ],
        }
    )


class ScriptedBackend:
    """Stands in for VerifyingReplBackend: `outcomes[code]` -> (exception, verify)."""

    outcomes: dict[str, tuple[str | None, dict[str, Any] | None]] = {}
    started: list[str] = []

    def __init__(self, inner: Any, reference: str) -> None:
        self.last_verify: dict[str, Any] | None = None

    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None:
        self.started.append(rollout_id)

    async def load_context(self, payload: Any, index: int | None = None) -> int:
        return 0

    async def execute(self, code: str) -> ExecResult:
        exception, verify = self.outcomes[code]
        self.last_verify = verify
        stdout = format_verify_report(verify) + "\n" if verify is not None else ""
        return ExecResult(stdout=stdout, exception=exception, locals_keys=["answer", "kernel_src"])

    async def stop(self) -> None:
        pass


TYPE_ERR = {
    "compiled": False,
    "correct": False,
    "error": "TypeError: 'tuple' object cannot be interpreted as an integer",
}
COMPILE_ERR = {"compiled": False, "correct": False, "error": "CompilationError: at 3:4: bad"}
CORRECT = {
    "compiled": True,
    "correct": True,
    "speedup": 1.5,
    "max_diff": 0.0,
    "ref_ms": 0.03,
    "kernel_ms": 0.02,
}


@pytest.fixture
def regen(monkeypatch):
    mod = load("ppbhatt_regen")
    monkeypatch.setattr(mod, "VerifyingReplBackend", ScriptedBackend)
    monkeypatch.setattr(mod, "SubprocessReplBackend", lambda: None)
    return mod


def task(regen, t: ParsedTrace) -> dict[str, Any]:
    prefix = [
        {"code": FAILED, "provenance": "teacher", "ours": "error[TypeError]", "source_turn": 0}
    ]
    executed = [(FAILED, ExecResult(exception=None, locals_keys=["answer", "kernel_src"]))]
    # prefix REPL output in the prompt is what OUR verifier said; use the same report
    messages = regen.render_prefix(t, executed, 8)
    limits = {
        "max_changed_lines": 12,
        "max_regen_turns": 3,
        "max_iterations": 8,
        "min_head_ratio": 0.6,
    }
    return regen.make_task(
        t, prefix, messages, round_no=0, regen_turns=0, limits=limits, start_reason="test"
    )


def test_execute_finishes_on_correct_with_submit_and_provenance(regen) -> None:
    ScriptedBackend.outcomes = {FAILED: (None, TYPE_ERR), FIXED: (None, CORRECT)}
    t = trace()
    out = asyncio.run(regen.execute_task(task(regen, t), FIXED, t))
    assert out["action"] == "finish", out["reason"]
    assert out["prefix_replay"] == ["error[TypeError]"]
    traj = out["trajectory"]
    assert traj["messages"][-1]["content"] == f"```repl\n{SUBMIT_IDIOM}\n```"
    assert traj["messages"][-4]["content"] == f"```repl\n{FIXED}\n```"
    assert "correct: True" in traj["messages"][-3]["content"]
    assert traj["messages"][-2]["content"].startswith("Turn 3/8")
    meta = traj["triton_rlm"]
    assert meta["provenance"] == "regenerated"
    assert meta["turn_provenance"] == ["teacher", "regenerated"]
    assert meta["repair"] is True and meta["repair_diff_lines"] == [2]
    assert meta["our_classes"] == ["error[TypeError]", "correct_fast"]
    assert [r["turn"] for r in out["rows"]] == [1, 2, 3]
    assert out["rows"][1]["messages"] == traj["messages"][:-3]


def test_execute_continues_on_new_error_with_next_task(regen) -> None:
    ScriptedBackend.outcomes = {FAILED: (None, TYPE_ERR), BROKEN: (None, COMPILE_ERR)}
    t = trace()
    out = asyncio.run(regen.execute_task(task(regen, t), BROKEN, t))
    assert out["action"] == "continue", out["reason"]
    nxt = out["next_task"]
    assert nxt["round"] == 1 and nxt["regen_turns"] == 1
    assert [p["provenance"] for p in nxt["prefix"]] == ["teacher", "regenerated"]
    assert nxt["messages"][-1]["content"].startswith("Turn 3/8")
    assert "CompilationError" in nxt["last_feedback"]


def test_execute_drops_repeated_error_and_nondeterministic_prefix(regen) -> None:
    ScriptedBackend.outcomes = {FAILED: (None, TYPE_ERR), FIXED: (None, TYPE_ERR)}
    t = trace()
    out = asyncio.run(regen.execute_task(task(regen, t), FIXED, t))
    assert out["action"] == "drop" and "same failure" in out["reason"]

    ScriptedBackend.outcomes = {FAILED: (None, COMPILE_ERR), FIXED: (None, CORRECT)}
    out = asyncio.run(regen.execute_task(task(regen, t), FIXED, t))
    assert out["action"] == "drop" and out["reason"].startswith("prefix nondeterministic")


def test_execute_rejects_prose_and_multiple_blocks(regen) -> None:
    t = trace()
    out = asyncio.run(regen.execute_task(task(regen, t), "Here is the fix:\n```repl\nx=1\n```", t))
    assert out["action"] == "drop" and "single block" in out["reason"]


def test_load_actions_strips_fence_and_rejects_duplicates(regen, tmp_path: Path) -> None:
    (tmp_path / "actions.shard0.jsonl").write_text(
        json.dumps({"sample_key": "a", "round": 0, "code": "```repl\nx = 1\n```"}) + "\n"
    )
    assert regen.load_actions(tmp_path, 0) == {"a": "x = 1"}
    (tmp_path / "actions.shard1.jsonl").write_text(
        json.dumps({"sample_key": "a", "round": 0, "code": "y"}) + "\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        regen.load_actions(tmp_path, 0)
