import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from triton_rlm.trace_translate import SUBMIT_IDIOM, ParsedTrace, ParsedTurn, TurnClass
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


def _traj(key: str, split: str, repair: bool, provenance: str = "teacher") -> dict[str, Any]:
    return {
        "example_id": key,
        "messages": [],
        "triton_rlm": {"split": split, "repair": repair, "provenance": provenance},
    }


def _rows(key: str, n: int) -> list[dict[str, Any]]:
    return [
        {"example_id": key, "turn": i, "messages": [], "triton_rlm": {"provenance": "x"}}
        for i in range(1, n + 1)
    ]


def _audit(
    key: str,
    deviation: list[str] | None,
    feedback: str = "exact",
    feedback_deviation: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "sample_key": key,
        "prompt_deviation": deviation,
        "feedback": feedback,
        "feedback_deviation": feedback_deviation or [],
        "disqualified": feedback == "stale" or (deviation is not None and "other" in deviation),
    }


def _result(key: str, action: str, reason: str) -> dict[str, Any]:
    return {"sample_key": key, "round": 0, "action": action, "reason": reason}


def test_collect_excludes_original_repairs_and_caps_regenerated(regen, tmp_path: Path) -> None:
    reexec = tmp_path / "reexec"
    rd = tmp_path / "regen" / "round0"
    reexec.mkdir()
    rd.mkdir(parents=True)
    originals = [_traj("a", "train", False), _traj("b", "train", False), _traj("c", "train", False)]
    originals.append(_traj("r", "train", True))
    regen.write_jsonl(reexec / "sft_trajectories.train.jsonl", originals)
    regen.write_jsonl(
        reexec / "sft_rows.train.jsonl",
        [*_rows("a", 2), *_rows("b", 2), *_rows("c", 2), *_rows("r", 3)],
    )
    regen.write_jsonl(reexec / "sft_trajectories.val.jsonl", [_traj("v", "val", True)])
    regen.write_jsonl(reexec / "sft_rows.val.jsonl", _rows("v", 3))
    finished_keys = ("r", "r1", "r2", "r3", "tr", "st", "v")
    finished = [
        _traj(k, "val" if k == "v" else "train", True, "regenerated") for k in finished_keys
    ]
    regen.write_jsonl(rd / "sft_trajectories.regen.jsonl", finished)
    regen.write_jsonl(
        rd / "sft_rows.regen.jsonl", [row for k in finished_keys for row in _rows(k, 3)]
    )
    regen.write_jsonl(
        rd / "results.jsonl",
        [
            *(_result(k, "finish", "correct_fast") for k in finished_keys),
            _result("rw", "drop", "rewrite, not repair: 20 changed lines > 12"),
            _result("co", "continue", "new failure incorrect"),
        ],
    )

    ns = type("NS", (), {})()
    ns.reexec_dir, ns.regen_dir, ns.max_share, ns.seed = (
        str(reexec),
        str(tmp_path / "regen"),
        0.3,
        0,
    )
    ns.exclude_original_repairs = True
    ns.truncated = ["tr"]
    with pytest.raises(SystemExit, match="audit"):
        regen.collect(ns)
    regen.write_jsonl(
        rd / "audit.jsonl",
        [
            _audit("r", []),
            _audit("r1", ["missing_repl_variables_line"]),
            _audit("r2", ["missing_repl_variables_line"], "cosmetic", ["float_values"]),
            _audit("r3", ["missing_repl_variables_line"], "cosmetic", ["repl_filename"]),
            _audit("tr", ["missing_repl_variables_line"]),
            _audit("v", ["missing_repl_variables_line"]),
            _audit("st", [], "stale"),
            _audit("rw", []),
            _audit("co", []),
        ],
    )
    regen.collect(ns)
    summary = json.loads((tmp_path / "regen" / "mix" / "summary.json").read_text())
    assert summary["original_trajectories_total"] == 5
    assert summary["original_repairs_excluded"] == 2
    assert summary["regenerated_tasks"] == 9
    assert summary["regenerated_finished"] == 7
    # int(0.3 * 3 / 0.7) == 1 regenerated trajectory fits under the cap next to 3 originals
    assert summary["regenerated_kept_after_cap"] == 1
    excluded = summary["excluded"]
    assert excluded["tr"] == "truncated_history"
    assert excluded["st"].startswith("audit_disqualified: feedback stale")
    assert excluded["rw"] == "no_clean_repair: rewrite, not repair: 20 changed lines > 12"
    assert excluded["co"] == "no_clean_repair: continue, round 1 not run"
    assert excluded["v"].startswith("regenerated_val_split")
    assert "a" not in excluded
    resampled = [k for k, why in excluded.items() if why.startswith("resampled_out")]
    assert len(resampled) == 3 and set(resampled) < {"r", "r1", "r2", "r3"}
    assert summary["excluded_by_reason"] == {
        "truncated_history": 1,
        "audit_disqualified": 1,
        "no_clean_repair": 2,
        "regenerated_val_split": 1,
        "resampled_out": 3,
    }
    assert summary["loss_masking"]["sft_rows.<split>.jsonl"].startswith("USE THIS for SFT")
    assert summary["per_split"]["train"] == {
        "original_trajectories": 3,
        "regenerated_trajectories": 1,
        "regenerated_share": 0.25,
        "rows": 9,
    }
    assert summary["per_split"]["val"] == {
        "original_trajectories": 0,
        "regenerated_trajectories": 0,
        "regenerated_share": 0.0,
        "rows": 0,
    }
    train = regen.load_jsonl(tmp_path / "regen" / "mix" / "sft_trajectories.train.jsonl")
    assert sorted(t["triton_rlm"]["provenance"] for t in train) == [
        "regenerated",
        "teacher",
        "teacher",
        "teacher",
    ]
    [kept] = [t for t in train if t["triton_rlm"]["provenance"] == "regenerated"]
    key = kept["example_id"]
    assert key not in excluded
    expected_prompt = {"r": [], "r1": ["missing_repl_variables_line"]}
    expected_prompt |= dict.fromkeys(("r2", "r3"), ["missing_repl_variables_line"])
    expected_feedback = {"r": [], "r1": [], "r2": ["float_values"], "r3": ["repl_filename"]}
    assert kept["triton_rlm"]["prompt_deviation"] == expected_prompt[key]
    assert kept["triton_rlm"]["feedback_deviation"] == expected_feedback[key]
    assert summary["regenerated_kept_prompt_deviation"] == {
        regen.deviation_label(expected_prompt[key]): 1
    }
    assert summary["regenerated_kept_feedback_deviation"] == dict.fromkeys(
        expected_feedback[key], 1
    )
    rows = regen.load_jsonl(tmp_path / "regen" / "mix" / "sft_rows.train.jsonl")
    regen_rows = [r for r in rows if r["example_id"] == key]
    assert len(regen_rows) == 3
    assert all(r["triton_rlm"]["prompt_deviation"] == expected_prompt[key] for r in regen_rows)
    assert all(r["triton_rlm"]["feedback_deviation"] == expected_feedback[key] for r in regen_rows)

    ns.max_share = 0.6  # int(0.6 * 3 / 0.4) == 4: every eligible regeneration fits
    regen.collect(ns)
    summary = json.loads((tmp_path / "regen" / "mix" / "summary.json").read_text())
    assert summary["regenerated_kept_after_cap"] == 4
    assert "resampled_out" not in summary["excluded_by_reason"]
    assert summary["regenerated_kept_prompt_deviation"] == {
        "exact": 1,
        "missing_repl_variables_line": 3,
    }
    assert summary["regenerated_kept_feedback_deviation"] == {"float_values": 1, "repl_filename": 1}
    assert summary["per_split"]["train"]["regenerated_share"] == 4 / 7

    ns.exclude_original_repairs = False
    regen.collect(ns)
    summary = json.loads((tmp_path / "regen" / "mix" / "summary.json").read_text())
    assert summary["original_repairs_excluded"] == 0
    assert summary["per_split"]["train"]["original_trajectories"] == 4
    assert summary["per_split"]["val"]["original_trajectories"] == 1
    assert summary["excluded"]["r"].startswith("original_kept")
    assert summary["excluded"]["v"].startswith("original_kept")
    assert summary["regenerated_kept_after_cap"] == 3


def test_prepare_recovers_locals_keys_from_emitted_rows_or_aborts(regen, tmp_path: Path) -> None:
    t = trace()
    t.turns.append(ParsedTurn(1, FIXED, TurnClass("correct")))
    traces = tmp_path / "traces.jsonl"
    traces.write_text(json.dumps(t.to_json()) + "\n")
    reexec = tmp_path / "reexec"
    regen.write_jsonl(
        reexec / "trace_summary.jsonl",
        [
            {
                "sample_key": t.sample_key,
                "kept": True,
                "first_correct_turn": 1,
                "our_classes": ["error[TypeError]", "correct_fast"],
                "truncated_at": None,
                "drop_reason": None,
            }
        ],
    )
    failed_log = {
        "index": 0,
        "stdout": format_verify_report(TYPE_ERR) + "\n",
        "stderr": "",
        "exception": None,
    }
    regen.write_jsonl(
        reexec / "exec_log.jsonl", [{"sample_key": t.sample_key, "turn_logs": [failed_log]}]
    )
    ns = type("NS", (), {})()
    ns.traces, ns.reexec_dir, ns.regen_dir = str(traces), str(reexec), str(tmp_path / "regen")
    ns.review_flags, ns.flags, ns.repairs, ns.dropped, ns.only, ns.shards = (
        None,
        [],
        True,
        False,
        [],
        1,
    )
    ns.max_changed_lines, ns.max_regen_turns, ns.max_iterations, ns.min_head_ratio = 12, 3, 8, 0.6
    with pytest.raises(SystemExit, match="no locals_keys"):
        regen.prepare(ns)

    keys = ["answer", "context", "kernel_src"]
    exact = regen.render_prefix(
        t, [(FAILED, ExecResult(stdout=failed_log["stdout"], locals_keys=keys))], 8
    )
    regen.write_jsonl(
        reexec / "sft_trajectories.train.jsonl",
        [{"example_id": t.sample_key, "messages": [*exact, {"role": "assistant", "content": "x"}]}],
    )
    regen.prepare(ns)
    [task_row] = regen.load_jsonl(tmp_path / "regen" / "round0" / "tasks.jsonl")
    assert task_row["messages"] == exact
    assert "REPL variables: ['answer', 'context', 'kernel_src']" in task_row["last_feedback"]
