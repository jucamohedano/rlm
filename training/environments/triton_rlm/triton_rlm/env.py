"""Triton kernel generation, trained through the RLM multi-turn harness.

The turn loop, the ```repl``` parsing, the subprocess REPL and the sub-LLM proxy
all come from `rlm_train`; this module adds the task prompt, the in-REPL
verifier (see `triton_rlm.repl`) and the reward.

Reference to keep open while you write this:
    rlm/training/environments/oolong/oolong/env.py     (same shape, 186 lines)

Manifest contract (one JSON object per line, produced by
`scripts/curriculum_from_ops6k.py` or `scripts/ppbhatt_parse.py`):
    {"_task_id": str, "_n_ops": int, "ops": str, "data_source": str, "code": str}

Row contract (what `_build_dataset` must return):
    {"example_id": str,
     "prompt": [{"role": "user", "content": "<placeholder>"}],
     "answer": "",
     "info": json.dumps({"context": ..., "root_prompt": ...})}
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import rlm_train
from datasets import Dataset

from triton_rlm.repl import VerifyingReplBackend
from triton_rlm.verifier import verify_triton

_TASK_INSTRUCTION = (
    "The PyTorch reference module below defines the task (it is also the REPL "
    "`context` variable). Write a correct and fast Triton kernel that computes the "
    "same function. Your submission must be Python source that defines "
    "`def triton_forward(*tensors) -> torch.Tensor` and returns the result. "
    "`tensors` are the module's forward inputs (`get_inputs()`), followed by the "
    "module's parameters in `module.parameters()` order if it has any. "
    "Include any imports your source needs.\n\n"
    "Verification happens in the REPL: whenever a ```repl``` block defines "
    "`triton_forward` and runs without raising, the harness compiles and checks "
    "that block's source, on its own, against the reference (so the block must be "
    "self-contained: imports, kernels and `triton_forward` in one block) and "
    "appends a `[verifier]` report to the output. The verified source is stored in "
    "the REPL variable `kernel_src`. Once the report says `correct: True`, submit "
    "with exactly:\n"
    '```repl\nanswer["content"] = kernel_src\nanswer["ready"] = True\n```\n\n'
    "Example PyTorch reference:\n"
    "```\nimport torch\nclass Model(torch.nn.Module):\n"
    "    def forward(self, x):\n        return torch.relu(x)\n```\n\n"
    "Example valid submission (a real Triton kernel):\n"
    "```\nimport torch\nimport triton\nimport triton.language as tl\n\n"
    "@triton.jit\ndef relu_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):\n"
    "    pid = tl.program_id(0)\n    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
    "    mask = offs < n\n    x = tl.load(x_ptr + offs, mask=mask)\n"
    "    tl.store(out_ptr + offs, tl.maximum(x, 0.0), mask=mask)\n\n"
    "def triton_forward(x):\n    out = torch.empty_like(x)\n    BLOCK = 1024\n"
    "    relu_kernel[(triton.cdiv(x.numel(), BLOCK),)](x, out, x.numel(), BLOCK=BLOCK)\n"
    "    return out\n```\n"
    "Do not just call the PyTorch reference from `triton_forward`: that is not a "
    "Triton kernel and will be scored as non-compiled."
)


def build_root_prompt(task_id: str, ops: str, code: str) -> str:
    # The reference module goes in the visible prompt, not only in the REPL
    # `context`: it is short (KernelBook modules are ~1-2k chars) and the model
    # needs it in the window to write the kernel at all.
    return (
        f"{_TASK_INSTRUCTION}\n\n"
        f"Task {task_id}.\nOperators: {ops}.\n\n"
        f"PyTorch reference module:\n```python\n{code}\n```"
    )


def _build_dataset(*, manifest_path: str, max_tasks: int | None = None, **kwargs: Any) -> Dataset:
    """Turn a curriculum manifest into verifiers dataset rows.

    For every line of the JSONL file, build one row:

      * example_id  -> the stable task id, for logging
      * prompt      -> placeholder; the RLM harness builds the real messages
      * answer      -> leave ""; the scorer does the real work
      * info        -> a JSON STRING (verifiers parses it back into a dict) with:
            "context"       -> the PyTorch reference code (REPL `context`, also verified against)
            "root_prompt"   -> the instruction plus the reference, in the visible messages
            "ops"           -> the operator list, handy for the scorer later

    `max_tasks` caps the number of rows (use it while developing; None = all).
    """
    rows = {"example_id": [], "prompt": [], "answer": [], "info": []}
    with open(manifest_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_tasks is not None and i >= max_tasks:
                break
            ex = json.loads(line)
            task_id = ex.get("_task_id", str(i))
            root_prompt = build_root_prompt(task_id, ex.get("ops", ""), ex["code"])
            rows["example_id"].append(task_id)
            rows["prompt"].append([{"role": "user", "content": "<placeholder>"}])
            rows["answer"].append("")
            rows["info"].append(
                json.dumps(
                    {
                        "context": ex["code"],
                        "root_prompt": root_prompt,
                        "ops": ex.get("ops", ""),
                    }
                )
            )
    return Dataset.from_dict(rows)


def milestone_reward(result: dict[str, Any]) -> float:
    """Reward for a non-empty submission with verifier `result`, in [0, 1].

        reward = 0.10 * [submitted]
               + 0.20 * [compiled / produced an output]
               + 0.40 * [correct]
               + 0.30 * clip(speedup, 0, 3) / 3     (only if correct)

    `speedup` = median reference time / median kernel time, measured only when
    the kernel is correct, via triton.testing.do_bench (warm-up + synchronize).
    See notes/reward-design.md for why correctness dominates and why speedup is
    capped.
    """
    reward = 0.10
    if result.get("compiled"):
        reward += 0.20
    if result.get("correct"):
        reward += 0.40
        speedup = result.get("speedup")
        if speedup is not None:
            reward += 0.30 * min(max(speedup, 0.0), 3.0) / 3.0
    return reward


async def score(info: Any, state: Any, **_kw: Any) -> float:
    """Terminal reward for one rollout: `milestone_reward` of the submission, 0 if none."""
    meta = json.loads(info) if isinstance(info, str) else (info or {})
    submitted = (state.get("rlm_final_answer") or state.get("final_answer") or "").strip()
    if not submitted:
        return 0.0
    reference = str(meta.get("context") or "")
    # blocking subprocess work must not block the asyncio event loop
    result = await asyncio.to_thread(verify_triton, reference, submitted)
    reward = milestone_reward(result)

    # persist the verification outcome so per-rollout metrics/logging can see it
    state["triton_verify"] = {
        "compiled": bool(result.get("compiled")),
        "correct": bool(result.get("correct")),
        "max_diff": result.get("max_diff"),
        "speedup": result.get("speedup"),
        "ref_ms": result.get("ref_ms"),
        "kernel_ms": result.get("kernel_ms"),
        "error": (result.get("error") or "")[:300],
    }
    state["triton_submission_chars"] = len(submitted)

    # roll-out trace dump (for later inspection / a blog; lands on the Modal Volume)
    trace_dir = os.environ.get("TRITON_TRACE_DIR")
    if trace_dir:
        try:
            import pathlib
            import uuid as _uuid

            out = pathlib.Path(trace_dir)
            out.mkdir(parents=True, exist_ok=True)
            rid = state.get("rlm_rollout_id") or state.get("trajectory_id") or _uuid.uuid4().hex
            rec = {
                "rollout_id": rid,
                "example_id": state.get("example_id"),
                "reward": reward,
                "verify": state.get("triton_verify"),
                "submission": submitted[:20000],
                "metrics": {
                    k: state.get(k)
                    for k in (
                        "rlm_iterations",
                        "rlm_repl_calls",
                        "rlm_sub_llm_calls",
                        "rlm_has_final_answer",
                    )
                },
                "completion": state.get("completion"),
            }
            (out / f"{rid}.json").write_text(json.dumps(rec, default=str))
        except Exception:
            pass
    return reward


async def verify_compiled(state: Any) -> int:
    return 1 if (state.get("triton_verify") or {}).get("compiled") else 0


async def verify_correct(state: Any) -> int:
    return 1 if (state.get("triton_verify") or {}).get("correct") else 0


async def verify_speedup(state: Any) -> float:
    v = (state.get("triton_verify") or {}).get("speedup")
    return round(float(v), 3) if v is not None else 0.0


async def verify_max_diff(state: Any) -> float:
    v = (state.get("triton_verify") or {}).get("max_diff")
    return round(float(v), 6) if v is not None else -1.0


class TritonRLMEnv(rlm_train.RLMTrainEnv):
    """RLMTrainEnv whose REPL verifies `triton_forward` blocks against the task's reference."""

    async def setup_state(self, state: Any) -> None:
        await super().setup_state(state)
        state["rlm_backend"] = VerifyingReplBackend(state["rlm_backend"], state["info"]["context"])


def load_environment(*, manifest_path: str, max_iterations: int = 8, **kwargs: Any) -> Any:
    """Wire the dataset and the scorer into the RLM harness.

    Shape (see oolong/env.py::load_environment):

        dataset = _build_dataset(manifest_path=manifest_path, **kwargs)
        return TritonRLMEnv(
            dataset=dataset,
            max_iterations=max_iterations,
            rubric=rlm_train.RLMTrainRubric(correctness=score, weight=1.0),
            sub_sampling_args={"max_tokens": 512},
        )

    Every keyword here is settable from the prime-rl TOML:
        [[orchestrator.train.env]]
        id = "triton-rlm"
        [orchestrator.train.env.args]
        manifest_path = "notes/curriculum/lvl1_seed42.jsonl"
        max_iterations = 8
    """
    dataset = _build_dataset(manifest_path=manifest_path, **kwargs)
    rubric = rlm_train.RLMTrainRubric(correctness=score, weight=1.0)
    # fine-grained monitoring: expose the verifier outcome as logged metrics
    rubric.add_metric(verify_compiled)
    rubric.add_metric(verify_correct)
    rubric.add_metric(verify_speedup)
    rubric.add_metric(verify_max_diff)
    return TritonRLMEnv(
        dataset=dataset,
        max_iterations=max_iterations,
        rubric=rubric,
        sub_sampling_args={"max_tokens": 512},
    )
