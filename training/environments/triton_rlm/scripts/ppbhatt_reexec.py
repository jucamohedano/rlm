"""GPU stage: re-execute parsed traces in the rlm_train worker and emit SFT rows.

Run inside a torch+triton container, importing `rlm_train` and `triton_rlm` from THIS
checkout (the worker protocol here differs from older `rlm_train` installs):

    PYTHONPATH=<checkout>/training/src:<checkout>/training/environments/triton_rlm \
      python scripts/ppbhatt_reexec.py parsed/traces.jsonl out_dir [--limit N] [--only KEY ...]

The script refuses to start if either package resolves elsewhere.

Per trace, one worker subprocess (persistent namespace across turns, exactly as
in a rollout) wrapped in `VerifyingReplBackend`, so each turn's ```repl``` block
is executed, then verified against `pytorch_code` and the report lands in the
REPL output. Messages are built with the same helpers `RLMTrainEnv` uses, so the
rows are byte-identical to what the harness would show a policy.

Writes to out_dir:
    exec_log.jsonl          every executed turn: our ExecResult + verifier report + class
    trace_summary.jsonl     one row per trace: the trajectory-level rollup (kept_prefix_len,
                            first_correct_turn, repair, final_class, truncation)
    summary.json            aggregate counts over the run (yield, repair share, drop reasons)
    reexec_report.txt       per trace: recorded vs our classes, decision, first correct turn
    sft_rows.<split>.jsonl  one row per assistant turn (assistant_loss_only_last: true)
    sft_trajectories.<split>.jsonl  one row per trajectory (all assistant turns supervised)
    manifest.jsonl          surviving tasks, env manifest contract

A trajectory is dropped when `decide` says so (rule 1 mismatch or ends on a
failure). Kept trajectories end with the correct turn, then ONE fixed submit
idiom turn; nothing else is added.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import dataclasses
import json
from pathlib import Path
from typing import Any

import triton_rlm
from rlm.utils.parsing import find_code_blocks
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from triton_rlm.env import build_root_prompt, milestone_reward
from triton_rlm.repl import VerifyingReplBackend
from triton_rlm.trace_translate import (
    DATASET_ID,
    SUBMIT_IDIOM,
    Decision,
    ParsedTrace,
    TurnClass,
    classify_ours,
    decide,
    repair_diff_lines,
    source_harness_markers,
)

import rlm_train
from rlm_train.env import _format_repl_outputs, _pack_exec
from rlm_train.repl.base import ExecResult
from rlm_train.repl.subprocess import SubprocessReplBackend

# The worker only contacts the proxy for llm_query(); the traces never call it.
DUMMY_PROXY_URL = "http://127.0.0.1:9"

TRAINING_DIR = Path(__file__).resolve().parents[3]


def check_source_tree() -> None:
    """Both packages must come from this checkout: the worker subprocess is spawned as
    `python -m rlm_train.worker`, so a stale editable install elsewhere silently changes
    the exec protocol (no `exception` field, no `set_local`, no linecache-registered
    compile) and every turn misclassifies."""
    expected = {
        "rlm_train": TRAINING_DIR / "src" / "rlm_train",
        "triton_rlm": TRAINING_DIR / "environments" / "triton_rlm" / "triton_rlm",
    }
    bad = []
    for name, pkg in (("rlm_train", rlm_train), ("triton_rlm", triton_rlm)):
        actual = Path(pkg.__file__).resolve().parent
        if actual != expected[name]:
            bad.append(f"{name} imported from {actual}, expected {expected[name]}")
    if "exception" not in {f.name for f in dataclasses.fields(ExecResult)}:
        bad.append("rlm_train.repl.base.ExecResult has no `exception` field (stale rlm_train)")
    if bad:
        raise RuntimeError(
            "wrong source tree:\n  "
            + "\n  ".join(bad)
            + f"\nrun with PYTHONPATH={TRAINING_DIR / 'src'}:{TRAINING_DIR / 'environments' / 'triton_rlm'}"
        )


def fence(code: str) -> str:
    return f"```repl\n{code}\n```"


async def reexec_trace(
    trace: ParsedTrace, *, max_iterations: int, min_head_ratio: float
) -> dict[str, Any]:
    recorded = [t.recorded for t in trace.turns]
    root_prompt = build_root_prompt(trace.sample_key, ", ".join(trace.ops), trace.pytorch_code)
    history: list[dict[str, str]] = list(
        build_rlm_system_prompt(
            system_prompt=RLM_SYSTEM_PROMPT,
            query_metadata=QueryMetadata(trace.pytorch_code),
            custom_tools=None,
            root_prompt=root_prompt,
            orchestrator=True,
        )
    )
    history.append(
        build_user_prompt(root_prompt=root_prompt, iteration=0, max_iterations=max_iterations)
    )

    backend = VerifyingReplBackend(SubprocessReplBackend(), trace.pytorch_code)
    await backend.start(DUMMY_PROXY_URL, f"ppbhatt_{trace.sample_key}", 1)
    await backend.load_context(trace.pytorch_code)

    ours: list[TurnClass] = []
    turn_logs: list[dict[str, Any]] = []
    history_end_after_turn: list[int] = []  # len(history) after turn i's REPL output
    decision = Decision("execute_next", None, "")
    try:
        for turn in trace.turns:
            code = turn.code
            if code is None or find_code_blocks(fence(code)) != [code]:
                ours.append(TurnClass("error", "NoCode", "no single <triton> block"))
                turn_logs.append({"index": turn.index, "ours": ours[-1].__dict__, "skipped": True})
                decision = decide(recorded, ours, min_head_ratio)
                break
            backend.last_verify = None
            result = await backend.execute(code)
            verify = backend.last_verify
            ours.append(classify_ours(result.exception, verify))
            history.append({"role": "assistant", "content": fence(code)})
            history.extend(_format_repl_outputs([_pack_exec(code, result)]))
            history_end_after_turn.append(len(history))
            turn_logs.append(
                {
                    "index": turn.index,
                    "recorded": turn.recorded.__dict__,
                    "ours": ours[-1].__dict__,
                    "exception": result.exception,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "execution_time": result.execution_time,
                    "locals_keys": result.locals_keys,
                    "verify": verify,
                }
            )
            decision = decide(recorded, ours, min_head_ratio)
            if decision.action != "execute_next":
                break
            history.append(
                build_user_prompt(
                    root_prompt=root_prompt, iteration=len(ours), max_iterations=max_iterations
                )
            )
    finally:
        await backend.stop()

    out: dict[str, Any] = {
        "sample_key": trace.sample_key,
        "cluster": trace.cluster,
        "split": trace.split,
        "recorded": [c.short() for c in recorded],
        "ours": [c.short() for c in ours],
        "decision": decision.__dict__,
        "turn_logs": turn_logs,
        "messages": None,
        "reward": None,
        "first_correct_turn": None,
        "repair": False,
        "repair_diff_lines": [],
    }
    if decision.action != "keep":
        return out
    keep_end = decision.keep_end
    assert keep_end is not None
    kept = history[: history_end_after_turn[keep_end]]
    leaks = [
        f"message {i} ({msg['role']}): {m!r}"
        for i, msg in enumerate(kept)
        for m in source_harness_markers(msg["content"])
    ]
    if leaks:
        out["decision"] = dataclasses.asdict(
            Decision("drop", None, f"source-harness marker at {'; '.join(leaks[:3])}")
        )
        return out
    kept.append(
        build_user_prompt(
            root_prompt=root_prompt, iteration=keep_end + 1, max_iterations=max_iterations
        )
    )
    kept.append({"role": "assistant", "content": fence(SUBMIT_IDIOM)})
    out["messages"] = kept
    out["reward"] = milestone_reward(turn_logs[keep_end]["verify"])
    first_correct = next(i for i, c in enumerate(ours) if c.kind.startswith("correct"))
    out["first_correct_turn"] = first_correct
    out["repair_diff_lines"] = repair_diff_lines([t.code for t in trace.turns], ours, first_correct)
    out["repair"] = any(n > 0 for n in out["repair_diff_lines"])
    return out


def trace_summary(trace: ParsedTrace, res: dict[str, Any]) -> dict[str, Any]:
    """Trajectory-level rollup: the unit the training decision is made on."""
    d = res["decision"]
    kept = d["action"] == "keep"
    keep_end = d["keep_end"] if kept else None
    ours = res["ours"]
    return {
        "sample_key": res["sample_key"],
        "cluster": res["cluster"],
        "split": res["split"],
        "n_source_turns": len(trace.turns),
        "n_executed_turns": len(ours),
        "recorded_classes": res["recorded"],
        "our_classes": ours,
        "kept": kept,
        "kept_prefix_len": keep_end + 1 if kept else 0,
        "first_correct_turn": res["first_correct_turn"],
        "repair": res["repair"],
        "repair_diff_lines": res["repair_diff_lines"],
        "final_class": ours[keep_end] if kept else (ours[-1] if ours else None),
        "truncated_at": None if kept else len(ours) - 1,
        "drop_reason": None if kept else d["reason"],
        "reward": res["reward"],
    }


def aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    kept = [s for s in summaries if s["kept"]]
    repair = [s for s in kept if s["repair"]]
    our_errors = collections.Counter(
        c for s in summaries for c in s["our_classes"] if c.startswith("error[")
    )
    return {
        "traces_reexecuted": len(summaries),
        "traces_with_verified_kernel": len(kept),
        "trajectories_with_repair": len(repair),
        "repair_share": len(repair) / len(kept) if kept else None,
        "kept_kernel_turns": sum(s["kept_prefix_len"] for s in kept),
        "kept_ending_correct_fast": sum(s["final_class"] == "correct_fast" for s in kept),
        "distinct_tasks_kept": len({s["cluster"] for s in kept}),
        "first_correct_turn_hist": dict(collections.Counter(s["first_correct_turn"] for s in kept)),
        "drop_reasons": dict(
            collections.Counter(
                s["drop_reason"].split(" at ")[0] for s in summaries if not s["kept"]
            )
        ),
        "our_error_classes": dict(our_errors.most_common()),
        "per_split": dict(collections.Counter(s["split"] for s in kept)),
    }


def sft_rows(res: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages = res["messages"]
    meta = {
        "task_id": res["sample_key"],
        "cluster": res["cluster"],
        "split": res["split"],
        "source_dataset": DATASET_ID,
        "recorded_classes": res["recorded"],
        "our_classes": res["ours"],
        "decision": res["decision"]["reason"],
        "first_correct_turn": res["first_correct_turn"],
        "repair": res["repair"],
        "repair_diff_lines": res["repair_diff_lines"],
        "final_verify": {
            k: res["turn_logs"][res["decision"]["keep_end"]]["verify"].get(k)
            for k in ("compiled", "correct", "speedup", "max_diff")
        },
    }
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    per_turn = [
        {
            "example_id": res["sample_key"],
            "turn": k + 1,
            "reward": res["reward"],
            "messages": messages[: idx + 1],
            "assistant_loss_only_last": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "triton_rlm": meta,
        }
        for k, idx in enumerate(assistant_idx)
    ]
    trajectory = {
        "example_id": res["sample_key"],
        "turn": len(assistant_idx),
        "reward": res["reward"],
        "messages": messages,
        "assistant_loss_only_last": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "triton_rlm": meta,
    }
    return per_turn, trajectory


async def main_async(args: argparse.Namespace) -> None:
    traces = [
        ParsedTrace.from_json(json.loads(line))
        for line in Path(args.traces).read_text(encoding="utf-8").splitlines()
    ]
    if args.only:
        traces = [t for t in traces if t.sample_key in set(args.only)]
    if args.limit:
        traces = traces[: args.limit]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    lines: list[str] = []
    with (
        (out / "exec_log.jsonl").open("w", encoding="utf-8") as log,
        (out / "trace_summary.jsonl").open("w", encoding="utf-8") as summ,
    ):
        for n, trace in enumerate(traces, 1):
            res = await reexec_trace(
                trace, max_iterations=args.max_iterations, min_head_ratio=args.min_head_ratio
            )
            results.append(res)
            log.write(json.dumps({k: v for k, v in res.items() if k != "messages"}) + "\n")
            log.flush()
            summaries.append(trace_summary(trace, res))
            summ.write(json.dumps(summaries[-1]) + "\n")
            summ.flush()
            d = res["decision"]
            verdict = f"KEEP turns 0..{d['keep_end']}" if d["action"] == "keep" else "DROP"
            line = (
                f"{trace.sample_key:18s} recorded: {' -> '.join(res['recorded'])}\n"
                f"{'':18s} ours:     {' -> '.join(res['ours'])}\n"
                f"{'':18s} {verdict} ({d['reason']})"
                + (
                    f"; first correct turn {res['first_correct_turn']}; repair={res['repair']}"
                    if d["action"] == "keep"
                    else ""
                )
            )
            lines.append(line)
            print(f"[{n}/{len(traces)}] {line}", flush=True)

    kept = [r for r in results if r["decision"]["action"] == "keep"]
    by_split: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = (
        collections.defaultdict(lambda: ([], []))
    )
    for r in kept:
        rows, traj = sft_rows(r)
        by_split[r["split"]][0].extend(rows)
        by_split[r["split"]][1].append(traj)
    for split, (rows, trajs) in by_split.items():
        with (out / f"sft_rows.{split}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
        with (out / f"sft_trajectories.{split}.jsonl").open("w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in trajs)
    kept_keys = {r["sample_key"] for r in kept}
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            if t.sample_key in kept_keys:
                f.write(
                    json.dumps(
                        {
                            "_task_id": t.sample_key,
                            "_n_ops": len(t.ops),
                            "ops": ", ".join(t.ops),
                            "data_source": DATASET_ID,
                            "code": t.pytorch_code,
                            "split": t.split,
                            "cluster": t.cluster,
                        }
                    )
                    + "\n"
                )

    agg = aggregate(summaries)
    (out / "summary.json").write_text(json.dumps(agg, indent=2) + "\n", encoding="utf-8")
    summary = [f"{k}: {v}" for k, v in agg.items()]
    report = "\n\n".join(lines) + "\n\n" + "\n".join(summary) + "\n"
    (out / "reexec_report.txt").write_text(report, encoding="utf-8")
    print("\n".join(summary))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("out_dir")
    ap.add_argument("--max-iterations", type=int, default=8, help="Turn k/N marker, as in the env")
    ap.add_argument(
        "--min-head-ratio", type=float, default=0.6, help="error message head similarity for rule 1"
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None, help="sample_keys to process")
    args = ap.parse_args()
    check_source_tree()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
