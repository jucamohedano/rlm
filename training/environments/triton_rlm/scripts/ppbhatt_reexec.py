"""GPU stage: re-execute parsed traces in the rlm_train worker and emit SFT rows.

Run inside a torch+triton container with this repo installed:

    uv run python scripts/ppbhatt_reexec.py parsed/traces.jsonl out_dir [--limit N] [--only KEY]

Per trace, one worker subprocess (persistent namespace across turns, exactly as
in a rollout) wrapped in `VerifyingReplBackend`, so each turn's ```repl``` block
is executed, then verified against `pytorch_code` and the report lands in the
REPL output. Messages are built with the same helpers `RLMTrainEnv` uses, so the
rows are byte-identical to what the harness would show a policy.

Writes to out_dir:
    exec_log.jsonl          every executed turn: our ExecResult + verifier report + class
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
import json
from pathlib import Path
from typing import Any

from rlm_train.env import _format_repl_outputs, _pack_exec
from rlm_train.repl.subprocess import SubprocessReplBackend
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
)

from rlm.utils.parsing import find_code_blocks
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)

# The worker only contacts the proxy for llm_query(); the traces never call it.
DUMMY_PROXY_URL = "http://127.0.0.1:9"


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
    }
    if decision.action != "keep":
        return out
    keep_end = decision.keep_end
    assert keep_end is not None
    kept = history[: history_end_after_turn[keep_end]]
    kept.append(
        build_user_prompt(
            root_prompt=root_prompt, iteration=keep_end + 1, max_iterations=max_iterations
        )
    )
    kept.append({"role": "assistant", "content": fence(SUBMIT_IDIOM)})
    out["messages"] = kept
    out["reward"] = milestone_reward(turn_logs[keep_end]["verify"])
    out["first_correct_turn"] = next(i for i, c in enumerate(ours) if c.kind.startswith("correct"))
    out["repair"] = any(c.kind in ("error", "incorrect") for c in ours[:keep_end])
    return out


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
    lines: list[str] = []
    with (out / "exec_log.jsonl").open("w", encoding="utf-8") as log:
        for n, trace in enumerate(traces, 1):
            res = await reexec_trace(
                trace, max_iterations=args.max_iterations, min_head_ratio=args.min_head_ratio
            )
            results.append(res)
            log.write(json.dumps({k: v for k, v in res.items() if k != "messages"}) + "\n")
            log.flush()
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

    n_kernel_turns = sum(r["decision"]["keep_end"] + 1 for r in kept)
    summary = [
        f"traces re-executed: {len(results)}",
        f"traces kept: {len(kept)}  dropped: {len(results) - len(kept)}",
        f"kept kernel turns: {n_kernel_turns}  (+{len(kept)} submit turns)",
        f"kept with error/incorrect -> recovery: {sum(r['repair'] for r in kept)}",
        f"kept ending correct_fast: {sum(r['ours'][r['decision']['keep_end']] == 'correct_fast' for r in kept)}",
        f"distinct tasks (dedupe clusters) kept: {len({r['cluster'] for r in kept})}",
        f"drop reasons: {dict(collections.Counter(r['decision']['reason'].split(' at ')[0] for r in results if r['decision']['action'] != 'keep'))}",
        f"per split: {dict(collections.Counter(r['split'] for r in kept))}",
    ]
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
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
