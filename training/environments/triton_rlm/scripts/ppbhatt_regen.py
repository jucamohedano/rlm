"""Regenerate flagged / dropped ppbhatt transitions against OUR feedback, in rounds.

The generator is external (a Devin session on a chosen model, run by the operator);
this script only prepares its input, executes and judges its output, and assembles the
mix. It never calls a model.

    prepare  (CPU)  parsed/traces.jsonl reexec_dir regen_dir [--review-flags ... --flags ...]
                    [--repairs] [--dropped] [--only KEY ...] [--shards N]
        selection: kept traces whose review flags (default THEIR_ONLY_TOKENS,
        OUR_LINE_UNTOUCHED) sit on a failed turn before the first correct one; or with
        --repairs every kept repair (its correct turn is regenerated); plus with
        --dropped every dropped trace, from its truncation turn.
        A log without `locals_keys` (67a4420) gets the `REPL variables:` line read back
        from the emitted trajectory (`ppbhatt_review.recover_locals_keys`); a prefix turn
        that cannot be rendered exactly aborts.
        -> regen_dir/round0/{tasks.jsonl, shard*.jsonl, SESSION_PROMPT.md}
           prints N tasks and the token totals, so the cost is known before anything runs.
    execute  (GPU)  parsed/traces.jsonl regen_dir --round r
        reads round<r>/tasks.jsonl + every round<r>/actions*.jsonl ({sample_key, round,
        code}) in one process (all shards, one container), replays each task's prefix in
        a fresh worker (persistent namespace, as in a rollout), runs the regenerated block,
        verifies it, applies `trace_regen.judge`
        -> round<r>/results.jsonl, round<r>/sft_{rows,trajectories}.regen.jsonl (finished),
           round<r+1>/{tasks.jsonl, shard*.jsonl, SESSION_PROMPT.md} (continued).
    audit    (CPU)  regen_dir --round r
        per task: what the generator was shown vs what our worker rendered on replay
        (`trace_regen.prompt_deviation` / `feedback_status`). `prompt_deviation`: the
        normalisations that make the shown REPL outputs equal to the replayed ones
        (missing_repl_variables_line, repl_filename, float_values), `[]` if byte-identical,
        `other` if none suffice, with the first differing message index. `feedback`:
        exact | cosmetic | stale | unverifiable -- the last REPL output the generator
        answered vs the replayed one for the same code, i.e. the error class and message
        it acted on; stale (or `other`, or a prefix class mismatch) disqualifies.
        -> round<r>/audit.jsonl, read by collect.
    collect  (CPU)  reexec_dir regen_dir [--max-share 0.3] [--seed 0]
                    [--exclude-original-repairs] [--truncated KEY ...]
        -> regen_dir/mix/sft_{rows,trajectories}.<split>.jsonl + summary.json: the
           original trajectories (first-shot successes only with
           --exclude-original-repairs) plus a seeded uniform sample of the regenerated
           ones capped at --max-share of the combined train set; val stays original-only.
           Regenerated rows carry `prompt_deviation` / `feedback_deviation` from the
           audit. `summary.json.excluded` names every regeneration task that is in
           neither the originals nor the kept regenerations, with its reason
           (no_clean_repair, truncated_history for --truncated keys, audit_disqualified,
           regenerated_val_split, resampled_out, original_kept).

A task is one trace: `prefix` = the turns kept from the source (all failed under our
verifier, the last one being the turn whose feedback the regenerated block answers),
plus any earlier regenerated turns; `messages` = the conversation exactly as
`TritonRLMEnv` would send it, ending with the `Turn k/N:` user prompt. Rows carry
`provenance: "regenerated"`, the per-turn provenance list and `regen_turns`.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import random
from pathlib import Path
from typing import Any

from ppbhatt_reexec import DUMMY_PROXY_URL, check_source_tree, fence
from ppbhatt_review import recover_locals_keys
from rlm.utils.parsing import find_code_blocks
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from triton_rlm.env import build_root_prompt, milestone_reward
from triton_rlm.repl import VerifyingReplBackend
from triton_rlm.trace_regen import (
    approx_tokens,
    cap_regenerated,
    feedback_status,
    judge,
    prompt_deviation,
    regen_start,
    strip_fence,
)
from triton_rlm.trace_translate import (
    DATASET_ID,
    SUBMIT_IDIOM,
    ParsedTrace,
    TurnClass,
    classify_ours,
    repair_diff_lines,
    source_harness_markers,
)

from rlm_train.env import _format_repl_outputs, _pack_exec
from rlm_train.repl.base import ExecResult
from rlm_train.repl.subprocess import SubprocessReplBackend

SESSION_PROMPT = """\
# Regenerate the next REPL turn ({n} tasks in {shard})

Each line of `{shard}` is a JSON task. `messages` is a complete chat transcript from a
Triton kernel-writing REPL environment: the system prompt, the task, the assistant's
earlier ```repl blocks and the REPL output each one produced (including a `[verifier]`
report), ending with a user message `Turn k/N:`. Write the assistant's reply for exactly
that turn.

Append one line per task to `{actions}`:

    {{"sample_key": "<task.sample_key>", "round": {round}, "code": "<python source of the block>"}}

`code` is the body of a single ```repl block (no fence, no prose). Rules, enforced by the
harness that executes your block:

- Answer the LAST REPL output only. Make the smallest edit to the previous block that
  addresses it; a block changing more than {max_changed_lines} lines, or nothing, is
  rejected. Keep the previous block's names, structure, block sizes and imports.
- The block stays self-contained (imports, kernels, `triton_forward`) exactly as the
  system prompt requires; do not add comments narrating the fix.
- No `answer[...]` lines: the harness appends the submit turn itself once the report
  says `correct: True`.
- Do not execute anything and do not write REPL output; the harness runs and verifies
  your block.
"""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def load_traces(path: Path) -> dict[str, ParsedTrace]:
    traces = [ParsedTrace.from_json(json.loads(line)) for line in path.read_text().splitlines()]
    return {t.sample_key: t for t in traces}


def render_prefix(
    trace: ParsedTrace, executed: list[tuple[str, ExecResult]], max_iterations: int
) -> list[dict[str, str]]:
    """History as `ppbhatt_reexec.reexec_trace` builds it, ending with the user prompt for
    the turn after the last executed block."""
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
    for i, (code, result) in enumerate(executed):
        history.append({"role": "assistant", "content": fence(code)})
        history.extend(_format_repl_outputs([_pack_exec(code, result)]))
        history.append(
            build_user_prompt(
                root_prompt=root_prompt, iteration=i + 1, max_iterations=max_iterations
            )
        )
    return history


def log_result(log: dict[str, Any]) -> ExecResult:
    return ExecResult(
        stdout=log["stdout"],
        stderr=log["stderr"],
        exception=log["exception"],
        locals_keys=log["locals_keys"],
    )


def flagged_turns(record: dict[str, Any] | None, names: set[str]) -> list[int]:
    if record is None:
        return []
    return sorted(
        {
            int(fl.split(":")[0][1:])
            for fl in record["flags"]
            if fl.startswith("t") and fl.split(":")[1] in names
        }
    )


def make_task(
    trace: ParsedTrace,
    prefix: list[dict[str, Any]],
    messages: list[dict[str, str]],
    *,
    round_no: int,
    regen_turns: int,
    limits: dict[str, Any],
    start_reason: str,
) -> dict[str, Any]:
    return {
        "sample_key": trace.sample_key,
        "cluster": trace.cluster,
        "split": trace.split,
        "round": round_no,
        "prefix": prefix,
        "regen_turns": regen_turns,
        "messages": messages,
        "last_feedback": messages[-2]["content"],
        "limits": limits,
        "start_reason": start_reason,
    }


def write_round(regen_dir: Path, round_no: int, tasks: list[dict[str, Any]], shards: int) -> None:
    rd = regen_dir / f"round{round_no}"
    write_jsonl(rd / "tasks.jsonl", tasks)
    shards = max(1, min(shards, len(tasks)))
    for i in range(shards):
        part = tasks[i::shards]
        shard = f"shard{i}.jsonl"
        write_jsonl(rd / shard, part)
        (rd / f"SESSION_PROMPT.shard{i}.md").write_text(
            SESSION_PROMPT.format(
                n=len(part),
                shard=shard,
                actions=f"actions.shard{i}.jsonl",
                round=round_no,
                max_changed_lines=tasks[0]["limits"]["max_changed_lines"],
            ),
            encoding="utf-8",
        )
    in_tok = sum(approx_tokens(t["messages"]) for t in tasks)
    out_tok = sum(len(t["prefix"][-1]["code"]) // 4 for t in tasks)
    reasons = collections.Counter(t["start_reason"].split(":")[0] for t in tasks)
    print(
        f"round {round_no}: {len(tasks)} tasks in {shards} shard(s) -> {rd}\n"
        f"  input tokens ~{in_tok} (mean {in_tok // max(1, len(tasks))}/task), "
        f"expected output tokens ~{out_tok}\n"
        f"  by selection: {dict(reasons)}"
    )


def prepare(args: argparse.Namespace) -> None:
    traces = load_traces(Path(args.traces))
    reexec = Path(args.reexec_dir)
    summaries = {s["sample_key"]: s for s in load_jsonl(reexec / "trace_summary.jsonl")}
    emitted = {
        t["example_id"]: t["messages"]
        for p in sorted(reexec.glob("sft_trajectories.*.jsonl"))
        for t in load_jsonl(p)
    }
    logs = {
        r["sample_key"]: recover_locals_keys(r["turn_logs"], emitted.get(r["sample_key"]))
        for r in load_jsonl(reexec / "exec_log.jsonl")
    }
    review = (
        {r["sample_key"]: r for r in load_jsonl(Path(args.review_flags))}
        if args.review_flags
        else {}
    )
    names = set(args.flags)
    limits = {
        "max_changed_lines": args.max_changed_lines,
        "max_regen_turns": args.max_regen_turns,
        "max_iterations": args.max_iterations,
        "min_head_ratio": args.min_head_ratio,
    }
    tasks: list[dict[str, Any]] = []
    skipped: collections.Counter[str] = collections.Counter()
    for key, summary in summaries.items():
        if args.only and key not in set(args.only):
            continue
        if not summary["kept"] and not args.dropped:
            skipped["dropped trace, --dropped not given"] += 1
            continue
        k, reason = regen_start(
            summary, flagged_turns(review.get(key), names), all_repairs=args.repairs
        )
        if k is None:
            skipped[reason] += 1
            continue
        turn_logs = [t for t in logs[key] if not t.get("skipped")]
        if len(turn_logs) <= k:
            skipped["prefix turn without a block (NoCode)"] += 1
            continue
        if k + 1 >= args.max_iterations - 1:
            skipped["no turn budget left after the prefix"] += 1
            continue
        trace = traces[key]
        executed: list[tuple[str, ExecResult]] = []
        prefix: list[dict[str, Any]] = []
        for log in turn_logs[: k + 1]:
            code = trace.turns[log["index"]].code
            assert code is not None
            if "locals_keys" not in log:
                raise SystemExit(
                    f"{key} turn {log['index']}: exec_log has no locals_keys and the turn is not "
                    "in an emitted trajectory; the prompt cannot be rendered exactly"
                )
            executed.append((code, log_result(log)))
            prefix.append(
                {
                    "code": code,
                    "provenance": "teacher",
                    "ours": summary["our_classes"][log["index"]],
                    "source_turn": log["index"],
                }
            )
        tasks.append(
            make_task(
                trace,
                prefix,
                render_prefix(trace, executed, args.max_iterations),
                round_no=0,
                regen_turns=0,
                limits=limits,
                start_reason=reason,
            )
        )
    print(f"skipped: {dict(skipped)}")
    if not tasks:
        raise SystemExit("no tasks selected")
    write_round(Path(args.regen_dir), 0, tasks, args.shards)


def load_actions(rd: Path, round_no: int) -> dict[str, str]:
    actions: dict[str, str] = {}
    for path in sorted(rd.glob("actions*.jsonl")):
        for row in load_jsonl(path):
            if row["round"] != round_no:
                raise ValueError(f"{path}: action for round {row['round']} in round {round_no}")
            if row["sample_key"] in actions:
                raise ValueError(f"{path}: duplicate action for {row['sample_key']}")
            actions[row["sample_key"]] = strip_fence(row["code"])
    if not actions:
        raise SystemExit(f"no actions*.jsonl under {rd}")
    return actions


def class_of(short: str) -> TurnClass:
    if short.startswith("error["):
        return TurnClass("error", short[6:-1])
    return TurnClass(short.split("[")[0])


async def execute_task(task: dict[str, Any], code: str, trace: ParsedTrace) -> dict[str, Any]:
    limits = task["limits"]
    max_iterations: int = limits["max_iterations"]
    out: dict[str, Any] = {
        "sample_key": task["sample_key"],
        "round": task["round"],
        "action": "drop",
        "reason": "",
        "new_class": None,
        "prefix_replay": [],
        "prompt_render_mismatch": None,
        "turn_log": None,
        "trajectory": None,
        "next_task": None,
    }
    # the env executes what `find_code_blocks` extracts, i.e. the stripped block
    code = code.strip()
    if find_code_blocks(fence(code)) != [code]:
        out["reason"] = "regenerated action is not a single block"
        return out

    backend = VerifyingReplBackend(SubprocessReplBackend(), trace.pytorch_code)
    await backend.start(DUMMY_PROXY_URL, f"ppbhatt_regen_{trace.sample_key}_{task['round']}", 1)
    await backend.load_context(trace.pytorch_code)
    executed: list[tuple[str, ExecResult]] = []
    classes: list[TurnClass] = []
    try:
        for p in task["prefix"]:
            backend.last_verify = None
            result = await backend.execute(p["code"])
            cls = classify_ours(result.exception, backend.last_verify)
            out["prefix_replay"].append(cls.short())
            if class_of(cls.short()) != class_of(p["ours"]):
                out["reason"] = (
                    f"prefix nondeterministic: turn {len(executed)} was {p['ours']}, "
                    f"replayed {cls.short()}"
                )
                return out
            executed.append((p["code"], result))
            classes.append(cls)
        backend.last_verify = None
        result = await backend.execute(code)
        verify = backend.last_verify
    finally:
        await backend.stop()

    new = classify_ours(result.exception, verify)
    out["new_class"] = new.short()
    out["turn_log"] = {
        "exception": result.exception,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "execution_time": result.execution_time,
        "locals_keys": result.locals_keys,
        "verify": verify,
    }
    history = render_prefix(trace, executed, max_iterations)
    out["prompt_render_mismatch"] = history != task["messages"]
    verdict = judge(
        classes[-1],
        new,
        task["prefix"][-1]["code"],
        code,
        kernel_turns=len(executed) + 1,
        regen_turns=task["regen_turns"] + 1,
        max_kernel_turns=max_iterations - 1,
        max_regen_turns=limits["max_regen_turns"],
        max_changed_lines=limits["max_changed_lines"],
        min_head_ratio=limits["min_head_ratio"],
    )
    out["action"], out["reason"] = verdict.action, verdict.reason
    if verdict.action == "drop":
        return out

    executed.append((code, result))
    classes.append(new)
    prefix = [
        *task["prefix"],
        {"code": code, "provenance": "regenerated", "ours": new.short(), "source_turn": None},
    ]
    history = render_prefix(trace, executed, max_iterations)
    leaks = [
        f"message {i} ({m['role']}): {mk!r}"
        for i, m in enumerate(history)
        for mk in source_harness_markers(m["content"])
    ]
    if leaks:
        out["action"], out["reason"] = "drop", f"source-harness marker at {'; '.join(leaks[:3])}"
        return out
    if verdict.action == "continue":
        out["next_task"] = make_task(
            trace,
            prefix,
            history,
            round_no=task["round"] + 1,
            regen_turns=task["regen_turns"] + 1,
            limits=limits,
            start_reason=task["start_reason"],
        )
        return out

    assert verify is not None
    messages = [*history, {"role": "assistant", "content": fence(SUBMIT_IDIOM)}]
    codes = [p["code"] for p in prefix]
    first_correct = len(codes) - 1
    meta = {
        "task_id": trace.sample_key,
        "cluster": trace.cluster,
        "split": trace.split,
        "source_dataset": DATASET_ID,
        "provenance": "regenerated",
        "turn_provenance": [p["provenance"] for p in prefix],
        "regen_turns": task["regen_turns"] + 1,
        "start_reason": task["start_reason"],
        "our_classes": [c.short() for c in classes],
        "first_correct_turn": first_correct,
        "repair": True,
        "repair_diff_lines": repair_diff_lines(codes, classes, first_correct),
        "final_verify": {k: verify.get(k) for k in ("compiled", "correct", "speedup", "max_diff")},
    }
    reward = milestone_reward(verify)
    assistant_idx = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    out["trajectory"] = {
        "example_id": trace.sample_key,
        "turn": len(assistant_idx),
        "reward": reward,
        "messages": messages,
        "assistant_loss_only_last": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "triton_rlm": meta,
    }
    out["rows"] = [
        {
            "example_id": trace.sample_key,
            "turn": k + 1,
            "reward": reward,
            "messages": messages[: idx + 1],
            "assistant_loss_only_last": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "triton_rlm": meta,
        }
        for k, idx in enumerate(assistant_idx)
    ]
    return out


async def execute_async(args: argparse.Namespace) -> None:
    traces = load_traces(Path(args.traces))
    regen = Path(args.regen_dir)
    rd = regen / f"round{args.round}"
    tasks = load_jsonl(rd / "tasks.jsonl")
    actions = load_actions(rd, args.round)
    results: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    next_tasks: list[dict[str, Any]] = []
    with (rd / "results.jsonl").open("w", encoding="utf-8") as log:
        for n, task in enumerate(tasks, 1):
            key = task["sample_key"]
            if key not in actions:
                res: dict[str, Any] = {
                    "sample_key": key,
                    "round": args.round,
                    "action": "drop",
                    "reason": "no action",
                }
            else:
                res = await execute_task(task, actions[key], traces[key])
            if res.get("trajectory"):
                trajectories.append(res.pop("trajectory"))
                rows.extend(res.pop("rows"))
            if res.get("next_task"):
                next_tasks.append(res.pop("next_task"))
            results.append(res)
            log.write(json.dumps(res) + "\n")
            log.flush()
            print(f"[{n}/{len(tasks)}] {key:18s} {res['action']:8s} {res['reason']}", flush=True)
    write_jsonl(rd / "sft_trajectories.regen.jsonl", trajectories)
    write_jsonl(rd / "sft_rows.regen.jsonl", rows)
    outcome = collections.Counter(r["action"] for r in results)
    mismatch = sum(1 for r in results if r.get("prompt_render_mismatch"))
    print(
        f"round {args.round}: {dict(outcome)}; finished {len(trajectories)} trajectories; "
        f"prompt/emitted prefix render mismatch on {mismatch} tasks (classify with `audit`)"
    )
    if next_tasks:
        write_round(regen, args.round + 1, next_tasks, args.shards)


def execute(args: argparse.Namespace) -> None:
    check_source_tree()
    asyncio.run(execute_async(args))


def replayed_renders(regen: Path, round_no: int) -> dict[str, list[dict[str, str]]]:
    """Per task, the history our worker rendered on replay: the finished trajectory's
    messages, or the continued task's messages in the next round (both are built by
    `render_prefix` from live `ExecResult`s in `execute_task`)."""
    rd = regen / f"round{round_no}"
    renders = {
        t["example_id"]: t["messages"] for t in load_jsonl(rd / "sft_trajectories.regen.jsonl")
    }
    nxt = regen / f"round{round_no + 1}" / "tasks.jsonl"
    if nxt.exists():
        renders |= {t["sample_key"]: t["messages"] for t in load_jsonl(nxt)}
    return renders


def audit_task(
    task: dict[str, Any], result: dict[str, Any], rendered: list[dict[str, str]] | None
) -> dict[str, Any]:
    shown: list[dict[str, str]] = task["messages"]
    replay: list[str] = result.get("prefix_replay", [])
    classes_match = len(replay) == len(task["prefix"]) and all(
        class_of(r) == class_of(p["ours"]) for r, p in zip(replay, task["prefix"], strict=True)
    )
    if rendered is None:
        deviation, first = None, None
    else:
        deviation, first = prompt_deviation(shown, rendered[: len(shown)])
    feedback, feedback_kinds = feedback_status(shown, rendered)
    return {
        "sample_key": task["sample_key"],
        "round": task["round"],
        "action": result["action"],
        "reason": result["reason"],
        "replay_classes_match": classes_match,
        "prompt_deviation": deviation,
        "first_differing_message": first,
        "feedback": feedback,
        "feedback_deviation": feedback_kinds,
        "disqualified": (
            feedback == "stale"
            or (deviation is not None and "other" in deviation)
            or not classes_match
        ),
    }


def audit(args: argparse.Namespace) -> None:
    regen = Path(args.regen_dir)
    rd = regen / f"round{args.round}"
    tasks = load_jsonl(rd / "tasks.jsonl")
    results = {r["sample_key"]: r for r in load_jsonl(rd / "results.jsonl")}
    renders = replayed_renders(regen, args.round)
    rows = [audit_task(t, results[t["sample_key"]], renders.get(t["sample_key"])) for t in tasks]
    write_jsonl(rd / "audit.jsonl", rows)
    counts = {
        "tasks": len(rows),
        "prompt_deviation": dict(
            collections.Counter(deviation_label(r["prompt_deviation"]) for r in rows)
        ),
        "feedback": dict(collections.Counter(r["feedback"] for r in rows)),
        "feedback_deviation": dict(
            collections.Counter(k for r in rows for k in r["feedback_deviation"])
        ),
        "replay_classes_mismatch": sum(not r["replay_classes_match"] for r in rows),
        "disqualified": sorted(r["sample_key"] for r in rows if r["disqualified"]),
    }
    for r in rows:
        first = r["first_differing_message"]
        print(
            f"{r['sample_key']:18s} {r['action']:8s} render={deviation_label(r['prompt_deviation'])} "
            f"feedback={r['feedback']}{r['feedback_deviation'] if r['feedback_deviation'] else ''}"
            + (f" msg{first}" if first is not None else "")
            + ("  DISQUALIFIED" if r["disqualified"] else "")
        )
    print(json.dumps(counts, indent=2))


def deviation_label(kinds: list[str] | None) -> str:
    """`prompt_deviation` as one counter key: `unverifiable` (no replayed render), `exact`,
    or the `+`-joined kinds."""
    if kinds is None:
        return "unverifiable"
    return "+".join(kinds) if kinds else "exact"


def tag_deviation(row: dict[str, Any], audit_row: dict[str, Any]) -> dict[str, Any]:
    meta = {
        **row["triton_rlm"],
        "prompt_deviation": audit_row["prompt_deviation"],
        "feedback_deviation": audit_row["feedback_deviation"],
    }
    return {**row, "triton_rlm": meta}


def load_audit(regen: Path, rd: Path) -> dict[str, dict[str, Any]]:
    path = rd / "audit.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} missing: run `audit {regen} --round {rd.name[5:]}` first")
    return {r["sample_key"]: r for r in load_jsonl(path)}


def collect(args: argparse.Namespace) -> None:
    reexec = Path(args.reexec_dir)
    regen = Path(args.regen_dir)
    truncated = set(args.truncated)
    original = {
        p.name.split(".")[1]: load_jsonl(p) for p in sorted(reexec.glob("sft_trajectories.*.jsonl"))
    }
    n_original_all = sum(len(v) for v in original.values())
    if args.exclude_original_repairs:
        original = {
            split: [t for t in trajs if not t["triton_rlm"]["repair"]]
            for split, trajs in original.items()
        }
    original_keys = {t["example_id"] for trajs in original.values() for t in trajs}
    rounds = sorted(rd for rd in regen.glob("round*") if (rd / "results.jsonl").exists())
    audits = {k: a for rd in rounds for k, a in load_audit(regen, rd).items()}
    # a task's final outcome is its last round's result (a `continue` whose next round
    # never ran stays `continue`)
    outcomes = {r["sample_key"]: r for rd in rounds for r in load_jsonl(rd / "results.jsonl")}
    finished = [t for rd in rounds for t in load_jsonl(rd / "sft_trajectories.regen.jsonl")]
    excluded: dict[str, str] = {}
    for key, res in outcomes.items():
        if key in original_keys:
            if res["action"] == "finish":
                excluded[key] = "original_kept: the teacher trajectory is in the mix"
            continue
        if res["action"] == "finish":
            a = audits[key]
            if key in truncated:
                excluded[key] = "truncated_history"
            elif a["disqualified"]:
                excluded[key] = (
                    f"audit_disqualified: feedback {a['feedback']}, render {a['prompt_deviation']}"
                )
        elif res["action"] == "continue":
            excluded[key] = f"no_clean_repair: continue, round {res['round'] + 1} not run"
        else:
            excluded[key] = f"no_clean_repair: {res['reason']}"
    eligible = [t for t in finished if t["example_id"] not in excluded]
    train_regen = [t for t in eligible if t["triton_rlm"]["split"] == "train"]
    for t in eligible:
        if t["triton_rlm"]["split"] != "train":
            excluded[t["example_id"]] = "regenerated_val_split: val stays original-only"
    n_original_train = len(original.get("train", []))
    kept_keys = set(
        cap_regenerated(
            [t["example_id"] for t in train_regen], n_original_train, args.max_share, args.seed
        )
    )
    for t in train_regen:
        if t["example_id"] not in kept_keys:
            excluded[t["example_id"]] = f"resampled_out: over --max-share {args.max_share}"
    kept_regen = [
        tag_deviation(t, audits[t["example_id"]])
        for t in train_regen
        if t["example_id"] in kept_keys
    ]
    mix = regen / "mix"
    summary: dict[str, Any] = {
        "original_trajectories_total": n_original_all,
        "original_repairs_excluded": n_original_all - len(original_keys),
        "regenerated_tasks": len(outcomes),
        "regenerated_finished": len(finished),
        "regenerated_kept_after_cap": len(kept_regen),
        "regenerated_kept_prompt_deviation": dict(
            collections.Counter(
                deviation_label(t["triton_rlm"]["prompt_deviation"]) for t in kept_regen
            )
        ),
        "regenerated_kept_feedback_deviation": dict(
            collections.Counter(
                k for t in kept_regen for k in t["triton_rlm"]["feedback_deviation"]
            )
        ),
        "max_share": args.max_share,
        "loss_masking": {
            "sft_rows.<split>.jsonl": "USE THIS for SFT: one row per assistant turn with "
            "the full history repeated, assistant_loss_only_last=true (only the final "
            "assistant turn is supervised; same convention as the OOLONG per-turn rows)",
            "sft_trajectories.<split>.jsonl": "trajectory-level eval/inspection only: one "
            "row per trajectory, assistant_loss_only_last=false (every assistant turn "
            "supervised under role-based masking); mixing it with OOLONG rows mixes two "
            "masking conventions",
        },
        "excluded": dict(sorted(excluded.items())),
        "excluded_by_reason": dict(
            collections.Counter(reason.split(":")[0] for reason in excluded.values())
        ),
        "per_split": {},
    }
    for split, trajs in original.items():
        extra = kept_regen if split == "train" else []
        out_trajs = [*trajs, *extra]
        rows = [
            r
            for r in load_jsonl(reexec / f"sft_rows.{split}.jsonl")
            if r["example_id"] in original_keys
        ]
        regen_keys = {t["example_id"] for t in extra}
        rows += [
            tag_deviation(r, audits[r["example_id"]])
            for rd in rounds
            for r in load_jsonl(rd / "sft_rows.regen.jsonl")
            if r["example_id"] in regen_keys
        ]
        random.Random(args.seed).shuffle(out_trajs)
        random.Random(args.seed).shuffle(rows)
        write_jsonl(mix / f"sft_trajectories.{split}.jsonl", out_trajs)
        write_jsonl(mix / f"sft_rows.{split}.jsonl", rows)
        summary["per_split"][split] = {
            "original_trajectories": len(trajs),
            "regenerated_trajectories": len(extra),
            "regenerated_share": len(extra) / len(out_trajs) if out_trajs else 0.0,
            "rows": len(rows),
        }
    (mix / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("traces")
    p.add_argument("reexec_dir")
    p.add_argument("regen_dir")
    p.add_argument("--review-flags", help="review/review_flags.jsonl from ppbhatt_review.py")
    p.add_argument(
        "--flags",
        nargs="*",
        default=["THEIR_ONLY_TOKENS", "OUR_LINE_UNTOUCHED"],
        help="flags whose earliest failed turn in a kept trace starts regeneration",
    )
    p.add_argument(
        "--repairs",
        action="store_true",
        help="every kept repair: regenerate its correct turn against our last failure",
    )
    p.add_argument("--dropped", action="store_true", help="also regenerate dropped traces")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--shards", type=int, default=8)
    p.add_argument("--max-changed-lines", type=int, default=12)
    p.add_argument("--max-regen-turns", type=int, default=3)
    p.add_argument("--max-iterations", type=int, default=8)
    p.add_argument("--min-head-ratio", type=float, default=0.6)
    p.set_defaults(fn=prepare)

    e = sub.add_parser("execute")
    e.add_argument("traces")
    e.add_argument("regen_dir")
    e.add_argument("--round", type=int, required=True)
    e.add_argument("--shards", type=int, default=8)
    e.set_defaults(fn=execute)

    c = sub.add_parser("collect")
    c.add_argument("reexec_dir")
    c.add_argument("regen_dir")
    c.add_argument("--max-share", type=float, default=0.3)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument(
        "--exclude-original-repairs",
        action="store_true",
        help="keep only first-shot originals; teacher repairs enter the mix only if regenerated",
    )
    c.add_argument(
        "--truncated",
        nargs="*",
        default=[],
        metavar="KEY",
        help="sample_keys generated from a truncated history: tagged `truncated_history`, excluded",
    )
    c.set_defaults(fn=collect)

    a = sub.add_parser("audit")
    a.add_argument("regen_dir")
    a.add_argument("--round", type=int, required=True)
    a.set_defaults(fn=audit)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
