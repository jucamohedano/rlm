"""Naturalness / label-validity review of re-executed ppbhatt traces. CPU only.

    PYTHONPATH=<checkout>/training/src:<checkout>/training/environments/triton_rlm \
      python scripts/ppbhatt_review.py traces.parquet parsed/traces.jsonl reexec_out_dir \
        [--all] [--max-iterations 8]

Reads the GPU stage's exec_log.jsonl / trace_summary.jsonl (and the emitted SFT
trajectories, when present) and, per trace, re-renders the conversation exactly as
the harness would send it: our system prompt, our `Turn k/N:` user prompts, the
teacher's code in ```repl``` fences, OUR worker+verifier REPL output. Their feedback
text is read from the parquet only to be compared against ours; it never enters the
rendering.

Per reviewed trace the report answers, mechanically where possible:
  (i)   could our model plausibly have written each turn given OUR feedback?
        -> flags: THEIR_ONLY_TOKENS (next turn's added lines use identifiers that
           appear in their feedback but not in ours), OUR_LINE_UNTOUCHED (our
           traceback points at a line the next turn did not change), IDENTICAL_CODE,
           REASONING_CITES_THEIRS (their recorded reasoning for the next turn quotes
           text that only their feedback contained).
  (ii)  is the feedback text OURS? -> the REPL output is regenerated from exec_log;
        LEAK flags any source-harness marker in a rendered message; RENDER_MISMATCH
        flags a kept trajectory whose emitted messages differ from the re-render.
  (iii) recorded-vs-our class divergence per turn: DIVERGENT (with compat verdict).

Flags are heuristics that direct a human read; the verdict column is left for the
reviewer. Selection (default): every dropped trace, every repair=True trace, every
trace with a recorded/our class divergence; `--all` reviews everything.

Writes to out_dir/review/: index.md (selection + flag table), one <sample_key>.md per
reviewed trace with the rendered turn sequence and both feedback texts side by side,
and review_flags.jsonl for tooling.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from rlm.utils.prompts import (
    RLM_SYSTEM_PROMPT,
    QueryMetadata,
    build_rlm_system_prompt,
    build_user_prompt,
)
from triton_rlm.env import build_root_prompt
from triton_rlm.trace_translate import (
    SUBMIT_IDIOM,
    ParsedTrace,
    TurnClass,
    compatible,
    error_head_ratio,
    source_harness_markers,
)

from rlm_train.env import _format_repl_outputs, _pack_exec
from rlm_train.repl.base import ExecResult

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{3,}")
_TRACEBACK_LINE = re.compile(r'File "([^"]+)", line (\d+)')
_TRITON_AT = re.compile(r"\bat (\d+):(\d+):")
# Vocabulary of Python tracebacks and of both harnesses: never evidence of anything.
_STOP = set(
    """
    traceback most recent call last file line module during handling above exception
    another occurred name defined object cannot interpreted integer argument arguments
    missing required positional keyword takes given error errors compilation runtime
    output incorrect correct slow fast verifier triton_forward reference compiled
    max_abs_diff speedup kernel kernels torch triton tensor tensors dtype device cuda
    float float32 float16 int32 int64 shape size self none true false return returns
    def class import from with while else elif break continue pass raise assert lambda
    global nonlocal yield async await print range list dict tuple super
    """.split()
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fence(code: str) -> str:
    return f"```repl\n{code}\n```"


def render(
    trace: ParsedTrace, turn_logs: list[dict[str, Any]], keep_end: int | None, max_iterations: int
) -> list[dict[str, str]]:
    """Exactly `ppbhatt_reexec.reexec_trace`'s history construction, replayed from the log."""
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
    executed = [t for t in turn_logs if not t.get("skipped")]
    for i, log in enumerate(executed):
        code = trace.turns[log["index"]].code
        assert code is not None
        result = ExecResult(
            stdout=log["stdout"],
            stderr=log["stderr"],
            exception=log["exception"],
            locals_keys=log.get("locals_keys", []),
        )
        history.append({"role": "assistant", "content": fence(code)})
        history.extend(_format_repl_outputs([_pack_exec(code, result)]))
        if keep_end is not None and i == keep_end:
            history.append(
                build_user_prompt(
                    root_prompt=root_prompt, iteration=i + 1, max_iterations=max_iterations
                )
            )
            history.append({"role": "assistant", "content": fence(SUBMIT_IDIOM)})
            break
        if i + 1 < len(executed):
            history.append(
                build_user_prompt(
                    root_prompt=root_prompt, iteration=i + 1, max_iterations=max_iterations
                )
            )
    return history


def our_feedback_text(history: list[dict[str, str]]) -> list[str]:
    """The REPL output messages, in turn order."""
    return [
        m["content"]
        for m in history
        if m["role"] == "user" and m["content"].startswith("REPL output")
    ]


_REPL_VARS_LINE = re.compile(r"^REPL variables: .*$", re.M)


def same_messages(
    emitted: list[dict[str, str]], rendered: list[dict[str, str]], executed: list[dict[str, Any]]
) -> bool:
    """Runs before `locals_keys` was logged cannot reproduce the `REPL variables:` line."""
    if all("locals_keys" in log for log in executed):
        return emitted == rendered

    def strip(msgs: list[dict[str, str]]) -> list[tuple[str, str]]:
        return [(m["role"], _REPL_VARS_LINE.sub("", m["content"])) for m in msgs]

    return strip(emitted) == strip(rendered)


def tokens(text: str) -> set[str]:
    return {t for t in _IDENT.findall(text) if t.lower() not in _STOP}


def added_lines(before: str, after: str) -> list[str]:
    return [
        line[1:]
        for line in difflib.ndiff(before.splitlines(), after.splitlines())
        if line.startswith("+ ")
    ]


def our_error_lines(code: str, our_feedback: str, log: dict[str, Any]) -> set[int]:
    """1-based lines of `code` that OUR feedback points at."""
    lines: set[int] = set()
    for fname, n in _TRACEBACK_LINE.findall(our_feedback):
        if fname.startswith("<repl-") or fname.endswith("submission.py"):
            lines.add(int(n))
    verify = log.get("verify") or {}
    err = str(verify.get("error") or "")
    m = _TRITON_AT.search(err)
    if m:
        # Triton reports `at L:C` relative to the jitted function's `def` line; map it
        # onto every jitted def in the block (ambiguous when there are several).
        src = code.splitlines()
        for j, line in enumerate(src):
            if line.lstrip().startswith("def ") and j > 0 and "triton.jit" in src[j - 1]:
                lines.add(j + int(m.group(1)))
    return {n for n in lines if 1 <= n <= len(code.splitlines())}


def turn_transition_flags(
    code: str,
    next_code: str,
    their_fb: str,
    our_fb: str,
    next_reasoning: str,
    log: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if code == next_code:
        flags.append("IDENTICAL_CODE")
        return flags
    theirs_only = tokens(their_fb) - tokens(our_fb) - tokens(code)
    added = "\n".join(added_lines(code, next_code))
    used = sorted(theirs_only & tokens(added))
    if used:
        flags.append(f"THEIR_ONLY_TOKENS:{','.join(used)}")
    cited = sorted(theirs_only & tokens(next_reasoning))
    if cited:
        flags.append(f"REASONING_CITES_THEIRS:{','.join(cited[:6])}")
    bad_lines = our_error_lines(code, our_fb, log)
    if bad_lines:
        src, nxt = code.splitlines(), set(next_code.splitlines())
        untouched = [n for n in sorted(bad_lines) if src[n - 1] in nxt]
        if untouched and len(untouched) == len(bad_lines):
            flags.append(f"OUR_LINE_UNTOUCHED:{','.join(map(str, untouched))}")
    return flags


def review_trace(
    trace: ParsedTrace,
    raw_turns: list[dict[str, Any]],
    summary: dict[str, Any],
    turn_logs: list[dict[str, Any]],
    emitted: list[dict[str, str]] | None,
    min_head_ratio: float,
    max_iterations: int,
) -> tuple[dict[str, Any], str]:
    keep_end = summary["kept_prefix_len"] - 1 if summary["kept"] else None
    history = render(trace, turn_logs, keep_end, max_iterations)
    our_fb = our_feedback_text(history)
    executed = [t for t in turn_logs if not t.get("skipped")]
    flags: list[str] = []

    leaks = [
        f"{i}:{m['role']}:{mk}"
        for i, m in enumerate(history)
        for mk in source_harness_markers(m["content"])
    ]
    if leaks:
        flags.append("LEAK:" + ";".join(leaks[:4]))
    if emitted is not None and not same_messages(emitted, history, executed):
        flags.append("RENDER_MISMATCH")

    per_turn: list[dict[str, Any]] = []
    for i, log in enumerate(executed):
        k = log["index"]
        recorded, ours = TurnClass(**log["recorded"]), TurnClass(**log["ours"])
        ours.chain = tuple(tuple(c) for c in ours.chain)
        row: dict[str, Any] = {
            "turn": k,
            "recorded": recorded.short(),
            "ours": ours.short(),
            "compatible": compatible(recorded, ours, min_head_ratio),
            "recorded_message": recorded.message[:160],
            "our_message": ours.message[:160],
            "our_chain": [f"{t}: {m[:100]}" for t, m in ours.chain],
            "head_ratio": max(
                (
                    error_head_ratio(recorded, TurnClass("error", t, m))
                    for t, m in ((ours.exc_type, ours.message), *ours.chain)
                    if t == recorded.exc_type
                ),
                default=None,
            ),
            "flags": [],
        }
        if recorded.short() != ours.short():
            row["flags"].append("DIVERGENT" + ("" if row["compatible"] else "_INCOMPATIBLE"))
        if i + 1 < len(executed):
            code, next_code = trace.turns[k].code, trace.turns[executed[i + 1]["index"]].code
            assert code is not None and next_code is not None
            row["flags"] += turn_transition_flags(
                code,
                next_code,
                raw_turns[k].get("feedback_given") or "",
                our_fb[i],
                raw_turns[executed[i + 1]["index"]].get("reasoning") or "",
                log,
            )
        per_turn.append(row)
        flags += [f"t{k}:{f}" for f in row["flags"]]

    record = {
        "sample_key": trace.sample_key,
        "kept": summary["kept"],
        "repair": summary["repair"],
        "drop_reason": summary["drop_reason"],
        "recorded_classes": summary["recorded_classes"],
        "our_classes": summary["our_classes"],
        "n_messages": len(history),
        "flags": flags,
        "per_turn": per_turn,
    }
    return record, render_markdown(trace, raw_turns, summary, history, executed, per_turn, flags)


def _system_digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def render_markdown(
    trace: ParsedTrace,
    raw_turns: list[dict[str, Any]],
    summary: dict[str, Any],
    history: list[dict[str, str]],
    executed: list[dict[str, Any]],
    per_turn: list[dict[str, Any]],
    flags: list[str],
) -> str:
    out: list[str] = [f"# {trace.sample_key}", ""]
    out.append(
        f"kept={summary['kept']} repair={summary['repair']} "
        f"kept_prefix_len={summary['kept_prefix_len']} first_correct_turn={summary['first_correct_turn']}"
    )
    out.append(f"drop_reason: {summary['drop_reason']}")
    out.append(f"recorded: {summary['recorded_classes']}")
    out.append(f"ours:     {summary['our_classes']}")
    out.append("flags: " + (", ".join(flags) if flags else "none"))
    out += ["", "## Rendered turn sequence (what our harness sends)", ""]
    for i, m in enumerate(history):
        if m["role"] == "system":
            out.append(
                f"### [{i}] system (RLM system prompt, {len(m['content'])} chars, "
                f"sha {_system_digest(m['content'])}; identical across traces, see index.md)"
            )
            out.append("")
            continue
        out.append(f"### [{i}] {m['role']}")
        out.append("")
        out.append("~~~text")
        out.append(m["content"].replace("~~~", "~ ~ ~"))
        out.append("~~~")
        out.append("")
    out += ["", "## Per-turn: their feedback vs ours", ""]
    for row, log in zip(per_turn, executed, strict=True):
        k = row["turn"]
        out.append(
            f"### turn {k}: recorded {row['recorded']} / ours {row['ours']} "
            f"(compatible={row['compatible']}) flags={row['flags'] or 'none'}"
        )
        out.append("")
        out.append("their feedback_given (source harness; NOT emitted):")
        out.append("~~~text")
        out.append(
            (raw_turns[k].get("feedback_given") or "<none: last turn>").replace("~~~", "~ ~ ~")
        )
        out.append("~~~")
        our_err = log.get("exception") or (log.get("verify") or {}).get("error") or ""
        out.append("our exception / verify.error:")
        out.append("~~~text")
        out.append(str(our_err).replace("~~~", "~ ~ ~") or "<none>")
        out.append("~~~")
        nxt = k + 1
        if nxt < len(raw_turns) and raw_turns[nxt].get("reasoning"):
            out.append(f"their recorded reasoning for turn {nxt} (NOT emitted):")
            out.append("~~~text")
            out.append(str(raw_turns[nxt]["reasoning"]).replace("~~~", "~ ~ ~"))
            out.append("~~~")
        out.append("")
    out += [
        "",
        "## Reviewer verdict",
        "",
        "- (i) plausible under our prompt+feedback: ",
        "- (ii) feedback is ours (no source text): ",
        "- (iii) conditioned on previous real result / not stitched: ",
        "",
    ]
    return "\n".join(out)


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")[:120] or "-"


def select(summaries: list[dict[str, Any]], review_all: bool) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for s in summaries:
        key = s["sample_key"]
        if review_all:
            groups["all"].append(key)
        if not s["kept"]:
            reason = s["drop_reason"] or ""
            if "WrapperContractError" in "".join(s["our_classes"]):
                groups["dropped:non_binding_wrapper"].append(key)
            elif reason.startswith("feedback mismatch"):
                groups["dropped:feedback_mismatch"].append(key)
            elif reason.startswith("source-harness marker at"):
                groups["dropped:source_leak"].append(key)
            else:
                groups["dropped:other"].append(key)
        if s["repair"]:
            groups["repair"].append(key)
        if any(r != o for r, o in zip(s["recorded_classes"], s["our_classes"], strict=False)):
            groups["class_divergence"].append(key)
    return dict(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("traces_jsonl")
    ap.add_argument("reexec_dir")
    ap.add_argument(
        "--all", action="store_true", help="review every trace, not only the priority sets"
    )
    ap.add_argument("--max-iterations", type=int, default=8)
    ap.add_argument("--min-head-ratio", type=float, default=0.6)
    args = ap.parse_args()

    reexec = Path(args.reexec_dir)
    out = reexec / "review"
    out.mkdir(parents=True, exist_ok=True)

    raw = {
        r["sample_key"]: (json.loads(r["turns"]) if isinstance(r["turns"], str) else r["turns"])
        for r in pq.read_table(args.parquet).to_pylist()
    }
    traces = {
        t["sample_key"]: ParsedTrace.from_json(t) for t in load_jsonl(Path(args.traces_jsonl))
    }
    summaries = load_jsonl(reexec / "trace_summary.jsonl")
    logs = {row["sample_key"]: row["turn_logs"] for row in load_jsonl(reexec / "exec_log.jsonl")}
    emitted: dict[str, list[dict[str, str]]] = {}
    for path in reexec.glob("sft_trajectories.*.jsonl"):
        for row in load_jsonl(path):
            emitted[row["triton_rlm"]["task_id"]] = row["messages"]

    groups = select(summaries, args.all)
    selected = sorted({k for keys in groups.values() for k in keys})
    by_key = {s["sample_key"]: s for s in summaries}

    records: list[dict[str, Any]] = []
    system_digests: collections.Counter[str] = collections.Counter()
    for key in selected:
        trace, summary = traces[key], by_key[key]
        turn_logs = logs[key]
        record, md = review_trace(
            trace,
            raw[key],
            summary,
            turn_logs,
            emitted.get(key),
            args.min_head_ratio,
            args.max_iterations,
        )
        records.append(record)
        (out / f"{key}.md").write_text(md, encoding="utf-8")
        history = render(
            trace,
            turn_logs,
            summary["kept_prefix_len"] - 1 if summary["kept"] else None,
            args.max_iterations,
        )
        system_digests[_system_digest(history[0]["content"])] += 1

    with (out / "review_flags.jsonl").open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    flag_counts = collections.Counter(
        re.sub(r"^t\d+:", "", fl).split(":")[0] for r in records for fl in r["flags"]
    )
    idx = [f"# ppbhatt review: {len(records)} traces of {len(summaries)}", ""]
    idx.append("## Selection")
    for g, keys in sorted(groups.items()):
        idx.append(f"- {g}: {len(keys)}")
    idx += ["", "## Flag totals (heuristic, over reviewed traces)"]
    for fl, n in flag_counts.most_common():
        idx.append(f"- {fl}: {n}")
    idx += ["", f"## System prompt digests across rendered traces: {dict(system_digests)}", ""]
    mismatches = [
        (r, row)
        for r in records
        if (r["drop_reason"] or "").startswith("feedback mismatch")
        for row in r["per_turn"][-1:]
    ]
    if mismatches:
        idx += [
            "## Feedback-mismatch drops: recorded vs our error at the truncating turn",
            "",
            "| trace | turn | recorded | ours | head_ratio | recorded message | our message | our chain |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r, row in mismatches:
            ratio = "-" if row["head_ratio"] is None else f"{row['head_ratio']:.2f}"
            idx.append(
                f"| {r['sample_key']} | {row['turn']} | {row['recorded']} | {row['ours']} | {ratio} | "
                f"`{_cell(row['recorded_message'])}` | `{_cell(row['our_message'])}` | "
                f"`{_cell(' // '.join(row['our_chain']))}` |"
            )
        idx.append("")
    idx += [
        "## Traces",
        "",
        "| trace | kept | repair | recorded | ours | drop_reason | flags |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in records:
        idx.append(
            f"| [{r['sample_key']}]({r['sample_key']}.md) | {r['kept']} | {r['repair']} | "
            f"{' '.join(r['recorded_classes'])} | {' '.join(r['our_classes'])} | "
            f"{(r['drop_reason'] or '')[:60]} | {'; '.join(r['flags']) or '-'} |"
        )
    (out / "index.md").write_text("\n".join(idx) + "\n", encoding="utf-8")
    print(f"reviewed {len(records)} traces -> {out}")
    for g, keys in sorted(groups.items()):
        print(f"  {g}: {len(keys)}")
    for fl, n in flag_counts.most_common():
        print(f"  flag {fl}: {n}")


if __name__ == "__main__":
    main()
