"""Regenerate the next turn of a translated trace against OUR feedback. Pure functions.

Translation keeps a teacher turn k+1 only if their feedback after turn k was
compatible with ours. Where that fails (rule-1 drops) or where the review shows the
teacher's edit reacting to text only *their* error carried, the deterministic path can
only drop. The regeneration path (`scripts/ppbhatt_regen.py`) instead shows a strong
model the conversation exactly as our env renders it, up to and including OUR REPL
output for turn k, and asks for the next ```repl block only. The block is executed in
our worker and judged by our verifier; nothing model-visible is fabricated.

Accept rule (`judge`), applied after each regenerated turn, our verifier the only judge:
  - the block must differ from the failed block and change at most `max_changed_lines`
    lines of it (a repair, not a rewrite: the SFT target is a 3B and its data should
    look like a small model's edit);
  - no source-harness marker in the block;
  - verified correct -> the trajectory finishes (correct turn, then the submit idiom);
  - a failure our env attributes to the task or the verifier, not the model
    (`NON_MODEL_ERRORS`) -> drop;
  - the same failure class as the turn it was meant to fix -> drop (the model did not
    act on the feedback; "same error twice" is exactly the loop we must not teach);
  - a *new* model error or incorrect result -> continue to another round if the turn
    and regeneration budgets allow, else drop. Unresolved traces are dropped, never
    padded.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Literal

from triton_rlm.trace_translate import (
    TurnClass,
    changed_line_count,
    compatible,
    source_harness_markers,
)

# Failure classes a regenerated turn may not introduce: `AmbiguousReference` is a
# property of the task's reference module, VerifierCrash/TimeoutError of the verifier
# run; the rest are contract failures (no block, no `triton_forward`, not a Triton kernel,
# wrong wrapper signature) rather than kernel bugs -- a trajectory that first breaks the
# contract and then fixes it teaches nothing about kernels.
NON_MODEL_ERRORS = frozenset(
    {
        "AmbiguousReference",
        "VerifierCrash",
        "TimeoutError",
        "NoCode",
        "NoEntryPoint",
        "NotTritonKernel",
        "WrapperContractError",
    }
)

_FAILED = ("error", "incorrect")
_FENCE = re.compile(r"\A\s*```[a-zA-Z]*\n(.*?)\n```\s*\Z", re.S)


def strip_fence(text: str) -> str:
    """Accept a bare block or one fenced ```repl block; anything else is left as is
    (and then fails the single-block check in the driver)."""
    m = _FENCE.match(text)
    return m.group(1) if m else text


def regen_start(
    summary: dict[str, Any], flagged_turns: list[int], *, all_repairs: bool = False
) -> tuple[int | None, str]:
    """Index k of the last turn to keep from the source trace; turn k+1 is regenerated.

    Dropped trace: k is the truncation turn (a rule-1 mismatch or a trailing failure);
    turns 0..k-1 are failures our verifier agreed with (`decide` would have kept a
    correct one). Kept trace: the earliest flagged failed turn before the first correct
    one (a flag at or after the first correct turn is outside the emitted prefix and
    needs nothing), or with `all_repairs` the turn before the first correct one, so the
    teacher's repairing edit itself is replaced by one written against our feedback.
    """
    ours: list[str] = summary["our_classes"]
    if not summary["kept"]:
        k = summary["truncated_at"]
        if k is None or not ours[k].startswith(_FAILED):
            return None, f"dropped without a trailing failure: {summary['drop_reason']}"
        return k, f"dropped: {summary['drop_reason']}"
    first_correct = summary["first_correct_turn"]
    if all_repairs:
        if first_correct == 0:
            return None, "kept; first-shot success, nothing to regenerate"
        return first_correct - 1, f"kept repair; regenerate correct turn t{first_correct}"
    candidates = sorted(
        k for k in flagged_turns if k < first_correct and ours[k].startswith(_FAILED)
    )
    if not candidates:
        return None, "kept; no flagged failed turn before the first correct turn"
    return candidates[0], f"kept; flagged transition t{candidates[0]}->t{candidates[0] + 1}"


@dataclass
class Verdict:
    action: Literal["finish", "continue", "drop"]
    reason: str


def judge(
    prev: TurnClass,
    new: TurnClass,
    prev_code: str,
    new_code: str,
    *,
    kernel_turns: int,
    regen_turns: int,
    max_kernel_turns: int,
    max_regen_turns: int,
    max_changed_lines: int,
    min_head_ratio: float,
) -> Verdict:
    """`kernel_turns` / `regen_turns` count the turns executed so far INCLUDING the new
    one; `max_kernel_turns` is the env's iteration budget minus the submit turn."""
    if markers := source_harness_markers(new_code):
        return Verdict("drop", f"source-harness marker in regenerated block: {markers}")
    changed = changed_line_count(prev_code, new_code)
    if changed == 0:
        return Verdict("drop", "regenerated block identical to the failed block")
    if changed > max_changed_lines:
        return Verdict(
            "drop", f"rewrite, not repair: {changed} changed lines > {max_changed_lines}"
        )
    if new.kind.startswith("correct"):
        return Verdict("finish", f"{new.short()} after {changed} changed lines")
    if new.exc_type in NON_MODEL_ERRORS:
        return Verdict("drop", f"non-model failure {new.short()}")
    if compatible(prev, new, min_head_ratio):
        return Verdict("drop", f"same failure repeated: {prev.short()} -> {new.short()}")
    if kernel_turns >= max_kernel_turns:
        return Verdict("drop", f"turn budget: {kernel_turns} kernel turns, {new.short()}")
    if regen_turns >= max_regen_turns:
        return Verdict(
            "drop", f"regeneration budget: {regen_turns} regenerated turns, {new.short()}"
        )
    return Verdict("continue", f"new failure {new.short()} after {changed} changed lines")


def approx_tokens(messages: list[dict[str, str]]) -> int:
    """~4 chars/token; enough to size a batch, not to bill it."""
    return sum(len(m["content"]) for m in messages) // 4


def cap_regenerated(
    regenerated_keys: list[str], n_original: int, max_share: float, seed: int
) -> list[str]:
    """Keys of the regenerated trajectories to keep so they are at most `max_share` of
    the mix. A uniform seeded sample over trajectories, never the first-N or the ones
    that finished in the fewest rounds: capping must not tilt the mix toward whichever
    traces regenerate easily."""
    if not 0.0 < max_share < 1.0:
        raise ValueError(f"max_share must be in (0, 1), got {max_share}")
    limit = int(max_share * n_original / (1.0 - max_share))
    if len(regenerated_keys) <= limit:
        return sorted(regenerated_keys)
    return sorted(random.Random(seed).sample(sorted(regenerated_keys), limit))
