import pytest
from triton_rlm.trace_regen import (
    cap_regenerated,
    judge,
    regen_start,
    strip_fence,
)
from triton_rlm.trace_translate import TurnClass

FAILED = "import triton\n\ndef triton_forward(x):\n    return k[grid, 4](x)\n"
FIXED = FAILED.replace("k[grid, 4](x)", "k[grid](x, num_warps=4)")
ERR = TurnClass("error", "TypeError", "'tuple' object cannot be interpreted as an integer")
OTHER = TurnClass("error", "CompilationError", "at 3:4: unsupported AST node")
LIMITS = dict(max_kernel_turns=7, max_regen_turns=3, max_changed_lines=12, min_head_ratio=0.6)


def test_strip_fence_accepts_bare_or_fenced_block() -> None:
    assert strip_fence(FIXED) == FIXED
    assert strip_fence(f"```repl\n{FIXED.rstrip()}\n```\n") == FIXED.rstrip()
    assert strip_fence("text\n```repl\nx\n```") == "text\n```repl\nx\n```"


def _summary(**kw):
    base = {
        "kept": True,
        "truncated_at": None,
        "drop_reason": None,
        "first_correct_turn": 2,
        "our_classes": ["error[TypeError]", "incorrect", "correct_fast"],
    }
    return {**base, **kw}


def test_regen_start_dropped_trace_starts_at_truncation() -> None:
    s = _summary(
        kept=False,
        truncated_at=1,
        first_correct_turn=None,
        drop_reason="feedback mismatch at turn 1: theirs error[TypeError] ours error[IndexError]",
        our_classes=["error[TypeError]", "error[IndexError]"],
    )
    k, reason = regen_start(s, [])
    assert k == 1 and reason.startswith("dropped: feedback mismatch")


def test_regen_start_kept_uses_earliest_flagged_failed_turn_before_first_correct() -> None:
    assert regen_start(_summary(), [1, 2])[0] == 1
    assert regen_start(_summary(), [0, 1])[0] == 0
    assert regen_start(_summary(), [2])[0] is None  # at the correct turn: not emitted
    assert regen_start(_summary(), [])[0] is None


def test_regen_start_all_repairs_replaces_the_correct_turn() -> None:
    assert regen_start(_summary(), [], all_repairs=True) == (
        1,
        "kept repair; regenerate correct turn t2",
    )
    first_shot = _summary(first_correct_turn=0, our_classes=["correct_fast"])
    assert regen_start(first_shot, [], all_repairs=True)[0] is None


def test_judge_rejects_identical_rewrite_and_leaks() -> None:
    fast = TurnClass("correct_fast", speedup=1.2)
    assert (
        judge(ERR, fast, FAILED, FAILED, kernel_turns=2, regen_turns=1, **LIMITS).action == "drop"
    )
    big = "\n".join(f"y{i} = {i}" for i in range(20))
    v = judge(ERR, fast, FAILED, big, kernel_turns=2, regen_turns=1, **LIMITS)
    assert v.action == "drop" and v.reason.startswith("rewrite")
    leak = FIXED + "\n# see /root/modal_app.py\n"
    assert (
        "marker" in judge(ERR, fast, FAILED, leak, kernel_turns=2, regen_turns=1, **LIMITS).reason
    )


def test_judge_finishes_on_correct_and_continues_on_new_error() -> None:
    fast = TurnClass("correct_fast", speedup=1.2)
    assert (
        judge(ERR, fast, FAILED, FIXED, kernel_turns=2, regen_turns=1, **LIMITS).action == "finish"
    )
    v = judge(ERR, OTHER, FAILED, FIXED, kernel_turns=2, regen_turns=1, **LIMITS)
    assert v.action == "continue"
    same = judge(ERR, ERR, FAILED, FIXED, kernel_turns=2, regen_turns=1, **LIMITS)
    assert same.action == "drop" and "same failure" in same.reason
    crash = TurnClass("error", "VerifierCrash", "verifier crashed")
    assert (
        "non-model"
        in judge(ERR, crash, FAILED, FIXED, kernel_turns=2, regen_turns=1, **LIMITS).reason
    )


def test_judge_budgets() -> None:
    v = judge(ERR, OTHER, FAILED, FIXED, kernel_turns=7, regen_turns=1, **LIMITS)
    assert v.action == "drop" and v.reason.startswith("turn budget")
    v = judge(ERR, OTHER, FAILED, FIXED, kernel_turns=3, regen_turns=3, **LIMITS)
    assert v.action == "drop" and v.reason.startswith("regeneration budget")
    fast = TurnClass("correct_slow", speedup=0.5)
    assert (
        judge(ERR, fast, FAILED, FIXED, kernel_turns=7, regen_turns=3, **LIMITS).action == "finish"
    )


def test_cap_regenerated_is_uniform_and_seeded() -> None:
    keys = [f"k{i}" for i in range(50)]
    kept = cap_regenerated(keys, n_original=70, max_share=0.3, seed=0)
    assert len(kept) == 30 and kept == cap_regenerated(keys, 70, 0.3, 0)
    assert kept != sorted(keys)[:30]
    assert cap_regenerated(keys[:10], 70, 0.3, 0) == sorted(keys[:10])
    with pytest.raises(ValueError):
        cap_regenerated(keys, 70, 1.0, 0)
