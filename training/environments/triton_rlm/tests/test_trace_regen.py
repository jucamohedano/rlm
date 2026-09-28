import pytest
from triton_rlm.trace_regen import (
    cap_regenerated,
    feedback_status,
    judge,
    message_deviations,
    prompt_deviation,
    regen_start,
    strip_fence,
    without_repl_vars,
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


TB = (
    "REPL output:\n\nTraceback (most recent call last):\n"
    '  File "<repl-ppbhatt_kernelbook_7-1>", line 4, in <module>\n'
    "TypeError: 'tuple' object cannot be interpreted as an integer\n"
)
VARS = "\n\nREPL variables: ['answer', 'context']\n"
REPORT = (
    "REPL output:\n\n[verifier] compiled: True  correct: False\n"
    "error: max_abs_diff 4.018e-02 > 1.0e-02 (ref 0.031ms)\n"
)


def test_without_repl_vars_matches_render_without_locals_keys() -> None:
    assert without_repl_vars(TB + VARS) == TB
    assert (
        without_repl_vars("REPL output:\nREPL variables: ['answer']\n") == "REPL output:\nNo output"
    )
    assert without_repl_vars(TB) == TB


def test_message_deviations_names_each_cosmetic_kind() -> None:
    regen_tb = TB.replace("<repl-ppbhatt_kernelbook_7-1>", "<repl-ppbhatt_regen_kernelbook_7_0-1>")
    assert message_deviations(TB, TB) == []
    assert message_deviations(TB, TB + VARS) == ["missing_repl_variables_line"]
    assert message_deviations(TB, regen_tb) == ["repl_filename"]
    assert message_deviations(TB, regen_tb + VARS) == [
        "missing_repl_variables_line",
        "repl_filename",
    ]
    drifted = REPORT.replace("4.018e-02", "3.754e-02").replace("0.031ms", "0.029ms")
    assert message_deviations(REPORT, drifted) == ["float_values"]
    # a different class or message is never cosmetic
    assert message_deviations(TB, TB.replace("TypeError", "ValueError")) == ["other"]
    assert message_deviations(REPORT, REPORT.replace("correct: False", "correct: True")) == [
        "other"
    ]
    assert message_deviations(TB, TB.replace("line 4", "line 5")) == ["other"]


def test_prompt_deviation_only_forgives_repl_output_messages() -> None:
    shown = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "```repl\nx\n```"},
        {"role": "user", "content": TB},
        {"role": "user", "content": "Turn 2/8:"},
    ]
    rendered = [*shown[:2], {"role": "user", "content": TB + VARS}, shown[3]]
    assert prompt_deviation(shown, shown) == ([], None)
    assert prompt_deviation(shown, rendered) == (["missing_repl_variables_line"], 2)
    other = [*shown[:3], {"role": "user", "content": "Turn 3/8:"}]
    assert prompt_deviation(shown, other) == (["other"], 3)
    assert prompt_deviation(shown, shown[:3]) == (["other"], 3)


def test_feedback_status_compares_the_answered_output() -> None:
    shown = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": REPORT},
        {"role": "user", "content": "Turn 2/8:"},
    ]
    exact = [
        *shown[:1],
        {"role": "user", "content": REPORT + VARS},
        shown[2],
        {"role": "assistant", "content": "fix"},
    ]
    assert feedback_status(shown, exact) == ("exact", [])
    drifted = [
        *shown[:1],
        {"role": "user", "content": REPORT.replace("4.018e-02", "3.754e-02")},
        shown[2],
    ]
    assert feedback_status(shown, drifted) == ("cosmetic", ["float_values"])
    stale = [*shown[:1], {"role": "user", "content": TB}, shown[2]]
    assert feedback_status(shown, stale) == ("stale", ["other"])
    assert feedback_status(shown, None) == ("unverifiable", [])
    assert feedback_status(shown, shown[:2]) == ("unverifiable", [])
    with pytest.raises(ValueError, match="not a REPL output"):
        feedback_status([shown[0], shown[2]], exact)
