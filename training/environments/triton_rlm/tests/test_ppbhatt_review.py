import importlib.util
from pathlib import Path

from triton_rlm.trace_translate import source_harness_markers

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ppbhatt_review.py"
spec = importlib.util.spec_from_file_location("ppbhatt_review", SCRIPT)
assert spec is not None and spec.loader is not None
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)

THEIR_FB = (
    "Compilation/runtime error:\nNameError: name 'scale' is not defined\n"
    '  File "/root/modal_app.py", line 516, in benchmark_kernelbench\n'
    "    kernel_output = triton_kernel_wrapper(*kernel_args[:n_required])\n"
    "NameError: name 'n_required' is not defined\n"
)
OUR_FB = (
    "REPL output:\n\n[verifier] triton_forward vs reference module\n  compiled: False\n"
    "  correct: False\n  error: NameError: name 'scale' is not defined\n"
)
CODE = "def triton_forward(x):\n    return x * scale\n"


def test_source_harness_markers_hit_their_text_not_ours() -> None:
    assert set(source_harness_markers(THEIR_FB)) == {
        "n_required",
        "/root/modal_app.py",
        "benchmark_kernelbench",
        "Compilation/runtime error:",
    }
    assert source_harness_markers(OUR_FB) == []


def test_transition_flags_catch_a_fix_for_their_error_text() -> None:
    stitched = "def triton_forward(x, n_required=1):\n    return x * scale\n"
    flags = review.turn_transition_flags(
        CODE, stitched, THEIR_FB, OUR_FB, "the harness wants n_required", {"verify": {}}
    )
    assert "THEIR_ONLY_TOKENS:n_required" in flags
    assert "REASONING_CITES_THEIRS:n_required" in flags


def test_transition_flags_silent_for_a_fix_of_our_error() -> None:
    fixed = "def triton_forward(x):\n    return x * 2.0\n"
    assert review.turn_transition_flags(CODE, fixed, THEIR_FB, OUR_FB, "", {"verify": {}}) == []


def test_transition_flags_identical_code() -> None:
    assert review.turn_transition_flags(CODE, CODE, THEIR_FB, OUR_FB, "", {}) == ["IDENTICAL_CODE"]


def test_our_line_untouched_uses_worker_traceback_lines() -> None:
    our_fb = 'REPL output:\nTraceback\n  File "<repl-x-1>", line 2, in triton_forward\nNameError'
    same_line = "import torch\ndef triton_forward(x):\n    return x * scale\n"
    flags = review.turn_transition_flags(CODE, same_line, "", our_fb, "", {"verify": {}})
    assert flags == ["OUR_LINE_UNTOUCHED:2"]


def test_select_groups_drops_repairs_and_divergence() -> None:
    summaries = [
        {
            "sample_key": "a",
            "kept": False,
            "repair": False,
            "drop_reason": "feedback mismatch at turn 0: theirs error[TypeError] ours error[IndexError]",
            "recorded_classes": ["error[TypeError]"],
            "our_classes": ["error[IndexError]"],
        },
        {
            "sample_key": "b",
            "kept": False,
            "repair": False,
            "drop_reason": "ends on error[WrapperContractError]",
            "recorded_classes": ["correct_fast"],
            "our_classes": ["error[WrapperContractError]"],
        },
        {
            "sample_key": "c",
            "kept": True,
            "repair": True,
            "drop_reason": None,
            "recorded_classes": ["error[TypeError]", "correct_fast"],
            "our_classes": ["error[CompilationError]", "correct_fast"],
        },
        {
            "sample_key": "d",
            "kept": True,
            "repair": False,
            "drop_reason": None,
            "recorded_classes": ["correct_fast"],
            "our_classes": ["correct_fast"],
        },
    ]
    groups = review.select(summaries, review_all=False)
    assert groups["dropped:feedback_mismatch"] == ["a"]
    assert groups["dropped:non_binding_wrapper"] == ["b"]
    assert groups["repair"] == ["c"]
    assert groups["class_divergence"] == ["a", "b", "c"]
    assert "all" not in groups


def test_same_messages_tolerates_missing_locals_keys_in_old_logs() -> None:
    emitted = [{"role": "user", "content": "REPL output:\nx\n\nREPL variables: ['a']\n"}]
    rendered = [{"role": "user", "content": "REPL output:\nx\n\nREPL variables: []\n"}]
    assert not review.same_messages(emitted, rendered, [{"locals_keys": []}])
    assert review.same_messages(emitted, rendered, [{}])
