import pytest
from triton_rlm.trace_translate import (
    AmbiguousReference,
    TurnClass,
    classify_ours,
    classify_recorded,
    compatible,
    convert_source,
    decide,
    dedupe_clusters,
    extract_code,
    normalize_pytorch,
    reference_root_class,
)

ERR = TurnClass("error", "CompilationError", "at 12:4: def kernel(x_ptr, ...)")
INC = TurnClass("incorrect")
SLOW = TurnClass("correct_slow", speedup=0.7)
SLOWER = TurnClass("correct_slow", speedup=0.5)
FAST = TurnClass("correct_fast", speedup=1.4)


def test_extract_and_convert() -> None:
    msg = "<triton>\nimport triton\ndef triton_kernel_wrapper(x):\n    return x\n</triton>"
    code = extract_code(msg)
    assert code == "import triton\ndef triton_kernel_wrapper(x):\n    return x"
    assert convert_source(code) == "import triton\ndef triton_forward(x):\n    return x"
    assert extract_code("no block") is None
    assert extract_code("<triton>a</triton><triton>b</triton>") is None


def test_classify_recorded_reads_feedback_not_speed() -> None:
    turn = {
        "feedback_given": "Compilation/runtime error:\nNameError: name 'n_required' is not defined",
        "result": {"correctness": False, "speedup": None},
    }
    assert classify_recorded(turn) == TurnClass(
        "error", "NameError", "name 'n_required' is not defined"
    )
    turn = {
        "feedback_given": "Correct but slow (0.5x)",
        "result": {"correctness": True, "speedup": 0.5},
    }
    assert classify_recorded(turn).kind == "correct_slow"
    turn = {"feedback_given": "", "result": {"correctness": True, "speedup": 1.5, "error": None}}
    assert classify_recorded(turn).kind == "correct_fast"


def test_classify_ours() -> None:
    assert (
        classify_ours("SyntaxError: invalid syntax (<repl-x-1>, line 3)", None).exc_type
        == "SyntaxError"
    )
    assert classify_ours(None, None).exc_type == "NoEntryPoint"
    assert (
        classify_ours(None, {"compiled": True, "correct": False, "error": "max diff 1e-1"}).kind
        == "incorrect"
    )
    assert (
        classify_ours(None, {"compiled": True, "correct": True, "speedup": 0.3}).kind
        == "correct_slow"
    )
    assert (
        classify_ours(None, {"compiled": True, "correct": True, "speedup": 2.0}).kind
        == "correct_fast"
    )


def test_compatible_requires_same_class_and_exception() -> None:
    ours = TurnClass("error", "CompilationError", "at 13:4: def kernel(x_ptr, ...)")
    assert compatible(ERR, ours, 0.6)
    assert not compatible(
        ERR, TurnClass("error", "TypeError", "at 12:4: def kernel(x_ptr, ...)"), 0.6
    )
    assert not compatible(ERR, INC, 0.6)
    assert compatible(INC, TurnClass("incorrect", message="max diff 0.5"), 0.6)


def test_wrapper_contract_error_matches_only_recorded_arity_type_error() -> None:
    ours = TurnClass(
        "error",
        "WrapperContractError",
        "triton_forward requires positional args ['x']; the harness passes 1 input tensor(s)",
    )
    for msg in (
        "triton_kernel_wrapper() missing 1 required positional argument: 'w0'",
        "triton_kernel_wrapper() missing 2 required keyword-only arguments: 'weight' and 'bias'",
        "triton_kernel_wrapper() takes 1 positional argument but 3 were given",
    ):
        assert compatible(TurnClass("error", "TypeError", msg), ours, 0.6)
    assert not compatible(
        TurnClass("error", "TypeError", "'tuple' object cannot be interpreted as an integer"),
        ours,
        0.6,
    )
    assert not compatible(
        TurnClass("error", "TypeError", "Identity.__init__() got an unexpected keyword argument"),
        ours,
        0.6,
    )
    assert not compatible(ERR, ours, 0.6)
    # the other direction is unchanged: a plain TypeError of ours is not a contract error
    assert not compatible(
        TurnClass(
            "error",
            "TypeError",
            "triton_kernel_wrapper() missing 1 required positional argument: 'w0'",
        ),
        TurnClass("error", "TypeError", "'tuple' object cannot be interpreted as an integer"),
        0.6,
    )


def test_decide_first_shot_correct() -> None:
    assert decide([FAST], [FAST], 0.6).action == "keep"
    assert decide([FAST], [FAST], 0.6).keep_end == 0


def test_decide_repair_then_correct() -> None:
    d = decide([ERR, FAST], [ERR], 0.6)
    assert d.action == "execute_next"
    d = decide([ERR, FAST], [ERR, FAST], 0.6)
    assert (d.action, d.keep_end) == ("keep", 1)


def test_decide_drops_on_feedback_mismatch() -> None:
    d = decide([ERR, FAST], [INC], 0.6)
    assert d.action == "drop"
    assert "mismatch" in d.reason


def test_decide_never_ends_on_failure() -> None:
    assert decide([ERR], [ERR], 0.6).action == "drop"
    assert decide([ERR, FAST], [ERR, INC], 0.6).action == "drop"


def test_decide_correct_slow_prefix_and_faster_tail() -> None:
    # slow, their feedback slow -> try the continuation
    assert decide([SLOW, FAST], [SLOW], 0.6).action == "execute_next"
    # continuation faster -> keep through it
    assert decide([SLOW, FAST], [SLOW, FAST], 0.6).keep_end == 1
    # continuation not faster -> keep the slow prefix
    assert decide([SLOW, SLOW], [SLOW, SLOWER], 0.6).keep_end == 0
    # continuation broken -> keep the slow prefix, not the failure
    assert decide([SLOW, ERR], [SLOW, ERR], 0.6).keep_end == 0
    # their feedback was not slow -> no continuation is valid, keep the slow turn
    assert decide([INC, FAST], [SLOW], 0.6).keep_end == 0
    # slow on the last turn -> keep
    assert decide([SLOW], [SLOW], 0.6).keep_end == 0


def test_root_class_skips_building_blocks_and_rejects_ambiguity() -> None:
    src = (
        "import torch\nimport torch.nn as nn\n"
        "class ShapeError(Exception):\n    pass\n"
        "class BasicBlock(nn.Module):\n"
        "    def __init__(self):\n        super(BasicBlock, self).__init__()\n"
        "class Net(torch.nn.Module):\n"
        "    def __init__(self):\n        super().__init__()\n        self.b = BasicBlock()\n"
    )
    assert reference_root_class(src) == "Net"
    assert reference_root_class(src + "class Model(nn.Module):\n    pass\n") == "Model"
    with pytest.raises(AmbiguousReference):
        reference_root_class(src + "class Other(nn.Module):\n    pass\n")
    with pytest.raises(AmbiguousReference):
        reference_root_class("class NotAModule:\n    pass\n")


def test_dedupe_ignores_comments_and_whitespace() -> None:
    a = "import torch\nclass M(torch.nn.Module):\n    def forward(self, x):\n        return torch.relu(x) + 1\n"
    b = a.replace("return", "# comment\n        return").replace("    ", "  ")
    c = a.replace("relu", "sigmoid").replace("+ 1", "* 2")
    assert normalize_pytorch(a) == normalize_pytorch(b)
    ids = dedupe_clusters([normalize_pytorch(t) for t in (a, b, c)])
    assert ids[0] == ids[1] != ids[2]
