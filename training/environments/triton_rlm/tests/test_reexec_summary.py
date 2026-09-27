import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ppbhatt_reexec.py"
spec = importlib.util.spec_from_file_location("ppbhatt_reexec", SCRIPT)
assert spec is not None and spec.loader is not None
reexec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reexec)


def trace(*codes: str) -> SimpleNamespace:
    return SimpleNamespace(turns=[SimpleNamespace(code=c) for c in codes])


def result(key: str, ours: list[str], decision: dict, **extra) -> dict:
    base = {
        "sample_key": key,
        "cluster": extra.pop("cluster", 0),
        "split": "train",
        "recorded": ours,
        "ours": ours,
        "decision": decision,
        "reward": extra.pop("reward", None),
        "first_correct_turn": extra.pop("first_correct_turn", None),
        "repair": extra.pop("repair", False),
    }
    assert not extra
    return base


def test_kept_repair_rollup() -> None:
    res = result(
        "k1",
        ["error[CompilationError]", "correct_fast", "error[TypeError]"],
        {"action": "keep", "keep_end": 1, "reason": "correct_fast at turn 1"},
        first_correct_turn=1,
        repair=True,
        reward=1.0,
    )
    s = reexec.trace_summary(trace("a", "b", "c"), res)
    assert s["kept"] and s["kept_prefix_len"] == 2
    assert s["first_correct_turn"] == 1 and s["repair"] is True
    assert s["final_class"] == "correct_fast"
    assert s["truncated_at"] is None and s["drop_reason"] is None
    assert s["n_source_turns"] == 3 and s["n_executed_turns"] == 3


def test_dropped_rollup_records_truncation() -> None:
    reason = "feedback mismatch at turn 1: theirs correct_fast ours error[WrapperContractError]"
    res = result(
        "k2",
        ["error[CompilationError]", "error[WrapperContractError]"],
        {"action": "drop", "keep_end": None, "reason": reason},
    )
    s = reexec.trace_summary(trace("a", "b", "c", "d"), res)
    assert not s["kept"] and s["kept_prefix_len"] == 0
    assert s["first_correct_turn"] is None and s["repair"] is False
    assert s["final_class"] == "error[WrapperContractError]"
    assert s["truncated_at"] == 1 and s["drop_reason"] == reason
    assert s["n_executed_turns"] == 2


def test_aggregate_counts_trajectories_not_turns() -> None:
    kept_repair = result(
        "k1",
        ["error[CompilationError]", "correct_fast"],
        {"action": "keep", "keep_end": 1, "reason": ""},
        first_correct_turn=1,
        repair=True,
    )
    kept_single = result(
        "k3",
        ["correct_slow[0.5x]"],
        {"action": "keep", "keep_end": 0, "reason": ""},
        first_correct_turn=0,
        cluster=1,
    )
    dropped = result(
        "k2",
        ["error[WrapperContractError]"],
        {"action": "drop", "keep_end": None, "reason": "feedback mismatch at turn 0: x"},
    )
    summaries = [
        reexec.trace_summary(trace("a", "b"), kept_repair),
        reexec.trace_summary(trace("a"), kept_single),
        reexec.trace_summary(trace("a", "b"), dropped),
    ]
    agg = reexec.aggregate(summaries)
    assert agg["traces_reexecuted"] == 3
    assert agg["traces_with_verified_kernel"] == 2
    assert agg["trajectories_with_repair"] == 1
    assert agg["repair_share"] == 0.5
    assert agg["kept_kernel_turns"] == 3
    assert agg["kept_ending_correct_fast"] == 1
    assert agg["distinct_tasks_kept"] == 2
    assert agg["first_correct_turn_hist"] == {1: 1, 0: 1}
    assert agg["drop_reasons"] == {"feedback mismatch": 1}
    assert agg["our_error_classes"] == {
        "error[CompilationError]": 1,
        "error[WrapperContractError]": 1,
    }
