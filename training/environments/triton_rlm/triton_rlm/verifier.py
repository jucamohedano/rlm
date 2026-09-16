"""Compile + correctness + speed verifier for submitted Triton kernels.

Runs OUT OF PROCESS on purpose: a bad or malicious kernel must never corrupt the
trainer or contaminate the next rollout. A segfault here costs one rollout, not
the run.

Submission contract (what the model is trained to emit):
    the submitted string is Python source; it may contain imports; it MUST define

        def triton_forward(*tensors) -> torch.Tensor

    The reference module is loaded from a file and provides `Model`,
    `get_inputs()` and `get_init_inputs()` (KernelBench style).

Speed protocol (see notes/reward-design.md):
    only measured when the kernel is correct; `triton.testing.do_bench` with
    warmup + `synchronize()` + return_mode="median"; speedup = ref_ms / kernel_ms.

`verify_triton` is synchronous and does blocking subprocess work; call it from
async code with `asyncio.to_thread(...)` so it never blocks the event loop.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any

_RUNNER = textwrap.dedent(
    """
    import importlib.util
    import json
    import sys

    import torch
    import triton.testing

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    REF_PATH, SUB_PATH = sys.argv[1], sys.argv[2]
    N_TRIALS = int(sys.argv[3])
    ATOL = float(sys.argv[4])
    RTOL = float(sys.argv[5])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = {
        "compiled": False,
        "correct": False,
        "max_diff": None,
        "speedup": None,
        "ref_ms": None,
        "kernel_ms": None,
        "error": "",
        "trials": N_TRIALS,
        "device": device,
    }

    try:
        ref_mod = _load(REF_PATH, "reference")
        Model = ref_mod.Model
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
        sub_mod = _load(SUB_PATH, "submission")
        triton_forward = sub_mod.triton_forward
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(result))
        raise SystemExit(0)

    max_diff = 0.0
    correct = True
    produced_output = False
    try:
        for _ in range(N_TRIALS):
            model = Model(*get_init_inputs()).to(device)
            model.eval()
            inputs = [t.to(device) for t in get_inputs()]
            with torch.no_grad():
                ref = model(*inputs)
                out = triton_forward(*inputs)   # first call triggers the Triton JIT
            produced_output = True
            if tuple(ref.shape) != tuple(out.shape):
                correct = False
                result["error"] = (
                    f"shape mismatch: ref {tuple(ref.shape)} vs out {tuple(out.shape)}"
                )
                break
            diff = (ref.float() - out.float()).abs().max().item()
            max_diff = max(max_diff, float(diff))
            if not torch.allclose(ref, out, atol=ATOL, rtol=RTOL):
                correct = False
                break
    except Exception as e:
        correct = False
        result["error"] = f"{type(e).__name__}: {e}"

    result["compiled"] = produced_output
    result["correct"] = correct
    result["max_diff"] = max_diff if produced_output else None

    if correct:
        try:
            def ref_fn():
                with torch.no_grad():
                    return model(*inputs)
            def kernel_fn():
                with torch.no_grad():
                    return triton_forward(*inputs)
            ref_ms = float(
                triton.testing.do_bench(ref_fn, warmup=25, rep=100, return_mode="median")
            )
            kernel_ms = float(
                triton.testing.do_bench(kernel_fn, warmup=25, rep=100, return_mode="median")
            )
            result["ref_ms"] = ref_ms
            result["kernel_ms"] = kernel_ms
            result["speedup"] = ref_ms / max(kernel_ms, 1e-9)
        except Exception as e:
            result["error"] = f"timing failed: {type(e).__name__}: {e}"

    print(json.dumps(result))
    raise SystemExit(0)
    """
)


def verify_triton(
    reference_code: str,
    submission_code: str,
    *,
    n_trials: int = 5,
    atol: float = 1e-3,
    rtol: float = 1e-4,
    timeout_s: float = 240.0,
    python: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Compile, run, and (if correct) time `submission_code` in a subprocess.

    Returns {"compiled", "correct", "max_diff", "speedup", "ref_ms", "kernel_ms",
    "error", "trials", "device"}.
    """
    python = python or os.environ.get("RLM_VERIFY_PYTHON") or sys.executable
    _fail = lambda msg: {   # noqa: E731
        "compiled": False, "correct": False, "max_diff": None,
        "speedup": None, "ref_ms": None, "kernel_ms": None,
        "error": msg, "trials": n_trials, "device": "?",
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_path = tmp / "reference.py"
        sub_path = tmp / "submission.py"
        runner_path = tmp / "runner.py"
        ref_path.write_text(reference_code, encoding="utf-8")
        sub_path.write_text(submission_code, encoding="utf-8")
        runner_path.write_text(_RUNNER, encoding="utf-8")
        cmd = [
            python, str(runner_path),
            str(ref_path), str(sub_path),
            str(int(n_trials)), repr(float(atol)), repr(float(rtol)),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=cwd)
        except subprocess.TimeoutExpired:
            return _fail(f"verification timed out after {timeout_s}s")
        if proc.returncode != 0:
            return _fail(f"verifier crashed (rc={proc.returncode}): {(proc.stderr or '')[-1500:]}")
        try:
            return json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return _fail(f"unparseable verifier output: {(proc.stdout or '')[-500:]}")
