"""Compile + correctness + speed verifier for submitted Triton kernels.

Runs OUT OF PROCESS on purpose: a bad or malicious kernel must never corrupt the
trainer or contaminate the next rollout. A segfault here costs one rollout, not
the run.

Submission contract (what the model is trained to emit):
    the submitted string is Python source; it may contain imports; it MUST define

        def triton_forward(*tensors) -> torch.Tensor

    `tensors` = the reference module's forward inputs (`get_inputs()`), followed
    by its parameters in `module.parameters()` order when it has any. A
    parameter-free module therefore gets exactly the forward inputs.

    The reference module is loaded from a file and provides `get_inputs()` and
    `get_init_inputs()`. Two source formats are accepted:
      * KernelBench: class `Model`, `get_init_inputs()` -> positional list;
      * KernelBook: the last `nn.Module` subclass defined in the file is the
        target, `get_init_inputs()` -> `[args_list, kwargs_dict]`.

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
    import ast
    import importlib.util
    import inspect
    import json
    import sys

    import torch
    import triton.testing

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        # Friendly execution namespace: submissions may use torch/np without
        # repeating imports. User imports still work normally.
        mod.__dict__.setdefault("torch", torch)
        try:
            import numpy as np
            mod.__dict__.setdefault("np", np)
        except Exception:
            pass
        try:
            import triton
            import triton.language as tl
            mod.__dict__.setdefault("triton", triton)
            mod.__dict__.setdefault("tl", tl)
        except Exception:
            pass
        spec.loader.exec_module(mod)
        return mod

    REF_PATH, SUB_PATH = sys.argv[1], sys.argv[2]
    N_TRIALS = int(sys.argv[3])
    ATOL = float(sys.argv[4])
    RTOL = float(sys.argv[5])
    sub_source = open(SUB_PATH, encoding="utf-8").read()

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
        "binding": None,
    }

    class AmbiguousReference(Exception):
        pass

    class WrapperContractError(Exception):
        pass

    def _check_binding(fn, n_inputs, n_params):
        # The harness calls triton_forward(*inputs, *model.parameters()). Those must
        # fill exactly the required positional slots: never let a parameter tensor
        # spill into a defaulted hyper-parameter slot (silent mis-binding) and never
        # leave a required slot empty.
        params = list(inspect.signature(fn).parameters.values())
        if any(p.kind == p.VAR_POSITIONAL for p in params):
            return
        required = [
            p.name
            for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty
        ]
        required_kw = [p.name for p in params if p.kind == p.KEYWORD_ONLY and p.default is p.empty]
        if len(required) != n_inputs + n_params or required_kw:
            raise WrapperContractError(
                f"triton_forward requires positional args {required}"
                + (f" and keyword-only {required_kw}" if required_kw else "")
                + f"; the harness passes {n_inputs} input tensor(s) from get_inputs() "
                f"followed by {n_params} parameter tensor(s) from module.parameters()"
            )

    def _target_class(mod, source):
        # `Model` (KernelBench) if defined; otherwise the one module-local nn.Module
        # subclass that no other class in the file instantiates (KernelBook defines
        # building blocks before the module that uses them). Never guess.
        if isinstance(getattr(mod, "Model", None), type):
            return mod.Model
        own = {
            k: v for k, v in vars(mod).items()
            if isinstance(v, type)
            and issubclass(v, torch.nn.Module)
            and v.__module__ == mod.__name__
        }
        tree = ast.parse(source)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        referenced = {
            node.id
            for cls in classes
            for node in ast.walk(cls)
            if isinstance(node, ast.Name) and node.id in own and node.id != cls.name
        }
        roots = [name for name in own if name not in referenced]
        if len(roots) != 1:
            raise AmbiguousReference(
                f"reference must define exactly one root nn.Module class, found {roots}"
            )
        return own[roots[0]]

    def _init_args(init):
        if len(init) == 2 and isinstance(init[0], list) and isinstance(init[1], dict):
            return init[0], init[1]
        return list(init), {}

    try:
        ref_mod = _load(REF_PATH, "reference")
        Model = _target_class(ref_mod, open(REF_PATH, encoding="utf-8").read())
        get_inputs = ref_mod.get_inputs
        get_init_inputs = getattr(ref_mod, "get_init_inputs", lambda: [])
        init_args, init_kwargs = _init_args(get_init_inputs())
        sub_mod = _load(SUB_PATH, "submission")
        triton_forward = sub_mod.triton_forward
        if "@triton.jit" not in sub_source:
            result["error"] = (
                "submission is not a Triton kernel (missing @triton.jit): "
                "do not just call the PyTorch reference"
            )
            print(json.dumps(result))
            raise SystemExit(0)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(result))
        raise SystemExit(0)

    max_diff = 0.0
    correct = True
    produced_output = False
    try:
        for _ in range(N_TRIALS):
            model = Model(*init_args, **init_kwargs).to(device)
            model.eval()
            inputs = [t.to(device) if isinstance(t, torch.Tensor) else t for t in get_inputs()]
            params = [p.detach() for p in model.parameters()]
            result["binding"] = {"inputs": len(inputs), "parameters": len(params)}
            _check_binding(triton_forward, len(inputs), len(params))
            with torch.no_grad():
                ref = model(*inputs)
                out = triton_forward(*inputs, *params)   # first call triggers the Triton JIT
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
                    return triton_forward(*inputs, *params)
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


_MAX_REPORT_ERROR_CHARS = 2000


def format_verify_report(result: dict[str, Any]) -> str:
    """Render a `verify_triton` result as the text the model sees in the REPL output."""
    lines = [
        "[verifier] triton_forward vs reference module",
        f"  compiled: {bool(result.get('compiled'))}",
        f"  correct: {bool(result.get('correct'))}",
    ]
    if result.get("max_diff") is not None:
        lines.append(f"  max_abs_diff: {float(result['max_diff']):.3e}")
    if result.get("speedup") is not None:
        lines.append(
            f"  speedup: {float(result['speedup']):.2f}x "
            f"(reference {float(result['ref_ms']):.4f} ms, kernel {float(result['kernel_ms']):.4f} ms)"
        )
    error = str(result.get("error") or "")
    if error:
        if len(error) > _MAX_REPORT_ERROR_CHARS:
            error = error[:_MAX_REPORT_ERROR_CHARS] + "..."
        lines.append(f"  error: {error}")
    return "\n".join(lines)


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
    _fail = lambda msg: {  # noqa: E731
        "compiled": False,
        "correct": False,
        "max_diff": None,
        "speedup": None,
        "ref_ms": None,
        "kernel_ms": None,
        "error": msg,
        "trials": n_trials,
        "device": "?",
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
            python,
            str(runner_path),
            str(ref_path),
            str(sub_path),
            str(int(n_trials)),
            repr(float(atol)),
            repr(float(rtol)),
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
