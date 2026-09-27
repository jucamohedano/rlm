"""Run the real verifier runner on CPU with a stub `triton` (no GPU needed).

Needs an interpreter with torch: `RLM_VERIFY_PYTHON` or the current one.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from triton_rlm.verifier import verify_triton

PYTHON = os.environ.get("RLM_VERIFY_PYTHON") or sys.executable
HAS_TORCH = subprocess.run([PYTHON, "-c", "import torch"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason=f"{PYTHON} has no torch")

STUB_TRITON = {
    "triton/__init__.py": "def jit(fn):\n    return fn\n\ndef cdiv(a, b):\n    return -(-a // b)\n",
    "triton/language.py": "constexpr = int\n",
    "triton/testing.py": "def do_bench(fn, **kw):\n    fn()\n    return 1.0\n",
}

REF_SINGLE = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class GetMask(nn.Module):
        def __init__(self, pad_idx=0):
            super(GetMask, self).__init__()
            self.pad_idx = pad_idx

        def forward(self, x):
            return (x != self.pad_idx).float()

    def get_inputs():
        return [torch.randint(0, 3, (4, 4))]

    def get_init_inputs():
        return [[], {}]
    """
)

REF_PARAM = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class BiasLayer(nn.Module):
        def __init__(self, n=4):
            super().__init__()
            self.bias = nn.Parameter(torch.randn(n))

        def forward(self, x):
            return x + self.bias

    def get_inputs():
        return [torch.rand(4, 4)]

    def get_init_inputs():
        return [[], {'n': 4}]
    """
)

REF_MULTI = textwrap.dedent(
    """
    import torch
    import torch.nn as nn

    class BasicBlock(nn.Module):
        def __init__(self):
            super(BasicBlock, self).__init__()

        def forward(self, x):
            return x * 2

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = BasicBlock()

        def forward(self, x):
            return self.block(x) + 1

    def get_inputs():
        return [torch.rand(4, 4)]

    def get_init_inputs():
        return [[], {}]
    """
)


def submission(sig: str, body: str) -> str:
    return (
        "import torch\nimport triton\nimport triton.language as tl\n\n"
        "@triton.jit\ndef k(x_ptr):\n    pass\n\n"
        f"def triton_forward({sig}):\n    return {body}\n"
    )


@pytest.fixture()
def run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for rel, src in STUB_TRITON.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(src, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return lambda ref, sub: verify_triton(ref, sub, n_trials=2, python=PYTHON)


def test_kernelbook_reference_single_input(run) -> None:
    r = run(REF_SINGLE, submission("x", "(x != 0).float()"))
    assert r["error"] == ""
    assert r["compiled"] and r["correct"]
    assert r["binding"] == {"inputs": 1, "parameters": 0}


def test_parameters_follow_inputs_and_defaults_are_free(run) -> None:
    r = run(REF_PARAM, submission("x, bias, n=4", "x + bias"))
    assert r["correct"], r["error"]
    assert r["binding"] == {"inputs": 1, "parameters": 1}


def test_binding_mismatch_is_a_loud_contract_error(run) -> None:
    for sig in ("x", "x, bias, extra", "x, bias, *, n"):
        r = run(REF_PARAM, submission(sig, "x"))
        assert not r["compiled"] and not r["correct"]
        assert r["error"].startswith("WrapperContractError:"), r["error"]


def test_varargs_wrapper_is_accepted(run) -> None:
    r = run(REF_PARAM, submission("*ts", "ts[0] + ts[1]"))
    assert r["correct"], r["error"]


def test_root_module_is_the_uninstantiated_one(run) -> None:
    r = run(REF_MULTI, submission("x", "x * 2 + 1"))
    assert r["correct"], r["error"]
    r = run(REF_MULTI, submission("x", "x * 2"))
    assert r["compiled"] and not r["correct"]


def test_ambiguous_reference_is_rejected(run) -> None:
    ref = REF_MULTI.replace("self.block = BasicBlock()", "pass").replace("self.block(x)", "x * 2")
    r = run(ref, submission("x", "x * 2 + 1"))
    assert not r["correct"]
    assert r["error"].startswith("AmbiguousReference:"), r["error"]
