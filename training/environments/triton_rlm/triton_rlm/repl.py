"""REPL backend that verifies every ```repl``` block defining `triton_forward`.

The model never writes its own test calls in the seed data (the source traces
were verified harness-side), so the harness does the check: when a block
defines `triton_forward` and runs without raising, the block's source is
verified as a standalone submission against the reference module and the
report is appended to the block's stdout. The source is also stored in the REPL
variable `kernel_src` so the submission idiom is a fixed two-liner:

    answer["content"] = kernel_src
    answer["ready"] = True

The block is verified on its own (not the accumulated REPL namespace) because
that is exactly what the scorer does with the submission; the prompt states
this contract.
"""

from __future__ import annotations

import ast
import asyncio
from collections.abc import Callable
from typing import Any

from rlm_train.repl.base import ExecResult, ReplBackend

from triton_rlm.verifier import format_verify_report, verify_triton

KERNEL_SRC_VAR = "kernel_src"
ENTRY_POINT = "triton_forward"

Verifier = Callable[[str, str], dict[str, Any]]


def defines_entry_point(code: str) -> bool:
    """True iff `code` parses and defines a module-level function `triton_forward`."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == ENTRY_POINT
        for node in tree.body
    )


class VerifyingReplBackend(ReplBackend):
    def __init__(self, inner: ReplBackend, reference_code: str, verifier: Verifier = verify_triton):
        self._inner = inner
        self._reference = reference_code
        self._verifier = verifier
        self.last_verify: dict[str, Any] | None = None

    async def start(self, proxy_url: str, rollout_id: str, depth: int = 1) -> None:
        await self._inner.start(proxy_url, rollout_id, depth)

    async def load_context(self, payload: Any, index: int | None = None) -> int:
        return await self._inner.load_context(payload, index)

    async def bootstrap(self, code: str) -> None:
        await self._inner.bootstrap(code)

    async def set_local(self, name: str, value: Any) -> None:
        await self._inner.set_local(name, value)

    async def execute(self, code: str) -> ExecResult:
        result = await self._inner.execute(code)
        if result.exception is not None or not defines_entry_point(code):
            return result
        report = await asyncio.to_thread(self._verifier, self._reference, code)
        self.last_verify = report
        await self._inner.set_local(KERNEL_SRC_VAR, code)
        sep = "\n" if result.stdout and not result.stdout.endswith("\n") else ""
        result.stdout = f"{result.stdout}{sep}{format_verify_report(report)}\n"
        return result

    async def stop(self) -> None:
        await self._inner.stop()
