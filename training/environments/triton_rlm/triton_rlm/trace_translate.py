"""Translate ppbhatt500/kernelbook-triton-multiturn-reasoning-traces into RLM trajectories.

Pure functions only (no torch, no worker): the CPU parse stage
(`scripts/ppbhatt_parse.py`) and the GPU re-execution stage
(`scripts/ppbhatt_reexec.py`) both import from here.

Source format (one parquet, 207 rows): each row is one multi-turn trace of a
teacher writing a Triton kernel for a KernelBook/KernelBench PyTorch module.
`turns[k]` holds `full_completion` (the assistant message: one `<triton>...</triton>`
block, no prose), `result` (their harness verdict) and `feedback_given` (the
user message the NEXT turn was conditioned on). `full_messages` omits the final
assistant turn, so code is taken from `turns`, never from `full_messages`.

Translation rules (why each exists is in the notes that led here):

  1. Their harness feedback is never reused. Every turn's code is re-executed in
     our worker and our REPL output replaces theirs. A turn k>0 is a valid target
     only if it was conditioned on feedback of the same KIND our worker produces
     for turn k-1: {error(type, message head), incorrect, correct_slow,
     correct_fast}. On the first mismatch the trace is truncated there.
  2. Their speed numbers (256-element inputs on an H100) are ignored; only the
     correctness bit of their verdict is used, and only to classify feedback.
  3. Keep the PREFIX through the first turn our verifier calls correct. Never
     end a trajectory on a failure. After a correct-but-slow turn, keep the next
     turn only if their feedback was also correct-but-slow and our verifier says
     the next turn is correct AND faster.
  4. The KernelBook reference Triton (inductor output) is not in this dataset
     and is never a target; the target is the teacher's code.
  5. Protocol conversion is mechanical: `<triton>` block -> ```repl fence,
     `triton_kernel_wrapper` -> `triton_forward`. Reasoning is dropped.
  6. Nothing model-visible is fabricated except ONE fixed submit idiom appended
     after the kept correct turn (the contract is stated in the prompt).
  7. `kernelbench_*` rows are excluded (KernelBench may be reported as eval).

Class compatibility for `error` is exception type equality plus a fuzzy match
on the first message line (paths/line numbers stripped), because their message
carries their harness's file paths and Python's SyntaxError formatting differs
across versions; the ratio is recorded so the threshold can be tightened.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import io
import json
import re
import tokenize
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DATASET_ID = "ppbhatt500/kernelbook-triton-multiturn-reasoning-traces"
SOURCE_ENTRY_POINT = "triton_kernel_wrapper"
ENTRY_POINT = "triton_forward"
SUBMIT_IDIOM = 'answer["content"] = kernel_src\nanswer["ready"] = True'

Kind = Literal["error", "incorrect", "correct_slow", "correct_fast"]

# Strings that exist only in the source harness (its prompt, driver and feedback).
# None may appear in an emitted trajectory: a hit in our own prompt/REPL text is a
# translation bug; a hit in the teacher's code means that turn was written against
# their feedback text, not ours.
SOURCE_HARNESS_MARKERS = (
    "n_required",
    "/root/modal_app.py",
    "benchmark_kernelbench",
    "Compilation/runtime error:",
    "Incorrect output",
    "Correct but slow",
    "TRITON PRIMER",
)


def source_harness_markers(text: str) -> list[str]:
    return [m for m in SOURCE_HARNESS_MARKERS if m in text]


_TRITON_BLOCK = re.compile(r"<triton>\s*\n?(.*?)\n?\s*</triton>", re.S)
_EXC_HEAD = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning)?)\s*:\s*(.*)$")


@dataclass
class TurnClass:
    kind: Kind
    exc_type: str = ""
    message: str = ""
    speedup: float | None = None
    # (exc_type, message) of the exceptions chained behind `exc_type`, outermost first
    chain: tuple[tuple[str, str], ...] = ()

    def short(self) -> str:
        if self.kind == "error":
            return f"error[{self.exc_type}]"
        if self.kind == "correct_slow" and self.speedup is not None:
            return f"correct_slow[{self.speedup:.2f}x]"
        return self.kind


def _split_exception(text: str) -> tuple[str, str]:
    first = text.strip().splitlines()[0] if text.strip() else ""
    m = _EXC_HEAD.match(first)
    if m is None:
        return ("Error", first)
    return (m.group(1).rsplit(".", 1)[-1], m.group(2))


def classify_recorded(turn: dict[str, Any]) -> TurnClass:
    """Their verdict for one turn, read from the feedback the next turn saw.

    The last turn of a trace has no feedback; its class comes from `result` and
    is never used for a compatibility check (nothing follows it).
    """
    feedback = turn.get("feedback_given") or ""
    result = turn["result"]
    if feedback.startswith("Compilation/runtime error:"):
        exc_type, message = _split_exception(feedback.split("\n", 1)[1])
        return TurnClass("error", exc_type, message)
    if feedback.startswith("Incorrect output"):
        return TurnClass("incorrect")
    if feedback.startswith("Correct but slow"):
        return TurnClass("correct_slow", speedup=result.get("speedup"))
    if feedback:
        raise ValueError(f"unrecognised feedback head: {feedback[:80]!r}")
    if result["correctness"] and not result.get("error"):
        speedup = float(result.get("speedup") or 0.0)
        return TurnClass("correct_fast" if speedup >= 1.0 else "correct_slow", speedup=speedup)
    if result.get("error"):
        exc_type, message = _split_exception(result["error"])
        return TurnClass("error", exc_type, message)
    return TurnClass("incorrect")


def classify_ours(exec_exception: str | None, verify: dict[str, Any] | None) -> TurnClass:
    """Our verdict: the worker exec outcome, then the verifier report (if the block ran)."""
    if exec_exception is not None:
        exc_type, message = _split_exception(exec_exception)
        return TurnClass("error", exc_type, message)
    if verify is None:
        # ran fine but never defined `triton_forward`: nothing to verify
        return TurnClass("error", "NoEntryPoint", f"block does not define {ENTRY_POINT}")
    error = str(verify.get("error") or "")
    if verify.get("correct"):
        speedup = float(verify.get("speedup") or 0.0)
        return TurnClass("correct_fast" if speedup >= 1.0 else "correct_slow", speedup=speedup)
    if not verify.get("compiled"):
        if error.startswith("submission is not a Triton kernel"):
            return TurnClass("error", "NotTritonKernel", error)
        if error.startswith("verification timed out"):
            return TurnClass("error", "TimeoutError", error)
        if error.startswith("verifier crashed"):
            return TurnClass("error", "VerifierCrash", error)
        exc_type, message = _split_exception(error)
        chain = tuple(_split_exception(str(c)) for c in verify.get("error_chain") or [])
        return TurnClass("error", exc_type, message, chain=chain)
    return TurnClass("incorrect", message=error)


_NOISE = re.compile(r"\(<[^>]*>, line \d+\)|\(line \d+\)|/[\w./-]+\.py|0x[0-9a-f]+|\d+")


def _norm_head(message: str) -> str:
    return " ".join(_NOISE.sub(" ", message.lower()).split())


def error_head_ratio(a: TurnClass, b: TurnClass) -> float:
    return _head_ratio(a.message, b.message)


def _head_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, _norm_head(a), _norm_head(b)).ratio()


# Python's own arity error on the entry-point call, i.e. the harness could not fill the
# wrapper's signature. Our verifier reports the same failure mode as WrapperContractError.
_ARITY_TYPE_ERROR = re.compile(
    r"^\w+\(\) (missing \d+ required (positional|keyword-only) argument|takes \d+ positional argument)"
)


def compatible(recorded: TurnClass, ours: TurnClass, min_head_ratio: float) -> bool:
    if recorded.kind != ours.kind:
        return False
    if recorded.kind != "error":
        return True
    if ours.exc_type == "WrapperContractError":
        return recorded.exc_type == "TypeError" and bool(_ARITY_TYPE_ERROR.match(recorded.message))
    # Their harness reported the innermost exception of a chain (e.g. the TypeError that
    # Triton re-raises as CompilationError), so ours may match at any depth of the chain.
    return any(
        recorded.exc_type == exc_type and _head_ratio(recorded.message, message) >= min_head_ratio
        for exc_type, message in ((ours.exc_type, ours.message), *ours.chain)
    )


def extract_code(full_completion: str | None) -> str | None:
    """The single `<triton>` block of an assistant message, or None."""
    if not full_completion:
        return None
    blocks = _TRITON_BLOCK.findall(full_completion)
    if len(blocks) != 1:
        return None
    return blocks[0].strip("\n")


_WRAPPER_NAME = re.compile(rf"\b{SOURCE_ENTRY_POINT}\b")


def convert_source(code: str) -> str:
    """Rename the entry point; nothing else changes (same line count, same body)."""
    return _WRAPPER_NAME.sub(ENTRY_POINT, code)


_NOT_OPS = {
    "nn.Module",
    "torch.nn.Module",
    "torch.rand",
    "torch.randn",
    "torch.ones",
    "torch.zeros",
}


def _torch_aliases(tree: ast.Module) -> set[str]:
    names = {"torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(
                (a.asname or a.name).split(".")[0] for a in node.names if a.name.startswith("torch")
            )
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("torch"):
            names.update(a.asname or a.name for a in node.names)
    return names


def extract_ops(pytorch_code: str) -> list[str]:
    """Heuristic operator list: torch-rooted callables anywhere in the module, plus
    tensor method calls and binary operators inside `forward`, in order of first use.
    KernelBook has no operator metadata, so this stands in for `ops`/`_n_ops`."""
    tree = ast.parse(pytorch_code)
    roots = _torch_aliases(tree)
    ops: list[str] = []

    def add(name: str) -> None:
        if name not in ops and name not in _NOT_OPS:
            ops.append(name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            root = node.func
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in roots:
                add(ast.unparse(node.func))
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "forward":
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    root = node.func
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if not (isinstance(root, ast.Name) and root.id in roots):
                        add(f".{node.func.attr}")
                elif isinstance(node, ast.BinOp):
                    add(type(node.op).__name__)
    return ops


class AmbiguousReference(ValueError):
    """The reference module does not have exactly one root nn.Module class."""


def reference_root_class(pytorch_code: str) -> str:
    """Name of the class the verifier instantiates as the reference model.

    `Model` if defined (KernelBench). Otherwise the one torch-derived class (any
    base rooted in a torch import, e.g. `nn.Module`, `nn.Conv1d`, `_Loss`, or a
    local subclass of one) that no other class in the file instantiates:
    KernelBook files define building blocks (e.g. `BasicBlock`) before the module
    that uses them, and the module is the task. Zero or several such roots raise
    `AmbiguousReference` rather than guess. The verifier applies the same rule at
    runtime with `issubclass(cls, nn.Module)` and is authoritative.
    """
    tree = ast.parse(pytorch_code)
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if "Model" in classes:
        return "Model"
    torch_names = _torch_aliases(tree)

    def is_module(cls: ast.ClassDef, seen: frozenset[str]) -> bool:
        for base in cls.bases:
            root = base
            while isinstance(root, ast.Attribute):
                root = root.value
            name = root.id if isinstance(root, ast.Name) else ""
            if name in torch_names:
                return True
            if name in classes and name not in seen:
                if is_module(classes[name], seen | {cls.name}):
                    return True
        return False

    referenced = {
        node.id
        for cls in classes.values()
        for node in ast.walk(cls)
        if isinstance(node, ast.Name) and node.id in classes and node.id != cls.name
    }
    roots = [
        name
        for name, cls in classes.items()
        if is_module(cls, frozenset()) and name not in referenced
    ]
    if len(roots) != 1:
        raise AmbiguousReference(f"expected exactly one root nn.Module class, found {roots}")
    return roots[0]


def normalize_pytorch(code: str) -> str:
    """Strip comments/docstrings and collapse whitespace so cosmetic edits dedupe."""
    out: list[str] = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type in (
            tokenize.COMMENT,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.ENCODING,
            tokenize.ENDMARKER,
        ):
            continue
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.DEDENT,
        ):
            prev_type = tok.type
            continue
        out.append(tok.string)
        prev_type = tok.type
    return " ".join(out)


def _shingles(text: str, k: int = 4) -> set[str]:
    toks = text.split()
    if len(toks) <= k:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def minhash_signature(text: str, n_perm: int = 64) -> list[int]:
    sig = [1 << 64] * n_perm
    for sh in _shingles(text):
        for i in range(n_perm):
            h = int.from_bytes(
                hashlib.blake2b(sh.encode(), digest_size=8, salt=i.to_bytes(2, "big")).digest(),
                "big",
            )
            if h < sig[i]:
                sig[i] = h
    return sig


def dedupe_clusters(texts: list[str], threshold: float = 0.85, n_perm: int = 64) -> list[int]:
    """Cluster ids (0..) such that near-duplicates (est. Jaccard >= threshold) share an id."""
    sigs = [minhash_signature(t, n_perm) for t in texts]
    parent = list(range(len(texts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            agree = sum(a == b for a, b in zip(sigs[i], sigs[j], strict=True)) / n_perm
            if agree >= threshold:
                parent[find(i)] = find(j)
    roots = {find(i) for i in range(len(texts))}
    ids = {r: k for k, r in enumerate(sorted(roots))}
    return [ids[find(i)] for i in range(len(texts))]


def split_for_cluster(cluster_key: str, holdout_frac: float, seed: int) -> str:
    """Deterministic task-level train/holdout assignment (hash of cluster key + seed)."""
    h = int.from_bytes(
        hashlib.blake2b(f"{seed}:{cluster_key}".encode(), digest_size=8).digest(), "big"
    )
    return "holdout" if (h / 2**64) < holdout_frac else "train"


@dataclass
class ParsedTurn:
    index: int
    code: str | None  # converted source (entry point renamed), None if no <triton> block
    recorded: TurnClass


@dataclass
class ParsedTrace:
    sample_key: str
    source: str
    pytorch_code: str
    stop_reason: str
    ops: list[str]
    root_class: str
    cluster: int
    split: str
    turns: list[ParsedTurn] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_json(d: dict[str, Any]) -> ParsedTrace:
        turns = [ParsedTurn(t["index"], t["code"], TurnClass(**t["recorded"])) for t in d["turns"]]
        return ParsedTrace(**{**{k: v for k, v in d.items() if k != "turns"}, "turns": turns})


def parse_trace(row: dict[str, Any], cluster: int, split: str) -> ParsedTrace:
    turns_raw = json.loads(row["turns"]) if isinstance(row["turns"], str) else row["turns"]
    turns = []
    for k, t in enumerate(turns_raw):
        code = extract_code(t.get("full_completion"))
        turns.append(
            ParsedTurn(k, convert_source(code) if code is not None else None, classify_recorded(t))
        )
    return ParsedTrace(
        sample_key=row["sample_key"],
        source=row["source"],
        pytorch_code=row["pytorch_code"],
        stop_reason=row["stop_reason"],
        ops=extract_ops(row["pytorch_code"]),
        root_class=reference_root_class(row["pytorch_code"]),
        cluster=cluster,
        split=split,
        turns=turns,
    )


@dataclass
class Decision:
    action: Literal["execute_next", "keep", "drop"]
    keep_end: int | None  # index of the last kept (correct) turn when action == "keep"
    reason: str


def decide(recorded: list[TurnClass], ours: list[TurnClass], min_head_ratio: float) -> Decision:
    """Apply rules 1 and 3 to the turns executed so far.

    `recorded[i]` is their feedback after turn i (what turn i+1 saw); `ours[i]`
    is our re-execution of turn i. Called after each execution; the caller
    executes turn len(ours) on "execute_next".
    """
    slow_end: int | None = None
    for i, c in enumerate(ours):
        if c.kind == "correct_fast":
            return Decision(
                "keep", i, "correct" if slow_end is None else "faster than correct-slow prefix"
            )
        if c.kind == "correct_slow":
            if slow_end is not None and (c.speedup or 0.0) <= (ours[slow_end].speedup or 0.0):
                return Decision("keep", slow_end, f"turn {i} correct but not faster")
            slow_end = i
            has_next = i + 1 < len(recorded)
            if has_next and recorded[i].kind == "correct_slow":
                if i + 1 == len(ours):
                    return Decision("execute_next", slow_end, "try slow-conditioned continuation")
                continue
            return Decision(
                "keep", i, "correct_slow" + ("; their feedback was not slow" if has_next else "")
            )
        # error / incorrect
        if slow_end is not None:
            return Decision("keep", slow_end, f"turn {i} {c.short()} after correct-slow")
        if i + 1 >= len(recorded):
            return Decision("drop", None, f"ends on {c.short()}")
        if not compatible(recorded[i], c, min_head_ratio):
            return Decision(
                "drop",
                None,
                f"feedback mismatch at turn {i}: theirs {recorded[i].short()} ours {c.short()}",
            )
        if i + 1 == len(ours):
            return Decision("execute_next", None, "")
    return Decision("execute_next", None, "")
