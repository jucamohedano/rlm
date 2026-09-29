"""Add an opt-in ``per_sample_loss`` to a prime-rl v0.7.0 SFT trainer checkout.

prime-rl's SFT loss is token-weighted: every supervised token contributes equally to the
gradient and to the reported train/validation loss, so a family with long targets
dominates a family with short ones. With ``per_sample_loss = true`` each packed sample
(a run of ``position_ids`` starting at 0 -- one SFT row under both ``pack_function``
values) contributes the mean over its own supervised tokens, and the step/validation
loss is the mean over supervised samples. Only the fused ``loss_impl`` paths are
patched; the config rejects the flag for the unfused ones.

Usage:
    uv run scripts/patch_prime_rl_sample_loss.py /path/to/prime-rl

Every anchor must match exactly once; the script fails loudly otherwise, and refuses to
patch a file that already carries the change.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CONFIG_REL = Path("packages/prime-rl-configs/src/prime_rl/configs/sft.py")
TRAINER_REL = Path("src/prime_rl/trainer/sft/train.py")

CFG_FIELD_OLD = '''    loss_impl: Literal["liger", "torch", "liger_fused", "quack_fused"] = "torch"
    """Cross-entropy loss implementation. ``liger_fused`` fuses the lm_head projection with the CE loss to avoid materializing full logits. ``quack_fused`` uses quack-kernels for chunked linear + CE with CuTe DSL CUDA kernels."""
'''

CFG_FIELD_NEW = (
    CFG_FIELD_OLD
    + '''
    per_sample_loss: bool = False
    """Normalize the fused cross-entropy loss per supervised sample instead of per token. A sample is a run of ``position_ids`` starting at 0 (one SFT row under both pack functions); each contributes the mean over its supervised tokens and the step/validation loss is the mean over supervised samples. Requires a fused ``loss_impl``."""
'''
)

CFG_VALIDATOR_OLD = """    @model_validator(mode="after")
    def validate_seq_len(self):
"""

CFG_VALIDATOR_NEW = """    @model_validator(mode="after")
    def validate_per_sample_loss(self):
        if self.per_sample_loss and self.loss_impl not in ("liger_fused", "quack_fused"):
            raise ValueError("per_sample_loss requires loss_impl to be 'liger_fused' or 'quack_fused'")
        return self

    @model_validator(mode="after")
    def validate_seq_len(self):
"""

TRAINER_OLD = """            if config.loss_impl in ("liger_fused", "quack_fused"):
                masked_target_ids = target_ids.clone()
                masked_target_ids[~loss_mask] = FUSED_CE_IGNORE_INDEX
                out = forward(model, input_ids, position_ids, labels=masked_target_ids)
                loss_sum = out["loss"] * token_count
            else:
"""

TRAINER_NEW = """            if config.loss_impl in ("liger_fused", "quack_fused"):
                masked_target_ids = target_ids.clone()
                masked_target_ids[~loss_mask] = FUSED_CE_IGNORE_INDEX
                if config.per_sample_loss:
                    assert not cp_enabled, "per_sample_loss does not support context parallelism"
                    flat_input_ids = input_ids.reshape(1, -1)
                    flat_position_ids = position_ids.reshape(1, -1)
                    flat_labels = masked_target_ids.reshape(1, -1)
                    flat_loss_mask = loss_mask.reshape(-1)
                    is_start = flat_position_ids[0] == 0
                    starts = is_start.nonzero(as_tuple=True)[0]
                    assert starts.numel() > 0 and starts[0].item() == 0, "packed sequence must start at position 0"
                    ends = torch.cat([starts[1:], starts.new_tensor([flat_loss_mask.numel()])])
                    sample_ids = torch.cumsum(is_start.to(torch.int64), dim=0) - 1
                    sample_tokens = torch.zeros(starts.numel(), dtype=torch.int64, device=flat_loss_mask.device)
                    sample_tokens.index_add_(0, sample_ids, flat_loss_mask.to(torch.int64))
                    sample_means = []
                    for start, end, n in zip(starts.tolist(), ends.tolist(), sample_tokens.tolist()):
                        if n == 0:
                            continue
                        if config.model.lora is not None:
                            set_lora_num_tokens(torch.full((1,), end - start, dtype=torch.int32, device="cuda"))
                        out = forward(
                            model,
                            flat_input_ids[:, start:end],
                            flat_position_ids[:, start:end],
                            labels=flat_labels[:, start:end],
                        )
                        sample_means.append(out["loss"].reshape(()))
                    assert sample_means, "micro-batch has no supervised samples"
                    loss_sum = torch.stack(sample_means).sum()
                    token_count = torch.tensor(len(sample_means), dtype=torch.int64, device="cuda")
                else:
                    out = forward(model, input_ids, position_ids, labels=masked_target_ids)
                    loss_sum = out["loss"] * token_count
            else:
"""


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        raise SystemExit(f"{path}: already patched")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected exactly one anchor match, found {count}")
    path.write_text(text.replace(old, new, 1))


def patch_config(path: Path) -> None:
    replace_once(path, CFG_FIELD_OLD, CFG_FIELD_NEW)
    replace_once(path, CFG_VALIDATOR_OLD, CFG_VALIDATOR_NEW)


def patch_trainer(path: Path) -> None:
    replace_once(path, TRAINER_OLD, TRAINER_NEW)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "prime_rl_root", type=Path, help="prime-rl v0.7.0 checkout to patch in place"
    )
    args = parser.parse_args(argv)
    root: Path = args.prime_rl_root
    patch_config(root / CONFIG_REL)
    patch_trainer(root / TRAINER_REL)
    print(f"patched {root / CONFIG_REL}")
    print(f"patched {root / TRAINER_REL}")


if __name__ == "__main__":
    main(sys.argv[1:])
