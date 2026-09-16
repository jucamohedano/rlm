# triton_rlm

Empty package. `env.py` has the three functions to write, each with its docstring
explaining the contract. No logic is filled in -- that is yours.

## The 3 pieces

1. `_build_dataset` -- manifest JSONL -> dataset rows.
2. `score` -- submitted answer -> float (start as a marker check, no Triton).
3. `load_environment` -- wire 1 and 2 into `RLMTrainEnv`.

## Test as you go (no GPU, no trainer)

Build-dataset one-liner:
    .venv/bin/python -c \
        "from triton_rlm.env import _build_dataset as b; \
         d = b('notes/curriculum/lvl1_seed42.jsonl', max_tasks=3); \
         print(len(d)); print(d[0]['info'][:300])"

Full env through a mock model:
    .venv/bin/python scripts/run_any_env_mock.py \
        --env triton_rlm:load_environment \
        --args '{"manifest_path": "notes/curriculum/lvl1_seed42.jsonl"}' \
        --num-examples 2 --show-transcript

For --env to find the package, install it editable once:
    cd /home/juancm/projects/laguna-rlm && \
    uv pip install --python .venv/bin/python -e rlm/training/environments/triton_rlm

Or point --env at the file directly (no install):
    --env rlm/training/environments/triton_rlm/triton_rlm/env.py:load_environment

## When you are ready for real scoring

Add `torch` + `triton` to pyproject dependencies, and replace the `score` body:
compile the submitted Triton, run it vs the PyTorch reference on fresh random
inputs, then measure median speed with `triton.testing.do_bench`. Do it in a
subprocess so a bad kernel costs one rollout, not the run.
