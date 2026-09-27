"""CPU stage: parquet -> per-trace turns JSONL + task manifest. No torch, no worker.

    uv run python scripts/ppbhatt_parse.py traces.parquet out_dir [--holdout-frac 0.0] [--seed 0]

Writes to out_dir:
    traces.jsonl    one ParsedTrace per line (converted code per turn + their recorded class)
    manifest.jsonl  one task per line: {"_task_id","_n_ops","ops","data_source","code","split","cluster"}
    parse_report.txt  the dry-run: per trace, turns found and their recorded classes

Dedupe is by MinHash over normalised `pytorch_code` and happens BEFORE the split;
the split is assigned per cluster so near-duplicate tasks never straddle it.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pyarrow.parquet as pq
from triton_rlm.trace_translate import (
    DATASET_ID,
    dedupe_clusters,
    normalize_pytorch,
    parse_trace,
    split_for_cluster,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("parquet")
    ap.add_argument("out_dir")
    ap.add_argument("--source", default="kernelbook", help="keep rows with this `source` (rule 7)")
    ap.add_argument("--holdout-frac", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minhash-threshold", type=float, default=0.85)
    args = ap.parse_args()

    rows = [r for r in pq.read_table(args.parquet).to_pylist() if r["source"] == args.source]
    clusters = dedupe_clusters(
        [normalize_pytorch(r["pytorch_code"]) for r in rows], args.minhash_threshold
    )
    cluster_key = {
        c: min(r["sample_key"] for r, cc in zip(rows, clusters, strict=True) if cc == c)
        for c in set(clusters)
    }
    traces = [
        parse_trace(r, c, split_for_cluster(cluster_key[c], args.holdout_frac, args.seed))
        for r, c in zip(rows, clusters, strict=True)
    ]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "traces.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_json()) + "\n")
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(
                json.dumps(
                    {
                        "_task_id": t.sample_key,
                        "_n_ops": len(t.ops),
                        "ops": ", ".join(t.ops),
                        "data_source": DATASET_ID,
                        "code": t.pytorch_code,
                        "split": t.split,
                        "cluster": t.cluster,
                    }
                )
                + "\n"
            )

    lines = []
    n_turns = 0
    n_no_code = 0
    turn_hist: collections.Counter[int] = collections.Counter()
    for t in traces:
        classes = " -> ".join(
            (turn.recorded.short() if turn.code is not None else "NO_CODE") for turn in t.turns
        )
        n_turns += len(t.turns)
        n_no_code += sum(turn.code is None for turn in t.turns)
        turn_hist[len(t.turns)] += 1
        lines.append(
            f"{t.sample_key:18s} cluster={t.cluster:3d} split={t.split:7s} turns={len(t.turns)} "
            f"stop={t.stop_reason:17s} ops={len(t.ops):2d}  {classes}"
        )
    dup_clusters = [c for c, n in collections.Counter(clusters).items() if n > 1]
    summary = [
        f"rows kept (source={args.source}): {len(traces)}",
        f"turns: {n_turns}  (per-trace histogram {dict(sorted(turn_hist.items()))})",
        f"turns without a <triton> block: {n_no_code}",
        f"distinct tasks after dedupe: {len(set(clusters))}  "
        f"(clusters with >1 trace: {len(dup_clusters)}: "
        f"{[[r['sample_key'] for r, c in zip(rows, clusters, strict=True) if c == d] for d in dup_clusters]})",
        f"split: {dict(collections.Counter(t.split for t in traces))}",
    ]
    report = "\n".join(lines + [""] + summary) + "\n"
    (out / "parse_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
