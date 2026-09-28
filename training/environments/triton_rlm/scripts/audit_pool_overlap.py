"""CPU check: is an env task pool (e.g. the ops6k curriculum manifests) disjoint from the
ppbhatt clusters? Same normalisation and MinHash as `ppbhatt_parse` (data vs data, no model).

    uv run python scripts/audit_pool_overlap.py \
        --ppbhatt-manifest /outputs/.../ppbhatt_holdout_parse/manifest.jsonl \
        --pool notes/curriculum/lvl1_seed42.jsonl notes/curriculum/lvl2_seed42.jsonl ... \
        --out notes/curriculum/overlap_vs_ppbhatt.json

Two passes: the parse rule (union clustered at --threshold, default 0.85; a pool task is a hit
if it lands in a cluster with any ppbhatt task) and a looser report of every pool task whose
best estimated Jaccard against any ppbhatt task is >= --report-above (default 0.7), so the
threshold's cliff is visible. Hits are resolved by dropping the POOL task (the ppbhatt side is
verified SFT data; the pool is the cheap side): `<pool>_disjoint.jsonl` is written next to each
pool file with the hits removed. Exit code 1 if there is any hit at --threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from triton_rlm.trace_translate import (  # noqa: E402
    dedupe_clusters,
    minhash_signature,
    normalize_pytorch,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def agreement(a: list[int], b: list[int]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ppbhatt-manifest", type=Path, required=True)
    ap.add_argument("--pool", type=Path, nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--report-above", type=float, default=0.7)
    ap.add_argument("--n-perm", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ppb = load_jsonl(args.ppbhatt_manifest)
    ppb_norm = [normalize_pytorch(r["code"]) for r in ppb]
    ppb_sigs = [minhash_signature(t, args.n_perm) for t in ppb_norm]

    report: dict = {
        "threshold": args.threshold,
        "report_above": args.report_above,
        "n_perm": args.n_perm,
        "ppbhatt_tasks": len(ppb),
        "ppbhatt_clusters": len({r["cluster"] for r in ppb}),
        "pools": {},
    }
    any_hit = False
    for pool_path in args.pool:
        pool = load_jsonl(pool_path)
        pool_norm = [normalize_pytorch(r["code"]) for r in pool]
        # pass 1: the parse rule, on the union
        union = dedupe_clusters(ppb_norm + pool_norm, args.threshold, args.n_perm)
        ppb_cluster_ids = set(union[: len(ppb)])
        hits = [i for i, c in enumerate(union[len(ppb) :]) if c in ppb_cluster_ids]
        # pass 2: best pairwise estimated Jaccard per pool task
        near: list[dict] = []
        for i, t in enumerate(pool_norm):
            sig = minhash_signature(t, args.n_perm)
            best_j, best = max(
                ((j, agreement(sig, s)) for j, s in enumerate(ppb_sigs)),
                key=lambda p: p[1],
            )
            if best >= args.report_above:
                near.append(
                    {
                        "pool_task": pool[i]["_task_id"],
                        "ppbhatt_task": ppb[best_j]["_task_id"],
                        "ppbhatt_cluster": ppb[best_j]["cluster"],
                        "est_jaccard": round(best, 3),
                        "hit_at_threshold": i in set(hits),
                    }
                )
        any_hit |= bool(hits)
        survivors = [r for i, r in enumerate(pool) if i not in set(hits)]
        disjoint = pool_path.with_name(pool_path.stem + "_disjoint.jsonl")
        with disjoint.open("w", encoding="utf-8") as f:
            for r in survivors:
                f.write(json.dumps(r) + "\n")
        report["pools"][str(pool_path)] = {
            "tasks": len(pool),
            "hits_at_threshold": [pool[i]["_task_id"] for i in hits],
            "near_matches": sorted(near, key=lambda d: -d["est_jaccard"]),
            "disjoint_written": str(disjoint),
            "disjoint_tasks": len(survivors),
        }
        print(
            f"{pool_path}: {len(pool)} tasks, {len(hits)} hit(s) at {args.threshold}, "
            f"{len(near)} within {args.report_above}; wrote {disjoint} ({len(survivors)})"
        )

    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report: {args.out}")
    if any_hit:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
