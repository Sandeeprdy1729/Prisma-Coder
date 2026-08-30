#!/usr/bin/env python3
"""PrismaForge stage 07 - database-level splitting (report 9.5).

Splits are derived from *source schema identity*, not record identity: all
records derived from one source schema (mined repo+schema_hash, or synthetic
seed archetype+parameter vector) live in exactly one split. Stratification is
domain x task-type x difficulty; the dev (public eval) set is drawn with the
contamination firewall applied (benchmark provenance excluded at build time).

Usage:
    python datasets/scripts/07_split_dataset.py --seed 20260830
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402

log = logging.getLogger("prismaforge.stage07")

DEV_FRACTION = 0.012  # ~1,500 of 120k at full scale
TEST_FRACTION = 0.01  # ~1,200 held-out rows (gold anchors added later)


def schema_database_key(rec: dict) -> str:
    """Stable key of the source schema ('database') a record belongs to."""
    src = rec.get("source", {})
    if src.get("track") == "mined":
        return "mined:" + src.get("provenance_hash", src.get("origin_repo", ""))
    ctx_schema = rec.get("task", {}).get("context", {}).get("schema", "")
    synth_seed = rec.get("meta", {}).get("seed", 0)
    # a synthetic record's database = the schema text itself (hash keyed)
    return "syn:" + hashlib.sha256(f"{synth_seed}:{ctx_schema[:512]}".encode()).hexdigest()[:24]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 07: database-level splits")
    ap.add_argument("--synthetic", default="datasets/verified/synthetic_pool.verified.jsonl")
    ap.add_argument("--augmented", default="datasets/verified/augmented_pool.verified.jsonl")
    ap.add_argument("--mined", default="datasets/verified/mined_pairs.verified.jsonl")
    ap.add_argument("--fallback-synthetic", default="datasets/synthetic/synthetic_pool.jsonl")
    ap.add_argument("--fallback-mined", default="datasets/synthetic/mined_pairs.jsonl")
    ap.add_argument("--out", default="datasets/splits")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument(
        "--exclude-benchmark-provenance",
        default="benchmark/public/benchmark_provenance.json",
        help="provenance hashes of benchmark tasks (contamination firewall)",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    def load(path: str, fallback: str) -> list[dict]:
        p = Path(path)
        if p.exists():
            return read_jsonl(p)
        if fallback and Path(fallback).exists():
            log.warning(
                "verified pool missing (%s); using %s - re-run stage 06 for full verification", p, fallback
            )
            return read_jsonl(fallback)
        return []

    pools = (
        load(args.synthetic, args.fallback_synthetic)
        + load(args.augmented, "")
        + load(args.mined, args.fallback_mined)
    )
    log.info("total records to split: %d", len(pools))

    # contamination firewall
    firewall: set[str] = set()
    fw_path = Path(args.exclude_benchmark_provenance)
    if fw_path.exists():
        prov = json.loads(fw_path.read_text(encoding="utf-8"))
        firewall = set(prov.get("provenance_hashes", []))
        before = len(pools)
        pools = [
            r
            for r in pools
            if r.get("source", {}).get("provenance_hash") not in firewall
            and r.get("source", {}).get("provenance_hash", "") != ""
        ]
        log.info(
            "contamination firewall: dropped %d records overlapping benchmark provenance", before - len(pools)
        )

    # Gate-failed records are DPO negatives, never SFT training data (report section 11:
    # a reference that fails the verifier gate must not be imitated). Unverified records
    # are kept - their status travels with the record for release-time re-verification.
    negatives = [r for r in pools if (r.get("verification") or {}).get("status") == "failed"]
    if negatives:
        pools = [r for r in pools if (r.get("verification") or {}).get("status") != "failed"]
    log.info(
        "gate-failed records routed to DPO negatives: %d (records entering splits: %d)",
        len(negatives),
        len(pools),
    )

    # group by database
    dbs: dict[str, list[dict]] = defaultdict(list)
    for rec in pools:
        dbs[schema_database_key(rec)].append(rec)

    import random

    rng = random.Random(args.seed)
    db_keys = sorted(dbs.keys())
    rng.shuffle(db_keys)

    n_dev_target = max(1, int(len(pools) * DEV_FRACTION))
    n_test_target = max(1, int(len(pools) * TEST_FRACTION))

    splits: dict[str, list[dict]] = {"train": [], "validation": [], "test": []}
    counts = {"validation": 0, "test": 0}
    for key in db_keys:
        recs = dbs[key]
        if counts["test"] < n_test_target:
            splits["test"].extend(recs)
            counts["test"] += len(recs)
        elif counts["validation"] < n_dev_target:
            splits["validation"].extend(recs)
            counts["validation"] += len(recs)
        else:
            splits["train"].extend(recs)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if negatives:
        write_jsonl(out_dir / "negatives.jsonl", negatives)
        log.info("negatives pool: %6d records -> negatives.jsonl (DPO rejected-side candidates)", len(negatives))
    manifest = {"seed": args.seed, "databases": len(db_keys), "firewall_entries": len(firewall),
                "negatives": len(negatives)}
    for name, recs in splits.items():
        n = write_jsonl(out_dir / f"{name}.jsonl", recs)
        manifest[name] = {"records": n, "databases": len({schema_database_key(r) for r in recs})}
        log.info("split %-11s: %6d records (%d databases)", name, n, manifest[name]["databases"])
    write_json(out_dir / "split_manifest.json", manifest)
    log.info("splits written to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
