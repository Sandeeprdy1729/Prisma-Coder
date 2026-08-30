#!/usr/bin/env python3
"""PrismaForge stage 02 - extract Prisma artifacts from raw crawl snapshots.

raw/{repo}/ -> processed/extraction_records.jsonl
Each record: schema + PSL parse + call-site census + layer classification.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.extractor import extract_all  # noqa: E402
from prismacoder.hashing import sha256_obj  # noqa: E402
from prismacoder.io_utils import ensure_dir, write_json, write_jsonl  # noqa: E402

log = logging.getLogger("prismaforge.stage02")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 02: artifact extraction")
    ap.add_argument("--raw", default="datasets/raw")
    ap.add_argument("--out", default="datasets/processed")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    raw_root = Path(args.raw)
    if not raw_root.exists():
        log.error("raw root %s missing - run stage 01 first (or crawl offline)", raw_root)
        return 1

    out_dir = ensure_dir(args.out)
    records = extract_all(raw_root)
    payload = []
    for rec in records:
        d = rec.to_dict()
        d["extraction_hash"] = sha256_obj({"schema": rec.schema, "callsites": len(rec.call_sites)})
        payload.append(d)
    n = write_jsonl(out_dir / "extraction_records.jsonl", payload)
    write_json(
        out_dir / "extraction_summary.json",
        {
            "records": n,
            "repos": sorted({r.repo for r in records}),
            "total_call_sites": sum(len(r.call_sites) for r in records),
            "feature_histogram": _hist(records),
        },
    )
    log.info("wrote %d extraction records to %s", n, out_dir / "extraction_records.jsonl")
    return 0


def _hist(records) -> dict[str, int]:
    hist: dict[str, int] = {}
    for rec in records:
        for key, val in rec.features.items():
            if val:
                hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    raise SystemExit(main())
