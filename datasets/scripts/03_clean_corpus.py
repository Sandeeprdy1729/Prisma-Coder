#!/usr/bin/env python3
"""PrismaForge stage 03 - cleaning pipeline (report 9.2, Part 3).

Per record:
  1. license gate          (permissive only; copyleft rejected)
  2. secret scan           (HIGH -> drop; MEDIUM -> redact)
  3. PII scrub             (structural placeholders)
  4. version normalization (v7 dialect)
  5. quality filter        (broken-schema detection, scoring)
  6. dedup                 (exact + MinHash-LSH + isomorphism)

Output: processed/clean_records.jsonl + clean_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.dedup import Deduplicator  # noqa: E402
from prismacoder.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from prismacoder.license_filter import check_license  # noqa: E402
from prismacoder.pii import scrub_text  # noqa: E402
from prismacoder.quality import check_schema  # noqa: E402
from prismacoder.secrets import scan_and_clean  # noqa: E402
from prismacoder.version_norm import detect_version, normalize_to_v7  # noqa: E402

log = logging.getLogger("prismaforge.stage03")


def clean_record(rec: dict, *, drop_stats: dict[str, int]) -> dict | None:
    # 1. license
    dec = check_license(rec.get("license", "UNKNOWN"))
    if not dec.allowed:
        drop_stats["license"] = drop_stats.get("license", 0) + 1
        return None
    rec["license"] = dec.spdx

    # 2. secrets (schema + code blobs)
    drop = False
    for key in ("schema",):
        text, findings, dropped = scan_and_clean(rec.get(key, ""))
        rec[key] = text
        if dropped:
            drop = True
    for f in rec.get("call_sites", []):
        text, findings, dropped = scan_and_clean(f.get("snippet", ""))
        f["snippet"] = text
        if dropped:
            drop = True
    if drop:
        drop_stats["secrets"] = drop_stats.get("secrets", 0) + 1
        return None

    # 3. PII
    rec["schema"], _edits = scrub_text(rec.get("schema", ""))

    # 4. version normalization
    original_version = detect_version(rec.get("schema", ""))
    rec["schema"], norm = normalize_to_v7(rec["schema"])
    if norm.changed:
        rec.setdefault("meta", {})["version_normalized_from"] = original_version

    # 5. quality
    chk = check_schema(rec.get("schema", ""))
    if not chk.ok:
        drop_stats["quality"] = drop_stats.get("quality", 0) + 1
        return None
    rec["quality_score"] = chk.score
    return rec


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 03: cleaning pipeline")
    ap.add_argument("--input", default="datasets/processed/extraction_records.jsonl")
    ap.add_argument("--out", default="datasets/processed")
    ap.add_argument("--min-quality", type=float, default=0.6)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    records = read_jsonl(args.input)
    log.info("loaded %d extraction records", len(records))

    drop_stats: dict[str, int] = {}
    cleaned: list[dict] = []
    dedup = Deduplicator()
    for rec in records:
        out = clean_record(rec, drop_stats=drop_stats)
        if out is None:
            continue
        out["dedup_text"] = out.get("schema", "") + json.dumps(
            [c.get("snippet", "") for c in out.get("call_sites", [])][:20]
        )
        if dedup.consider(out):
            out.pop("dedup_text", None)
            cleaned.append(out)
        else:
            drop_stats["dedup"] = drop_stats.get("dedup", 0) + 1

    out_path = Path(args.out)
    write_jsonl(out_path / "clean_records.jsonl", cleaned)
    write_json(
        out_path / "clean_report.json",
        {
            "input": len(records),
            "output": len(cleaned),
            "dropped": drop_stats,
            "dedup_stats": dedup.stats.as_dict(),
        },
    )
    log.info("clean corpus: %d kept, drops=%s", len(cleaned), drop_stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
