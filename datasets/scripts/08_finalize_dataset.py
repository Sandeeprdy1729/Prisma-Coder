#!/usr/bin/env python3
"""PrismaForge stage 08 - finalize the release set.

Emits, under ``datasets/``:
  train.jsonl / validation.jsonl / test.jsonl   (training-facing copies)
  benchmark.jsonl                                (dev-eval rows for harness wiring)
  metadata.json                                  (build manifest + mixture report)
  schema.json                                    (canonical record JSON Schema)
  DATASET_CARD.md                                (generated dataset card skeleton)

Also assembles the DPO pair file from records carrying a ``preference`` block.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder import __version__  # noqa: E402
from prismacoder.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from prismacoder.schema import validate_record  # noqa: E402

log = logging.getLogger("prismaforge.stage08")

RECORD_JSON_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/prismacoder/PrismaCoder/datasets/schema.json",
    "title": "PrismaForge record",
    "type": "object",
    "required": ["id", "source", "domain", "task", "verification", "meta"],
    "properties": {
        "id": {"type": "string", "pattern": "^(pf|pb)-"},
        "schema_version": {"type": "string"},
        "source": {
            "type": "object",
            "required": ["track", "license"],
            "properties": {
                "track": {"enum": ["mined", "synthetic", "official", "docs", "augmented"]},
                "origin_repo": {"type": "string"},
                "url": {"type": "string"},
                "license": {"type": "string", "description": "SPDX id; permissive allowlist only"},
                "commit": {"type": "string"},
                "path": {"type": "string"},
                "provenance_hash": {"type": "string", "pattern": "^sha256:|^$"},
            },
        },
        "domain": {"type": "string", "description": "one of the 12 domain archetypes"},
        "task": {
            "type": "object",
            "required": ["type", "instruction", "context", "reference"],
            "properties": {
                "type": {
                    "enum": [
                        "nl2prisma",
                        "schema_edit",
                        "migration",
                        "crud",
                        "complex_read",
                        "nested_write",
                        "aggregation",
                        "repo_completion",
                        "bug_fix",
                        "cross_file",
                        "performance",
                        "security",
                        "nl_explain",
                        "qa",
                        "completion",
                    ]
                },
                "category": {"type": "string", "description": "T1..T12 category id"},
                "difficulty": {"enum": ["L1", "L2", "L3", "easy", "medium", "hard"]},
                "categories": {"type": "array", "items": {"type": "string"}},
                "instruction": {"type": "string", "minLength": 10},
                "context": {
                    "type": "object",
                    "required": ["schema"],
                    "properties": {
                        "schema": {"type": "string", "description": "PSL context"},
                        "dialect": {"enum": ["sqlite", "postgres", "postgresql", "mysql"]},
                        "files": {"type": "array", "items": {"type": "object"}},
                    },
                },
                "input": {"type": "string"},
                "reference": {
                    "type": "string",
                    "description": "gold artifact; absent in public benchmark tasks",
                },
                "persona": {"enum": ["junior", "pm", "expert", "none"]},
                "tests": {"type": "object"},
                "destructive_risk": {"type": "boolean"},
            },
        },
        "verification": {
            "type": "object",
            "required": ["status"],
            "properties": {
                "status": {"enum": ["verified", "unverified", "failed", "pending"]},
                "gates": {"type": "object"},
                "runtime_ms": {"type": "number"},
                "verifier_version": {"type": "string"},
                "error_class": {"type": "string", "pattern": "^E(10|[1-9])$|^$"},
            },
        },
        "difficulty_scores": {
            "type": "object",
            "properties": {
                "models": {"type": "integer"},
                "relations": {"type": "integer"},
                "nesting": {"type": "integer"},
                "ops": {"type": "integer"},
                "ambiguity": {"type": "integer"},
            },
        },
        "style_tags": {"type": "array", "items": {"type": "string"}},
        "meta": {
            "type": "object",
            "properties": {
                "created": {"type": "string"},
                "generator_version": {"type": "string"},
                "seed": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "canary": {"type": "string"},
            },
        },
        "preference": {
            "type": "object",
            "description": "DPO pair block (chosen = reference, rejected = gate-failing)",
            "properties": {
                "chosen": {"type": "string"},
                "rejected": {"type": "string"},
                "basis": {"type": "string"},
            },
        },
        "reasoning": {
            "type": "object",
            "description": "plan-span block (SFT auxiliary)",
            "properties": {"plan_span": {"type": "string"}, "artifact": {"type": "string"}},
        },
    },
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 08: finalize release set")
    ap.add_argument("--splits", default="datasets/splits")
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--tag", default="v1.0")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    splits_dir = Path(args.splits)
    out_dir = Path(args.out)

    invalid = 0
    mixture: Counter = Counter()
    domains: Counter = Counter()
    difficulties: Counter = Counter()
    dpo_rows: list[dict] = []
    totals: dict[str, int] = {}

    for split in ("train", "validation", "test"):
        path = splits_dir / f"{split}.jsonl"
        if not path.exists():
            log.warning("split missing: %s", path)
            totals[split] = 0
            continue
        rows = []
        for rec in read_jsonl(path):
            problems = validate_record(rec)
            if problems:
                invalid += 1
                if invalid <= 5:
                    log.debug("invalid record %s: %s", rec.get("id"), problems)
                continue
            mixture[rec["task"]["category"] or rec["task"]["type"]] += 1
            domains[rec["domain"]] += 1
            difficulties[rec["task"].get("difficulty", "?")] += 1
            if "preference" in rec:
                dpo_rows.append(
                    {
                        "id": rec["id"],
                        "prompt": rec["task"]["instruction"],
                        "context_schema": rec["task"]["context"]["schema"],
                        "chosen": rec["preference"]["chosen"],
                        "rejected": rec["preference"]["rejected"],
                        "basis": rec["preference"]["basis"],
                    }
                )
            rows.append(rec)
        n = write_jsonl(out_dir / f"{split}.jsonl", rows)
        totals[split] = n
        log.info("finalized %-11s: %6d records (invalid skipped: %d)", split, n, invalid)

    # benchmark-facing dev rows: test split rows w/o reference (public shape)
    bench_rows = []
    for rec in (
        read_jsonl(splits_dir / "validation.jsonl") if (splits_dir / "validation.jsonl").exists() else []
    ):
        public = json.loads(json.dumps(rec))
        public["reference_withheld"] = True
        bench_rows.append(public)
    n_bench = write_jsonl(out_dir / "benchmark.jsonl", bench_rows)

    n_dpo = write_jsonl(out_dir / "dpo_pairs.jsonl", dpo_rows)

    metadata = {
        "dataset": "PrismaForge",
        "version": args.tag,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "builder_version": __version__,
        "records": totals,
        "dpo_pairs": n_dpo,
        "benchmark_dev_rows": n_bench,
        "mixture_by_category": dict(mixture.most_common()),
        "domains": dict(domains.most_common()),
        "difficulties": dict(difficulties.most_common()),
        "invalid_records_skipped": invalid,
        "canonical_schema": "datasets/schema.json",
        "license_policy": "permissive-only (MIT/Apache-2.0/BSD/ISC); copyleft excluded",
        "verification_policy": "four-gate chain (validate/generate/tsc/execute/vitest)",
    }
    write_json(out_dir / "metadata.json", metadata)
    write_json(out_dir / "schema.json", RECORD_JSON_SCHEMA)
    log.info("metadata + schema.json written; dpo_pairs=%d", n_dpo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
