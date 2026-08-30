#!/usr/bin/env python3
"""PrismaForge stage 05 - augmentation + mined-track record assembly.

1. Augments the synthetic pool (persona paraphrase, entity rename, schema
   shuffle, distractor insertion) - re-deduplicated, tagged 'augmented'.
2. Converts cleaned *mined* records into SFT-shaped samples:
   (schema context + call-site census) -> CRUD/nl2prisma-style training pairs
   with the mined code as reference (mechanical conversion, no LLM).

Output: datasets/synthetic/augmented_pool.jsonl, datasets/synthetic/mined_pairs.jsonl
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.dedup import Deduplicator  # noqa: E402
from prismacoder.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from prismacoder.synthetic.augment import Augmenter  # noqa: E402
from prismacoder.synthetic.vocab import DOMAINS_SPECS  # noqa: E402

log = logging.getLogger("prismaforge.stage05")


def mined_to_pairs(rec: dict, rng: random.Random) -> list[dict]:
    """Mechanically convert a cleaned mined record into SFT-shaped pairs."""
    pairs: list[dict] = []
    schema = rec.get("schema", "")
    snippets = [c for c in rec.get("call_sites", []) if c.get("snippet")]
    if not schema or not snippets:
        return pairs
    by_family: dict[str, list[dict]] = {}
    for c in snippets:
        by_family.setdefault(c["family"], []).append(c)

    domain = rng.choice([d.key for d in DOMAINS_SPECS])
    for family, sites in sorted(by_family.items()):
        site = rng.choice(sites)
        task_type = {
            "crud": "crud",
            "aggregation": "aggregation",
            "group_by": "aggregation",
            "raw_sql": "security",
            "transaction": "nested_write",
        }.get(family)
        if task_type is None:
            continue
        pairs.append(
            {
                "id": f"pf-mined-{rec.get('schema_hash', 'x')[:16]}-{family}",
                "schema_version": "1.0",
                "source": {
                    "track": "mined",
                    "origin_repo": rec.get("repo", ""),
                    "url": "",
                    "license": rec.get("license", "MIT"),
                    "commit": rec.get("snapshot_ref", ""),
                    "path": rec.get("schema_path", ""),
                    "provenance_hash": rec.get("schema_hash", ""),
                },
                "domain": domain,
                "task": {
                    "type": task_type,
                    "category": (
                        "T4_client_crud"
                        if task_type == "crud"
                        else (
                            "T7_aggregations"
                            if task_type == "aggregation"
                            else ("T6_nested_writes_txn" if task_type == "nested_write" else "T12_security")
                        )
                    ),
                    "difficulty": (
                        "L2"
                        if 6 < len(rec.get("models", [])) <= 10
                        else ("L3" if len(rec.get("models", [])) > 10 else "L1")
                    ),
                    "categories": ["T4_client_crud"],
                    "instruction": (
                        f"Using the provided schema, write the {site['op']} call used by "
                        f"{rec.get('repo', 'the original project')} on the `{site['model_hint']}` model. "
                        f"Follow the schema's naming exactly."
                    ),
                    "context": {"schema": schema, "dialect": "sqlite", "files": []},
                    "input": "",
                    "reference": site["snippet"],
                    "persona": "expert",
                    "tests": {},
                    "destructive_risk": False,
                },
                "verification": {
                    "status": "unverified",
                    "gates": {},
                    "runtime_ms": 0,
                    "verifier_version": "1.0.0",
                },
                "difficulty_scores": rec.get(
                    "difficulty_scores",
                    {
                        "models": len(rec.get("models", [])),
                        "relations": 0,
                        "nesting": 1,
                        "ops": 1,
                        "ambiguity": 0,
                    },
                ),
                "style_tags": ["mined_grounding", family],
                "meta": {
                    "created": "",
                    "generator_version": "1.0.0",
                    "seed": 0,
                    "tags": ["mined"],
                    "canary": "",
                },
            }
        )
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 05: augmentation + mined pairs")
    ap.add_argument("--synthetic", default="datasets/synthetic/synthetic_pool.jsonl")
    ap.add_argument("--clean", default="datasets/processed/clean_records.jsonl")
    ap.add_argument("--out", default="datasets/synthetic")
    ap.add_argument(
        "--augment-rate", type=float, default=0.20, help="fraction of synthetic records to augment"
    )
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")
    rng = random.Random(args.seed)

    out_dir = Path(args.out)
    # --- augmentation -------------------------------------------------------
    synthetic = read_jsonl(args.synthetic) if Path(args.synthetic).exists() else []
    log.info("synthetic pool records: %d", len(synthetic))
    augmenter = Augmenter(rng)
    pool_rng = random.Random(args.seed + 1)
    distractor_pool = []
    for spec in DOMAINS_SPECS[:4]:
        from prismacoder.synthetic.schema_gen import SchemaSampler

        s = SchemaSampler(pool_rng).sample(spec, entity_count=3)
        distractor_pool.append(s.render())

    dedup_aug = Deduplicator()
    augmented: list[dict] = []
    for rec in synthetic:
        if rng.random() > args.augment_rate:
            continue
        for aug in augmenter.augment_record(rec, distractor_pool):
            aug["id"] = rec["id"] + ("-aug" + str(len(augmented) + 1))
            if dedup_aug.consider(aug):
                augmented.append(aug)
    n_aug = write_jsonl(out_dir / "augmented_pool.jsonl", augmented)
    log.info("augmented records: %d", n_aug)

    # --- mined pairs ----------------------------------------------------------
    mined_pairs: list[dict] = []
    if Path(args.clean).exists():
        clean = read_jsonl(args.clean)
        for rec in clean:
            mined_pairs.extend(mined_to_pairs(rec, rng))
    dedup_mined = Deduplicator()
    mined_pairs = dedup_mined.dedup(mined_pairs)
    n_mined = write_jsonl(out_dir / "mined_pairs.jsonl", mined_pairs)
    write_json(
        out_dir / "augment_report.json",
        {
            "synthetic_input": len(synthetic),
            "augmented": n_aug,
            "mined_pairs": n_mined,
            "seed": args.seed,
            "dedup_augmented": dedup_aug.stats.as_dict(),
        },
    )
    log.info("mined pairs: %d", n_mined)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
