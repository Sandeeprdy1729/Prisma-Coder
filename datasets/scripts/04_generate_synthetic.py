#!/usr/bin/env python3
"""PrismaForge stage 04 - synthetic track generation (report 9.4).

    python datasets/scripts/04_generate_synthetic.py --limit 120000 --seed 20260830

Produces datasets/synthetic/synthetic_pool.jsonl (+ generation_report.json).
Scale targets follow Table 9.1: 120k SFT triples at full scale; ``--limit``
scales the same mixture for smoke runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.io_utils import ensure_dir, write_json, write_jsonl  # noqa: E402
from prismacoder.synthetic.generator import SyntheticGenerator  # noqa: E402
from prismacoder.synthetic.vocab import DOMAINS_SPECS  # noqa: E402

log = logging.getLogger("prismaforge.stage04")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 04: synthetic generation")
    ap.add_argument("--limit", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out", default="datasets/synthetic")
    ap.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="restrict to families: schema_tasks client_crud nested_agg_txn repo_repair docs_misc",
    )
    ap.add_argument(
        "--reasoning-fraction",
        type=float,
        default=0.10,
        help="fraction of records receiving a mechanical plan-span",
    )
    ap.add_argument(
        "--include-ood-domains",
        action="store_true",
        help="include the 2 withheld OOD domains (benchmark builds only)",
    )
    ap.add_argument(
        "--distractor-seed",
        type=int,
        default=7,
        help="seed for the distractor schema pool used by augmentation",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    gen = SyntheticGenerator(args.seed, include_ood_domains=args.include_ood_domains)

    # seed the distractor pool with tiny schemas from a fixed sub-seed
    pool_rng_gen = SyntheticGenerator(args.distractor_seed)
    pool = []
    for spec in DOMAINS_SPECS[:4]:
        pool_rng_gen._current_domain = spec.key
        s = pool_rng_gen.sampler.sample(spec, entity_count=3)
        pool.append(s.render())
    gen.seed_distractor_pool(pool)

    records = gen.generate(args.limit, families=args.families)
    gen.attach_reasoning(records, fraction=args.reasoning_fraction)

    out_dir = ensure_dir(args.out)
    n = write_jsonl(out_dir / "synthetic_pool.jsonl", records)
    write_json(
        out_dir / "generation_report.json",
        {
            "requested": args.limit,
            "emitted": n,
            "seed": args.seed,
            "reasoning_fraction": args.reasoning_fraction,
            "report": gen.report.as_dict(),
        },
    )
    log.info("synthetic pool: %d records -> %s", n, out_dir / "synthetic_pool.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
