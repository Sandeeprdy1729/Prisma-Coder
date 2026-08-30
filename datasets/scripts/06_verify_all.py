#!/usr/bin/env python3
"""PrismaForge stage 06 - verification sweep (the four-gate chain).

Runs V1-V5 over every record in the pool when Node >= 18.18 is available;
falls back to the static LightValidator otherwise (records keep
``verification.status='unverified'`` so full-gate re-verification is visible).

Verdicts:
  verified   all requested gates green        -> training split eligible
  failed     first failing gate + E1-E10      -> DPO negative pool
  unverified static-only pass (no node)       -> requires full re-verify before release

Usage:
    python datasets/scripts/06_verify_all.py --input datasets/synthetic/synthetic_pool.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.io_utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from prismacoder.verification.sandbox import node_available  # noqa: E402
from prismacoder.verification.verifier import LightValidator, Verifier  # noqa: E402

log = logging.getLogger("prismaforge.stage06")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 06: verification sweep")
    ap.add_argument(
        "--input",
        action="append",
        default=[],
        help="pool jsonl; repeatable (default: synthetic_pool + augmented + mined)",
    )
    ap.add_argument("--out", default="datasets/verified")
    ap.add_argument(
        "--full-gates", action="store_true", help="force full V1-V5 (requires node; default = auto)"
    )
    ap.add_argument("--light-only", action="store_true", help="force static validation")
    ap.add_argument(
        "--gates", nargs="*", default=["V1_validate", "V2_generate", "V3_tsc", "V4_execute", "V5_vitest"]
    )
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="verify at most N records per pool (0 = all)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    inputs = args.input or [
        "datasets/synthetic/synthetic_pool.jsonl",
        "datasets/synthetic/augmented_pool.jsonl",
        "datasets/synthetic/mined_pairs.jsonl",
    ]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    node, _ = node_available()
    use_full = (args.full_gates or bool(node)) and not args.light_only
    if use_full:
        verifier = Verifier(timeout_seconds=args.timeout, gates=tuple(args.gates))
        log.info("running FULL gates (V1-V5 subset %s)", args.gates)
    else:
        log.info("Node unavailable or --light-only: using LightValidator (static)")

    summary = {"pools": {}, "mode": "full" if use_full else "static", "verdicts": {}}
    for pool in inputs:
        pool_path = Path(pool)
        if not pool_path.exists():
            log.warning("pool missing, skipped: %s", pool)
            continue
        records = read_jsonl(pool_path)
        if args.limit:
            records = records[: args.limit]
        verified, failed, unverified = 0, 0, 0
        out_records = []
        for rec in records:
            schema = rec.get("task", {}).get("context", {}).get("schema", "")
            artifact = rec.get("task", {}).get("reference", "")
            tests = rec.get("task", {}).get("tests") or {}
            if use_full:
                try:
                    res = verifier.verify(
                        schema,
                        {"artifact.ts": _wrap_artifact(artifact)},
                        _test_files(tests),
                        provider=rec.get("task", {}).get("context", {}).get("dialect", "sqlite"),
                        artifact_kind=_artifact_kind(rec),
                    )
                except Exception:  # noqa: BLE001 - sandbox failure must not stop the sweep
                    log.exception("sandbox crashed for %s", rec.get("id"))
                    res = None
                if res is None:
                    unverified += 1
                    rec["verification"]["status"] = "unverified"
                    rec["verification"]["gates"] = {"sandbox": {"passed": False, "error": "sandbox crash"}}
                else:
                    rec["verification"] = {**rec.get("verification", {}), **res.to_record_block()}
                    if res.status == "verified":
                        verified += 1
                    else:
                        failed += 1
            else:
                res = LightValidator().validate(schema, artifact)
                rec["verification"] = {**rec.get("verification", {}), **res.to_record_block()}
                rec["verification"]["gates"]["V0_static_mode"] = "light"
                if res.status == "failed":
                    failed += 1
                else:
                    unverified += 1
            out_records.append(rec)

        name = pool_path.stem
        n = write_jsonl(out_dir / f"{name}.verified.jsonl", out_records)
        summary["pools"][name] = {
            "records": n,
            "verified": verified,
            "failed": failed,
            "unverified": unverified,
        }
        log.info(
            "%s: %d records (verified=%d failed=%d unverified=%d)", name, n, verified, failed, unverified
        )

    write_json(out_dir / "verification_summary.json", summary)
    log.info("verification sweep complete: %s", json.dumps(summary["pools"]))
    return 0


PSL_TASKS = {"nl2prisma", "schema_edit", "migration"}


def _artifact_kind(rec: dict) -> str:
    return "PSL" if rec.get("task", {}).get("type") in PSL_TASKS else "TS"


def _wrap_artifact(artifact: str) -> str:
    """Strip markdown fences if the artifact carries them."""
    if "```" in artifact:
        import re

        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", artifact, re.S)
        if blocks:
            return "\n\n".join(blocks)
    return artifact


def _test_files(tests: dict) -> dict[str, str] | None:
    if not tests or not tests.get("cases"):
        return None
    cases = tests["cases"]
    lines = ['import { describe, it, expect } from "vitest";', ""]
    for c in cases:
        lines.append(f'it("{c["name"]}", () => {{ {c["assert"]} }});')
    return {"artifact.test.ts": "\n".join(lines) + "\n"}


if __name__ == "__main__":
    raise SystemExit(main())
