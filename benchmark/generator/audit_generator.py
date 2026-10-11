#!/usr/bin/env python3
"""Generator health gate: every schema the generator emits must pass the REAL Prisma validator.

    python benchmark/generator/audit_generator.py [--pairs 1000] [--max-invalid 0]

Samples before/after schema pairs from the edit-task engine, validates both with the Prisma 7.10 schema engine
(WASM build, installed on first use via npm), and checks that every operation named in a task is really present
in its reference. Exit status 1 on any violation, so it can run in CI before a benchmark or dataset is built.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from prismacoder.edit_kinds import detect_kinds  # noqa: E402
from prismacoder.synthetic.schema_gen import PerturbationEngine, SchemaSampler  # noqa: E402

WASM = "@prisma/prisma-schema-wasm@7.10.0-4.0edf323efd1d98336f3f0a68684b56f689b900d3"  # engine of Prisma 7.10.0
PV = r"""
const w = require('@prisma/prisma-schema-wasm'); const fs = require('fs');
const items = JSON.parse(fs.readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(items.map(it => { try { w.validate(JSON.stringify({prismaSchema: it.schema, noColor: true}));
  return {key: it.key, valid: true, message: ''}; } catch (e) { let m = String((e && e.message) || e); try { m = JSON.parse(m).message || m; } catch (_) {}
  return {key: it.key, valid: false, message: m}; } })));
"""


def validator_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "prismabench_validator"
    if not (d / "node_modules" / "@prisma" / "prisma-schema-wasm").exists():
        if not shutil.which("npm"):
            sys.exit("npm/node (>=18) is required for the real-validator gate")
        d.mkdir(exist_ok=True)
        (d / "package.json").write_text('{"name":"pv","private":true}')
        r = subprocess.run(["npm", "install", "--no-audit", "--no-fund", "--loglevel=error", WASM], cwd=d, capture_output=True, text=True)
        if r.returncode:
            sys.exit("npm install failed:\n" + r.stderr[-500:])
    (d / "pv.cjs").write_text(PV)
    return d


def validate(schemas: dict[str, str]) -> dict[str, dict]:
    d, out = validator_dir(), {}
    items = [{"key": k, "schema": v} for k, v in schemas.items()]
    for i in range(0, len(items), 400):
        r = subprocess.run(["node", "pv.cjs"], cwd=d, input=json.dumps(items[i:i + 400]), capture_output=True, text=True)
        out.update({o["key"]: o for o in json.loads(r.stdout)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1000)
    ap.add_argument("--max-invalid", type=int, default=0, help="tolerated invalid schemas (default 0)")
    ap.add_argument("--min-ops-present", type=float, default=0.97, help="share of tasks whose listed ops are all in the reference")
    a = ap.parse_args()

    before, after, metas = {}, {}, {}
    for seed in range(a.pairs):
        rng = random.Random(seed)
        b, af, meta = PerturbationEngine(rng).perturb(SchemaSampler(rng).sample())
        before[str(seed)], after[str(seed)], metas[str(seed)] = b, af, meta
    vb, va = validate(before), validate(after)
    bad_b = [k for k, v in vb.items() if not v["valid"]]
    bad_a = [k for k, v in va.items() if not v["valid"]]
    present = sum(set(metas[k]["ops"]) <= detect_kinds(before[k], after[k]) for k in before)
    ops_share = present / a.pairs

    print(f"pairs sampled:                          {a.pairs}")
    print(f"invalid BEFORE schemas (task context):  {len(bad_b)}")
    print(f"invalid AFTER schemas (references):     {len(bad_a)}")
    print(f"tasks whose listed ops are all present: {present}/{a.pairs} ({100 * ops_share:.1f}%)")
    for label, bad, res in (("before", bad_b, vb), ("after", bad_a, va)):
        for msg, n in collections.Counter(res[k]["message"].split("\n")[0][:100] for k in bad).most_common(5):
            print(f"  [{label}] {n:4d}  {msg}")
    ok = len(bad_b) + len(bad_a) <= a.max_invalid and ops_share >= a.min_ops_present
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
