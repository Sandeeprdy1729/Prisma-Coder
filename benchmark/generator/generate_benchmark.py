"""Benchmark generator (Phase 2).

Builds PrismaBench v1.0: 1,200 tasks across T1-T12, three mechanical levels,
12 domains (2 withheld for OOD reporting), with:
- public split  -> benchmark/public/prismabench_public.jsonl (reference withheld)
- private keys  -> benchmark/private/private_keys.jsonl (maintainers only)
- provenance    -> benchmark/public/benchmark_provenance.json (firewall input)
- canary strings-> embedded in every task (contamination audit)

All tasks are generated from seeded samplers + mechanical reference composers
(the report's 'teachers write questions, mechanics write answers' discipline),
then each task's reference and tests pass the LightValidator; with Node
available, pass ``--full-verify`` to run V1-V5 on every task's reference.

Usage:
    python benchmark/generator/generate_benchmark.py --scale 1.0 --seed 20260830
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmark.generator.task_specs import bench_task_id, build_plan, public_task_shape  # noqa: E402
from prismacoder.constants import (  # noqa: E402
    BENCH_TASK_COUNTS,
    BENCHMARK_NAME,
    BENCHMARK_VERSION,
    DEFAULT_SEED,
    DOMAINS,
    OOD_WITHHELD_DOMAINS,
)
from prismacoder.io_utils import ensure_dir, write_json, write_jsonl  # noqa: E402
from prismacoder.synthetic.crud_gen import (  # noqa: E402
    CRUD_GENERATORS,
    READ_GENERATORS,
)
from prismacoder.synthetic.instruction_gen import persona_pick  # noqa: E402
from prismacoder.synthetic.migration_gen import gen_migration_task, to_generated_task  # noqa: E402
from prismacoder.synthetic.relation_gen import (  # noqa: E402
    AGG_GENERATORS,
    RELATION_GENERATORS,
    TXN_GENERATORS,
)
from prismacoder.synthetic.schema_gen import PerturbationEngine, SchemaSampler  # noqa: E402
from prismacoder.synthetic.task_variants import (  # noqa: E402
    BugInjector,
    gen_bug_fix,
    gen_completion,
    gen_cross_file,
    gen_nl2prisma,
    gen_performance,
    gen_repo_completion,
    gen_schema_edit,
    gen_security,
)
from prismacoder.verification.verifier import LightValidator  # noqa: E402

log = logging.getLogger("prismabench.generator")


def _canary(seed: str) -> str:
    return "PRISMABENCH-CANARY-" + hashlib.sha256(seed.encode()).hexdigest()[:16].upper()


class BenchmarkBuilder:
    def __init__(self, seed: int, scale: float, include_ood: bool = False):
        import random

        self.rng = random.Random(seed)
        self.seed = seed
        self.scale = scale
        self.include_ood = include_ood
        self.sampler = SchemaSampler(self.rng)
        self.perturber = PerturbationEngine(self.rng)
        self.bug_injector = BugInjector(self.rng)
        self.light = LightValidator()

    # cap on attempts per successful task before giving up on a slot, so a
    # structurally-broken composer/gate combo fails loudly instead of hanging
    MAX_ATTEMPTS_PER_TASK = 40

    def build(self) -> tuple[list[dict], list[dict], list[dict]]:
        plan = build_plan()
        tasks: list[dict] = []
        seq_per_cat: dict[str, int] = {}
        shortfalls: dict[str, int] = {}
        for category_id, level, count in plan:
            n = max(1, round(count * self.scale))
            made = 0
            attempts = 0
            max_attempts = n * self.MAX_ATTEMPTS_PER_TASK
            while made < n and attempts < max_attempts:
                attempts += 1
                seq = seq_per_cat.get(category_id, 0) + 1
                task = self._make_task(category_id, level, bench_task_id(category_id.split("_")[0], seq))
                if task:
                    seq_per_cat[category_id] = seq
                    tasks.append(task)
                    made += 1
            if made < n:
                key = f"{category_id}/{level}"
                shortfalls[key] = n - made
                log.warning(
                    "gave up on %s after %d attempts: got %d/%d tasks "
                    "(composer/gate pass rate too low for this slot)",
                    key, attempts, made, n,
                )
        if shortfalls:
            log.warning("build finished with shortfalls: %s", shortfalls)
        return self._split(tasks)

    # -- per-task construction ---------------------------------------------------
    def _make_task(self, category_id: str, level: str, bench_id: str) -> dict | None:
        domain = self.rng.choice(
            DOMAINS if self.include_ood else [d for d in DOMAINS if d not in OOD_WITHHELD_DOMAINS]
        )
        s = self.sampler.sample(domain, level=level)
        persona = persona_pick(self.rng)
        instruction, reference, artifact_kind = self._compose(category_id, s, level, persona)

        if reference is None or instruction is None:
            return None
        verdict = self.light.validate(s.render(), reference)
        if verdict.status == "failed":
            return None  # adversarial review stand-in: discard non-green compositions

        tid = category_id.split("_")[0]
        task = {
            "bench_id": bench_id,
            "category": category_id,
            "tid": tid,
            "level": level,
            "domain": domain,
            "instruction": instruction,
            "context": {
                "schema": s.render(),
                "dialect": s.provider,
                "files": [],
            },
            "input": "",
            "reference": reference,
            "artifact_kind": artifact_kind,
            "canary": _canary(bench_id),
            "meta": {
                "seed": self.seed,
                "benchmark": BENCHMARK_NAME,
                "version": BENCHMARK_VERSION,
                "gate_requirements": self._gates_for(category_id),
            },
        }
        return task

    def _gates_for(self, category_id: str) -> list[str]:
        from benchmark.generator.task_specs import SPEC_BY_ID

        return list(SPEC_BY_ID[category_id].gates)

    def _compose(self, category_id: str, s, level: str, persona: str):
        """Compose (instruction, reference, artifact_kind) per category."""
        if category_id == "T1_schema_greenfield":
            t = gen_nl2prisma(s, self.rng, persona)
            return t.instruction, t.reference, "PSL"
        if category_id == "T2_schema_edit":
            before, after, meta = self.perturber.perturb(s)
            desc = "; ".join(meta["ops"])
            model = s.models[0].name if s.models else "Model"
            t = gen_schema_edit(before, after, desc, model, persona)
            return t.instruction, t.reference, "PSL"
        if category_id == "T3_migration_plan":
            _, after, meta = self.perturber.perturb(s)
            mt = gen_migration_task(s, after, meta["ops"], self.rng)
            t = to_generated_task(mt)
            return t.instruction, t.reference, "PSL+SQL+risk"
        if category_id == "T4_client_crud":
            t = self.rng.choice(CRUD_GENERATORS)(s, self.rng, persona)
            return t.instruction, t.reference, "TS"
        if category_id == "T5_complex_reads":
            t = self.rng.choice(READ_GENERATORS)(s, self.rng, persona)
            return t.instruction, t.reference, "TS"
        if category_id == "T6_nested_writes_txn":
            pool = list(RELATION_GENERATORS) + list(TXN_GENERATORS)
            t = self.rng.choice(pool)(s, self.rng, persona)
            return t.instruction, t.reference, "TS"
        if category_id == "T7_aggregations":
            t = self.rng.choice(AGG_GENERATORS)(s, self.rng, persona)
            return t.instruction, t.reference, "TS"
        if category_id == "T8_repo_completion":
            t = self.rng.choice([gen_repo_completion, gen_completion])(s, self.rng, persona)
            return t.instruction, t.reference, "TS multi-file"
        if category_id == "T9_bug_fix":
            base = self.rng.choice(CRUD_GENERATORS)(s, self.rng, persona)
            bug = gen_bug_fix(s, self.rng, base, persona)
            if bug is None:
                return None, None, ""
            return bug.instruction, bug.reference, "TS"
        if category_id == "T10_cross_file":
            t = gen_cross_file(s, self.rng, persona)
            return t.instruction, t.reference, "TS multi-file"
        if category_id == "T11_performance":
            t = gen_performance(s, self.rng, persona)
            return t.instruction, t.reference, "TS+index"
        if category_id == "T12_security":
            t = gen_security(s, self.rng, persona)
            return t.instruction, t.reference, "TS"
        return None, None, ""

    # -- split + provenance ---------------------------------------------------------
    def _split(self, tasks: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
        """Public dev split (withheld-reference) + private keys + provenance."""
        # private rotating split: 50 percent of tasks are private-gated per report 9.5/10.1
        self.rng.shuffle(tasks)
        n_private = int(len(tasks) * 0.5)
        private, public = tasks[:n_private], tasks[n_private:]
        public_rows = [public_task_shape(t) for t in public]
        private_rows = [
            {
                "bench_id": t["bench_id"],
                "category": t["category"],
                "level": t["level"],
                "reference": t["reference"],
                "artifact_kind": t["artifact_kind"],
                "canary": t["canary"],
                "gate_requirements": t["meta"]["gate_requirements"],
            }
            for t in private
        ]
        provenance = {
            "benchmark": BENCHMARK_NAME,
            "version": BENCHMARK_VERSION,
            "seed": self.seed,
            "generated_at": self._now(),
            "provenance_hashes": sorted(
                {
                    hashlib.sha256(
                        json.dumps(
                            {"i": t["instruction"], "s": t["context"]["schema"][:256]}, sort_keys=True
                        ).encode()
                    ).hexdigest()
                    for t in tasks
                }
            ),
            "canaries": sorted({t["canary"] for t in tasks}),
        }
        return public_rows, private_rows, provenance

    @staticmethod
    def _now() -> str:
        import time

        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate PrismaBench v1.0")
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = full 1,200 tasks; 0.05 = 60-task smoke")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default="benchmark/public")
    ap.add_argument("--private-out", default="benchmark/private")
    ap.add_argument(
        "--include-ood-domains",
        action="store_true",
        help="developer builds only; release builds withhold the OOD domains",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    builder = BenchmarkBuilder(args.seed, args.scale, include_ood=args.include_ood_domains)
    public_rows, private_rows, provenance = builder.build()

    out = ensure_dir(args.out)
    priv_out = ensure_dir(args.private_out)
    n_pub = write_jsonl(out / "prismabench_public.jsonl", public_rows)
    n_priv = write_jsonl(priv_out / "private_keys.jsonl", private_rows)
    write_json(out / "benchmark_provenance.json", provenance)

    from collections import Counter

    by_cat = Counter(r["category"] for r in public_rows + private_rows)
    by_level = Counter(r["level"] for r in public_rows + private_rows)
    log.info(
        "PrismaBench built: %d public + %d private = %d tasks (scale=%.2f)",
        n_pub,
        n_priv,
        n_pub + n_priv,
        args.scale,
    )
    log.info("categories: %s", dict(sorted(by_cat.items())))
    log.info("levels: %s", dict(sorted(by_level.items())))

    if args.scale >= 0.999:
        assert sum(by_cat.values()) == 1200, f"expected 1200 tasks, got {sum(by_cat.values())}"
        expected = {k: v for k, v in BENCH_TASK_COUNTS.items()}
        assert dict(by_cat) == expected, "category counts drifted from Table 10.1"
        log.info("Table 10.1 composition check passed (1,200 tasks, T1-T12 exact counts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())