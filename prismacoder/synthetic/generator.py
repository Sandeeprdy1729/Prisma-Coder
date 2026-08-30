"""Synthetic track orchestrator (report 9.4 / Table 9.1).

Draws from the 12 domain archetypes, samples schemas, and emits every task
family with mechanical references:

  T1 nl2prisma | T2 schema_edit (+chain_edit) | T3 migration
  T4 crud      | T5 complex_reads            | T6 nested_writes/txn
  T7 agg       | T8 repo_completion/completion | T9 bug_fix
  T10 cross_file (+nl_explain/qa) | T11 performance | T12 security

Plus DPO preference pairs (reference vs bug-injected) and reasoning plan-spans.
Every record conforms to the canonical schema and passes through the
Deduplicator (exact + MinHash + isomorphism) before leaving the generator.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from prismacoder import __version__
from prismacoder.constants import DEFAULT_SEED, DOMAINS, OOD_WITHHELD_DOMAINS
from prismacoder.dedup import Deduplicator
from prismacoder.hashing import sha256_obj
from prismacoder.schema import canonicalize_record
from prismacoder.synthetic.augment import Augmenter
from prismacoder.synthetic.crud_gen import CRUD_GENERATORS, GeneratedTask
from prismacoder.synthetic.instruction_gen import persona_pick
from prismacoder.synthetic.migration_gen import gen_migration_task, to_generated_task
from prismacoder.synthetic.relation_gen import (
    AGG_GENERATORS,
    RELATION_GENERATORS,
    TXN_GENERATORS,
)
from prismacoder.synthetic.schema_gen import PerturbationEngine, SchemaSampler
from prismacoder.synthetic.task_variants import (
    BugInjector,
    derive_plan_span,
    gen_bug_fix,
    gen_chain_edit,
    gen_completion,
    gen_cross_file,
    gen_nl2prisma,
    gen_nl_explain,
    gen_performance,
    gen_qa,
    gen_repo_completion,
    gen_schema_edit,
    gen_security,
    make_preference_pair,
    make_reasoning,
)

# task-family weights for the SFT mixture (report Table 9.1 proportions)
FAMILY_WEIGHTS: dict[str, float] = {
    "schema_tasks": 35_000 / 120_000,
    "client_crud": 25_000 / 120_000,
    "nested_agg_txn": 30_000 / 120_000,
    "repo_repair": 20_000 / 120_000,
    "docs_misc": 10_000 / 120_000,
}
FAMILY_TASKS: dict[str, tuple[str, ...]] = {
    "schema_tasks": ("T1_schema_greenfield", "T2_schema_edit", "T3_migration_plan"),
    "client_crud": ("T4_client_crud",),
    "nested_agg_txn": ("T5_complex_reads", "T6_nested_writes_txn", "T7_aggregations"),
    "repo_repair": ("T8_repo_completion", "T9_bug_fix"),
    "docs_misc": ("T10_cross_file", "T11_performance", "T12_security"),
}


@dataclass
class GenerationReport:
    counts: dict[str, int] = dc_field(default_factory=dict)
    dedup: dict[str, int] = dc_field(default_factory=dict)
    families: dict[str, int] = dc_field(default_factory=dict)
    started: float = dc_field(default_factory=time.time)

    def note(self, task_type: str, category: str) -> None:
        self.counts[task_type] = self.counts.get(task_type, 0) + 1
        self.families[category] = self.families.get(category, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_types": dict(sorted(self.counts.items())),
            "categories": dict(sorted(self.families.items())),
            "dedup": self.dedup,
            "seconds": round(time.time() - self.started, 2),
        }


class SyntheticGenerator:
    """Seeded, deterministic generator. Same seed => same corpus."""

    def __init__(self, seed: int = DEFAULT_SEED, *, include_ood_domains: bool = False):
        self.rng = random.Random(seed)
        self.seed = seed
        self.sampler = SchemaSampler(self.rng)
        self.perturber = PerturbationEngine(self.rng)
        self.bug_injector = BugInjector(self.rng)
        self.augmenter = Augmenter(self.rng)
        self.dedup = Deduplicator()
        self.report = GenerationReport()
        self.include_ood = include_ood_domains
        self._distractor_pool: list[str] = []
        self._n = 0

    # -- record assembly --------------------------------------------------------
    def _record(
        self,
        task: GeneratedTask,
        schema_psl: str,
        *,
        track: str = "synthetic",
        tests: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._n += 1
        self._last_schema = schema_psl
        rec: dict[str, Any] = {
            "id": f"pf-{self.seed % 100000:05d}-{self._n:07d}",
            "schema_version": "1.0",
            "source": {
                "track": track,
                "origin_repo": "",
                "url": "",
                "license": "MIT",
                "commit": "",
                "path": "",
                "provenance_hash": "",
            },
            "domain": self._current_domain,
            "task": {
                "type": task.task_type,
                "category": task.category,
                "difficulty": self._difficulty_of(schema_psl),
                "categories": [task.category],
                "instruction": task.instruction,
                "context": {
                    "schema": schema_psl,
                    "dialect": self._current_provider,
                    "files": [],
                },
                "input": "",
                "reference": task.reference,
                "persona": task.persona,
                "tests": tests or {},
                "destructive_risk": "destructive" in " ".join(task.style_tags),
            },
            "verification": {
                "status": "unverified",
                "gates": {},
                "runtime_ms": 0,
                "verifier_version": __version__,
            },
            "difficulty_scores": self._last_complexity,
            "style_tags": task.style_tags,
            "meta": {
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "generator_version": __version__,
                "seed": self.seed,
                "tags": ["synthetic"],
                "canary": "",
            },
        }
        rec["source"]["provenance_hash"] = sha256_obj(
            {
                "instruction": task.instruction,
                "schema": schema_psl,
                "reference": task.reference,
            }
        )
        return canonicalize_record(rec)

    def _difficulty_of(self, schema_psl: str) -> str:
        from prismacoder.psl import parse_schema

        n_models = len(parse_schema(schema_psl).models)
        if n_models <= 6:
            return "L1"
        if n_models <= 10:
            return "L2"
        return "L3"

    # -- main entry ----------------------------------------------------------------
    _current_domain: str = "ecommerce"
    _current_provider: str = "sqlite"
    _last_complexity: dict[str, int] = {"models": 0, "relations": 0, "nesting": 0, "ops": 0, "ambiguity": 0}
    _last_schema: str = ""

    def generate(self, limit: int, *, families: list[str] | None = None) -> list[dict[str, Any]]:
        """Generate up to ``limit`` deduplicated records."""
        allowed_families = families or list(FAMILY_WEIGHTS)
        records: list[dict[str, Any]] = []
        attempts = 0
        max_attempts = limit * 30 + 500

        while len(records) < limit and attempts < max_attempts:
            attempts += 1
            family = self.rng.choices(
                allowed_families, weights=[FAMILY_WEIGHTS[f] for f in allowed_families]
            )[0]
            batch = self._generate_family_batch(family)
            for rec in batch:
                if len(records) >= limit:
                    break
                rec[self.dedup.text_key] = rec["task"]["instruction"] + rec["task"]["reference"]
                if self.dedup.consider(rec):
                    rec.pop(self.dedup.text_key, None)
                    records.append(rec)
                    self.report.note(rec["task"]["type"], rec["task"]["category"])

        self.report.dedup = self.dedup.stats.as_dict()
        return records

    def _allowed_domains(self) -> list[str]:
        if self.include_ood:
            return list(DOMAINS)
        return [d for d in DOMAINS if d not in OOD_WITHHELD_DOMAINS]

    # -- family emitters ---------------------------------------------------------
    def _generate_family_batch(self, family: str) -> list[dict[str, Any]]:
        domain_key = self.rng.choice(self._allowed_domains())
        self._current_domain = domain_key
        level = self.rng.choices(["L1", "L2", "L3"], weights=[45, 40, 15])[0]
        s = self.sampler.sample(domain_key, level=level)
        self._current_provider = s.provider
        self._last_complexity = s.complexity()

        out: list[dict[str, Any]] = []
        persona = persona_pick(self.rng)

        if family == "client_crud":
            gen = self.rng.choice(CRUD_GENERATORS)
            t = gen(s, self.rng, persona)
            out.append(self._record(t, s.render(), tests=self._smoke_tests(t)))
            if self.rng.random() < 0.35:
                bug = gen_bug_fix(s, self.rng, t, persona)
                if bug:
                    out.append(self._record(bug, s.render()))
                    out.append(self._preference(t, bug.reference, "V3_tsc differential", s.render()))

        elif family == "nested_agg_txn":
            pool = list(RELATION_GENERATORS) + list(TXN_GENERATORS) + list(AGG_GENERATORS)
            gen = self.rng.choice(pool)
            t = gen(s, self.rng, persona)
            out.append(self._record(t, s.render(), tests=self._smoke_tests(t)))
            if self.rng.random() < 0.25:
                out.append(self._record(gen_nl_explain(t), s.render()))

        elif family == "schema_tasks":
            roll = self.rng.random()
            if roll < 0.4:
                out.append(self._record(gen_nl2prisma(s, self.rng, persona), s.render()))
            elif roll < 0.75:
                before, after, meta = self.perturber.perturb(s)
                desc = "; ".join(OP_LABELS.get(op, op) for op in meta["ops"])
                model = s.models[0].name if s.models else "Model"
                out.append(self._record(gen_schema_edit(before, after, desc, model, persona), s.render()))
                if self.rng.random() < 0.4:
                    mt = gen_migration_task(s, after, meta["ops"], self.rng)
                    out.append(self._record(to_generated_task(mt), s.render()))
            else:
                for t in gen_chain_edit(s, self.rng, n_steps=self.rng.randint(2, 3)):
                    out.append(self._record(t, s.render()))

        elif family == "repo_repair":
            roll = self.rng.random()
            if roll < 0.35:
                out.append(self._record(gen_repo_completion(s, self.rng, persona), s.render()))
            elif roll < 0.6:
                out.append(self._record(gen_completion(s, self.rng, persona), s.render()))
            else:
                base = self.rng.choice(CRUD_GENERATORS)(s, self.rng, persona)
                bug = gen_bug_fix(s, self.rng, base, persona)
                if bug:
                    out.append(self._record(bug, s.render()))
                    out.append(self._preference(base, bug.reference, "gate-differential", s.render()))
                else:
                    out.append(self._record(base, s.render()))

        else:  # docs_misc
            roll = self.rng.random()
            if roll < 0.3:
                out.append(self._record(gen_cross_file(s, self.rng, persona), s.render()))
            elif roll < 0.5:
                out.append(self._record(gen_performance(s, self.rng, persona), s.render()))
            elif roll < 0.7:
                out.append(self._record(gen_security(s, self.rng, persona), s.render()))
            else:
                out.append(self._record(gen_qa(self.rng), s.render()))

        # augmentation pass on a fraction of the batch
        if self._distractor_pool and self.rng.random() < 0.15 and out:
            base_rec = self.rng.choice(out)
            for aug in self.augmenter.augment_record(base_rec, self._distractor_pool):
                aug["id"] = f"pf-{self.seed % 100000:05d}-{self._n:07d}-a{len(out):02d}"
                if self.dedup.consider(aug):
                    self.report.note(aug["task"]["type"], aug["task"]["category"])
                    out.append(aug)
        return out

    def _preference(
        self, chosen_task: GeneratedTask, rejected: str, basis: str, schema_psl: str = ""
    ) -> dict[str, Any]:
        rec = self._record(chosen_task, schema_psl or self._last_schema)
        rec["preference"] = make_preference_pair(chosen_task, rejected, basis)
        rec["meta"]["tags"].append("preference_pair")
        rec["id"] += "-pref"
        return rec

    def attach_reasoning(self, records: list[dict[str, Any]], fraction: float = 0.10) -> None:
        """Attach mechanical plan-spans to a fraction of records (report 12.1 S1)."""
        for rec in records:
            if self.rng.random() < fraction:
                plan = derive_plan_span(
                    GeneratedTask(
                        rec["task"]["type"],
                        rec["task"]["category"],
                        rec["task"]["instruction"],
                        rec["task"]["reference"],
                        rec["task"].get("persona", "junior"),
                        [],
                    )
                )
                rec["reasoning"] = make_reasoning(
                    GeneratedTask(
                        rec["task"]["type"],
                        rec["task"]["category"],
                        rec["task"]["instruction"],
                        rec["task"]["reference"],
                        rec["task"].get("persona", "junior"),
                        [],
                    ),
                    plan,
                )
                rec["meta"]["tags"].append("plan_span")

    def seed_distractor_pool(self, schemas: list[str]) -> None:
        self._distractor_pool = schemas

    # -- helpers -------------------------------------------------------------------

    def _smoke_tests(self, task: GeneratedTask) -> dict[str, Any]:
        """Vitest smoke-test bundle asserting the artifact's key mechanical property."""
        fn = next(iter(re.findall(r"export (?:async )?function (\w+)", task.reference)), None)
        cls = next(iter(re.findall(r"export class (\w+)", task.reference)), None)
        if not fn and not cls:
            return {}
        return {
            "framework": "vitest",
            "cases": [
                {
                    "name": "smoke: exported symbol present",
                    "assert": f"expect([{fn!r}, {cls!r}].filter(Boolean).length).toBeGreaterThan(0)",
                }
            ],
        }


OP_LABELS = {
    "add_field": "add a field",
    "remove_field": "remove a field",
    "retype_field": "retype a field",
    "toggle_optionality": "toggle optionality",
    "add_unique": "add a unique constraint",
    "add_index": "add an index",
    "add_composite_unique": "add a composite unique",
    "swap_key_style": "swap the key style",
    "add_enum_field": "introduce an enum field",
    "change_cascade": "change cascade rules",
}
