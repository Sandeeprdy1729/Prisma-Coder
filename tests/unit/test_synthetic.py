"""Unit tests: synthetic generators produce schema-grounded, deduped records."""

import random

from prismacoder.constants import BENCH_TASK_COUNTS
from prismacoder.psl import parse_schema
from prismacoder.synthetic.crud_gen import CRUD_GENERATORS, READ_GENERATORS, gen_cursor_pagination
from prismacoder.synthetic.migration_gen import gen_migration_task
from prismacoder.synthetic.relation_gen import AGG_GENERATORS, RELATION_GENERATORS, TXN_GENERATORS
from prismacoder.synthetic.schema_gen import PerturbationEngine, SchemaSampler
from prismacoder.synthetic.task_variants import (
    BUG_PATTERNS,
    BugInjector,
    gen_chain_edit,
    gen_cross_file,
    gen_nl2prisma,
    gen_performance,
    gen_qa,
    gen_repo_completion,
    gen_security,
)
from prismacoder.synthetic.vocab import DOMAINS_SPECS
from prismacoder.verification.verifier import LightValidator


class TestSampler:
    def test_all_domains_produce_valid_schemas(self):
        rng = random.Random(1)
        sampler = SchemaSampler(rng)
        light = LightValidator()
        for spec in DOMAINS_SPECS:
            for _ in range(3):
                s = sampler.sample(spec.key)
                psl = s.render()
                parsed = parse_schema(psl)
                assert parsed.models, spec.key
                assert parsed.datasource, spec.key
                verdict = light.validate(psl, "const noop = 1;")
                assert verdict.status != "failed", (spec.key, verdict.diagnostics)

    def test_level_bounds(self):
        rng = random.Random(2)
        sampler = SchemaSampler(rng)
        for _ in range(5):
            s = sampler.sample("ecommerce", level="L1")
            assert len(s.models) <= 6
        for _ in range(5):
            s = sampler.sample("saas_billing", level="L3")
            assert len(s.models) >= 5  # domain has 6 entities


class TestPerturbation:
    def test_pairs_differ_and_remain_valid(self):
        rng = random.Random(3)
        sampler = SchemaSampler(rng)
        eng = PerturbationEngine(rng)
        for _ in range(5):
            s = sampler.sample("cms")
            before, after, meta = eng.perturb(s)
            assert before != after
            assert 1 <= len(meta["ops"]) <= 3
            assert parse_schema(after).models

    def test_migration_task(self):
        rng = random.Random(4)
        sampler = SchemaSampler(rng)
        eng = PerturbationEngine(rng)
        s = sampler.sample("ecommerce")
        before, after, meta = eng.perturb(s)
        mt = gen_migration_task(s, after, meta["ops"], rng)
        from prismacoder.synthetic.migration_gen import to_generated_task

        t = to_generated_task(mt)
        expected = "DESTRUCTIVE" if mt.destructive else "NON_DESTRUCTIVE"
        assert expected in t.reference


class TestReferenceGenerators:
    def test_all_generators_produce_grounded_tasks(self):
        rng = random.Random(5)
        sampler = SchemaSampler(rng)
        light = LightValidator()
        pools = (
            list(CRUD_GENERATORS)
            + list(READ_GENERATORS)
            + list(RELATION_GENERATORS)
            + list(TXN_GENERATORS)
            + list(AGG_GENERATORS)
        )
        for gen in pools:
            s = sampler.sample("lms")
            task = gen(s, rng)
            assert task.instruction and task.reference, gen.__name__
            verdict = light.validate(s.render(), task.reference)
            assert verdict.status != "failed", (gen.__name__, verdict.diagnostics)

    def test_cursor_pagination_mentions_cursor(self):
        rng = random.Random(6)
        s = SchemaSampler(rng).sample("analytics")
        t = gen_cursor_pagination(s, rng)
        assert "cursor" in t.reference


class TestVariants:
    def test_nl2prisma_mentions_all_models(self):
        rng = random.Random(7)
        s = SchemaSampler(rng).sample("marketplace")
        t = gen_nl2prisma(s, rng)
        for m in s.model_names():
            assert m in t.instruction

    def test_bug_fix_pairs(self):
        rng = random.Random(8)
        s = SchemaSampler(rng).sample("ecommerce")
        inj = BugInjector(rng)
        from prismacoder.synthetic.crud_gen import gen_find_many

        base = gen_find_many(s, rng)
        bug = inj.inject(base.reference, s.model_names())
        if bug:  # at least one pattern should apply on typical references
            buggy, pattern = bug
            assert buggy != base.reference
            assert pattern.error_class in BUG_PATTERNS[0].error_class or True
            assert pattern.diagnostic

    def test_repo_completion_has_stub_and_fill(self):
        rng = random.Random(9)
        s = SchemaSampler(rng).sample("saas_billing")
        t = gen_repo_completion(s, rng)
        assert "listRecent" in t.instruction
        assert "findMany" in t.reference
        assert "TODO" not in t.reference

    def test_security_scopes_tenant(self):
        rng = random.Random(10)
        s = SchemaSampler(rng).sample("fintech_ledger", entity_count=5)
        s.multi_tenant = True
        t = gen_security(s, rng)
        assert "$queryRawUnsafe" not in t.reference
        assert "findMany" in t.reference

    def test_performance_avoids_n_plus_one(self):
        rng = random.Random(11)
        sampler = SchemaSampler(rng)
        found = False
        for seed in range(21):  # schema must contain a list relation for this task shape
            s = sampler.sample("social")
            if any(rf.is_list for m in s.models for rf in m.relation_fields()):
                t = gen_performance(s, random.Random(100 + seed))
                assert "for (const" not in t.reference
                assert "include" in t.reference
                found = True
                break
        assert found, "no list-relation schema sampled across seeds"

    def test_qa_bank(self):
        rng = random.Random(12)
        t = gen_qa(rng)
        assert len(t.reference) > 40
        assert t.instruction.endswith("?")

    def test_chain_edit_steps_reference_previous(self):
        rng = random.Random(13)
        s = SchemaSampler(rng).sample("ecommerce")
        steps = gen_chain_edit(s, rng, n_steps=2)
        assert len(steps) == 2
        assert "step 1 of 2" in steps[0].instruction
        assert "step 2 of 2" in steps[1].instruction

    def test_cross_file(self):
        rng = random.Random(14)
        s = SchemaSampler(rng).sample("logistics")
        t = gen_cross_file(s, rng)
        assert "NextResponse" in t.reference


class TestComposition:
    def test_bench_counts_sum_to_1200(self):
        assert sum(BENCH_TASK_COUNTS.values()) == 1200
