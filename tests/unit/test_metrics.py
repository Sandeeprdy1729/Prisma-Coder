"""Unit tests: metrics (exact match, pass@k, BLEU, CodeBLEU, TED, registry)."""

from prismacoder.eval_metrics.bleu import corpus_bleu, sentence_bleu
from prismacoder.eval_metrics.code_bleu import code_bleu
from prismacoder.eval_metrics.exact_match import exact_match
from prismacoder.eval_metrics.pass_at_k import pass_at_k_curve, unbiased_pass_at_k
from prismacoder.eval_metrics.registry import aggregate_results, default_registry, wilson_ci
from prismacoder.eval_metrics.tree_edit_distance import (
    bracket_tree,
    prisma_tree_edit_distance,
    tree_similarity,
)


class TestExactMatch:
    def test_whitespace_insensitive(self):
        assert exact_match("a b;\nc", "a   b;\nc") == 1.0

    def test_comment_insensitive(self):
        assert exact_match("// note\nfindMany()", "findMany()") == 1.0

    def test_different(self):
        assert exact_match("prisma.user.findMany()", "prisma.post.findMany()") == 0.0


class TestPassAtK:
    def test_chen_examples(self):
        # values from the Codex paper's worked example table
        assert unbiased_pass_at_k(1, 1, 1) == 1.0
        assert unbiased_pass_at_k(10, 1, 1) == 0.1
        assert abs(unbiased_pass_at_k(10, 1, 10) - 1.0) < 1e-9
        assert abs(unbiased_pass_at_k(10, 1, 5) - 0.5) < 1e-6  # Codex worked example

    def test_curve_monotone(self):
        curve = pass_at_k_curve([True, False, False, False, False], ks=(1, 2, 3, 5))
        vals = [curve[k] for k in sorted(curve)]
        assert vals == sorted(vals)


class TestBLEU:
    def test_identical_is_one(self):
        text = "prisma.user.findMany({ where: { email: { contains: q } } })"
        assert sentence_bleu(text, text) > 0.99

    def test_disjoint_is_zero(self):
        assert sentence_bleu("aaa bbb ccc", "xxx yyy zzz") == 0.0

    def test_corpus(self):
        text = (
            "export async function listUsers() {\n"
            "  return prisma.user.findMany({ take: 20, orderBy: { createdAt: 'desc' } });\n"
            "}"
        )
        assert corpus_bleu([text], [text]) > 0.99


class TestCodeBLEU:
    def test_same_code_scores_high(self):
        code = "export async function listUsers() { return prisma.user.findMany({ take: 20 }); }"
        assert code_bleu(code, code) > 0.9

    def test_different_code_scores_lower(self):
        a = "prisma.user.findMany({ take: 20 })"
        b = "prisma.product.aggregate({ _avg: { price: true } })"
        assert code_bleu(a, b) < 0.6


class TestTED:
    def test_identical_zero_distance(self):
        code = "model User { id Int @id }"
        assert prisma_tree_edit_distance(code, code) == 0
        assert tree_similarity(code, code) == 1.0

    def test_bracket_tree(self):
        tree = bracket_tree("f({ a: 1 })")
        assert tree[0] == "root"

    def test_distance_positive(self):
        assert prisma_tree_edit_distance("{a:1}", "{a:2}") > 0


class TestRegistry:
    def test_wilson_ci(self):
        lo, hi = wilson_ci(6, 10)
        assert 0.25 < lo < 0.35 < hi < 0.95

    def test_aggregate(self):
        rows = [
            {
                "bench_id": "t1",
                "category": "T4",
                "level": "L1",
                "passed": True,
                "exact_match": 1.0,
                "code_bleu": 0.8,
            },
            {
                "bench_id": "t2",
                "category": "T4",
                "level": "L2",
                "passed": False,
                "error_class": "E4",
                "exact_match": 0.0,
                "code_bleu": 0.3,
            },
        ]
        agg = aggregate_results(rows)
        assert agg["n_tasks"] == 2
        assert agg["pass@1"]["rate"] == 0.5
        assert agg["by_category"]["T4"]["n"] == 2
        assert agg["error_taxonomy"] == {"E4": 1}

    def test_registry_evaluate(self):
        reg = default_registry()
        out = reg.evaluate("x", "x", {})
        assert "exact_match" in out and "code_bleu" in out
