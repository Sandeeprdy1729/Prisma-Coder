"""Metric suite (report 11.1/11.2): exact match, pass@k, BLEU, CodeBLEU,
tree edit distance, latency/memory aggregation. Dependency-free."""

from prismacoder.eval_metrics.bleu import corpus_bleu, sentence_bleu
from prismacoder.eval_metrics.code_bleu import code_bleu
from prismacoder.eval_metrics.exact_match import exact_match, normalized_artifact
from prismacoder.eval_metrics.pass_at_k import pass_at_k, unbiased_pass_at_k
from prismacoder.eval_metrics.registry import MetricRegistry, aggregate_results
from prismacoder.eval_metrics.tree_edit_distance import prisma_tree_edit_distance

__all__ = [
    "MetricRegistry",
    "aggregate_results",
    "exact_match",
    "normalized_artifact",
    "pass_at_k",
    "unbiased_pass_at_k",
    "corpus_bleu",
    "sentence_bleu",
    "code_bleu",
    "prisma_tree_edit_distance",
]
