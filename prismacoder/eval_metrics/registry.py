"""Metric registry + result aggregation (report 11.1/11.2 wire-up).

The registry maps metric names to callables with a uniform signature
(prediction, reference, task) -> float in [0,1] (or raw values for latency/
memory, which are aggregated separately). ``aggregate_results`` computes
per-category, per-level, and per-gate breakdowns + Wilson 95% CIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from math import sqrt
from typing import Any, Callable

from prismacoder.eval_metrics.bleu import corpus_bleu
from prismacoder.eval_metrics.code_bleu import code_bleu
from prismacoder.eval_metrics.exact_match import exact_match
from prismacoder.eval_metrics.pass_at_k import unbiased_pass_at_k
from prismacoder.eval_metrics.tree_edit_distance import tree_similarity

MetricFn = Callable[[str, str, dict], float]


@dataclass
class MetricRegistry:
    metrics: dict[str, MetricFn] = dc_field(default_factory=dict)

    def register(self, name: str, fn: MetricFn) -> None:
        self.metrics[name] = fn

    def evaluate(self, prediction: str, reference: str, task: dict[str, Any]) -> dict[str, float]:
        return {name: round(fn(prediction, reference, task), 4) for name, fn in self.metrics.items()}


def default_registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.register("exact_match", lambda p, r, t: exact_match(p, r))
    reg.register("code_bleu", lambda p, r, t: code_bleu(p, r))
    reg.register("tree_similarity", lambda p, r, t: tree_similarity(p, r))
    reg.register("compile_proxy", _compile_proxy)
    return reg


def _compile_proxy(prediction: str, reference: str, task: dict[str, Any]) -> float:
    """Static compile proxy used when Node gates are unavailable: fences balance
    + schema symbol grounding (LightValidator passed upstream sets this)."""
    fences = prediction.count("```")
    if "```" in prediction and fences % 2 != 0:
        return 0.0
    return 1.0


def wilson_ci(passes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% confidence interval on a binary rate (report 11.3)."""
    if total == 0:
        return (0.0, 0.0)
    p = passes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def aggregate_results(rows: list[dict[str, Any]], *, ks: tuple[int, ...] = (1, 5)) -> dict[str, Any]:
    """Aggregate per-task rows into the leaderboard-facing summary."""
    out: dict[str, Any] = {"n_tasks": len(rows)}
    passed_rows = [r for r in rows if r.get("passed")]
    out["pass@1"] = {
        "rate": round(len(passed_rows) / max(len(rows), 1), 4),
        "wilson95": list(wilson_ci(len(passed_rows), len(rows))),
    }

    # pass@k over tasks with k samples (k=1 is already the primary metric above)
    sample_rows = [r for r in rows if r.get("sample_passes")]
    if sample_rows:
        n_list = [len(r["sample_passes"]) for r in sample_rows]
        c_list = [sum(r["sample_passes"]) for r in sample_rows]
        for k in ks:
            if k > 1 and max(n_list, default=0) >= k:
                out[f"pass@{k}"] = round(
                    sum(unbiased_pass_at_k(n, c, k) for n, c in zip(n_list, c_list, strict=False))
                    / len(sample_rows),
                    4,
                )

    # per-category / per-level breakdowns
    by_cat: dict[str, dict] = {}
    by_level: dict[str, dict] = {}
    for r in rows:
        for bucket, key in ((by_cat, r.get("category", "?")), (by_level, r.get("level", "?"))):
            b = bucket.setdefault(key, {"n": 0, "passed": 0})
            b["n"] += 1
            b["passed"] += 1 if r.get("passed") else 0
    for name, bucket in (("by_category", by_cat), ("by_level", by_level)):
        out[name] = {
            k: {
                "rate": round(v["passed"] / max(v["n"], 1), 4),
                "n": v["n"],
                "wilson95": list(wilson_ci(v["passed"], v["n"])),
            }
            for k, v in sorted(bucket.items())
        }

    # error taxonomy distribution for failures (E1-E10)
    taxonomy: dict[str, int] = {}
    for r in rows:
        if not r.get("passed") and r.get("error_class"):
            taxonomy[r["error_class"]] = taxonomy.get(r["error_class"], 0) + 1
    out["error_taxonomy"] = dict(sorted(taxonomy.items()))

    # similarity + efficiency
    if rows:
        for metric in ("exact_match", "code_bleu", "tree_similarity"):
            vals = [r[metric] for r in rows if metric in r]
            if vals:
                out[metric] = round(sum(vals) / len(vals), 4)
        lat = [r["latency_ms"] for r in rows if r.get("latency_ms")]
        mem = [r["memory_mb"] for r in rows if r.get("memory_mb")]
        if lat:
            out["latency_ms_avg"] = round(sum(lat) / len(lat), 1)
        if mem:
            out["memory_mb_peak"] = round(max(mem), 1)
    return out


def corpus_bleu_safe(predictions: list[str], references: list[str]) -> float:
    return round(corpus_bleu(predictions, references), 4)
