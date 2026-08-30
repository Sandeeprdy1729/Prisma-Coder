#!/usr/bin/env python3
"""Paired model comparison with statistical tests (report 11.3).

McNemar's test for paired binary outcomes (same tasks, two systems) +
paired bootstrap (10,000 resamples) for composite differences. Holm-Bonferroni
correction across the comparison grid.

Usage:
    python evaluation/compare_models.py --baseline base.json --treatment sft.json \
        [--treatment more.json ...] --alpha 0.05
"""

from __future__ import annotations

import argparse
import random
import sys
from math import comb
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prismacoder.io_utils import read_json, write_json  # noqa: E402


def _pass_map(results: dict) -> dict[str, bool]:
    return {r["bench_id"]: bool(r.get("passed")) for r in results.get("rows", [])}


def mcnemar_exact(b: int, c: int) -> float:
    """Exact McNemar p-value for discordant counts (b, c): two-sided binomial test."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, tail * (2 if b != c else 1))


def paired_bootstrap(
    composites_a: list[float], composites_b: list[float], resamples: int = 10_000, seed: int = 20260830
) -> tuple[float, float]:
    """Returns (mean delta = B - A, two-sided p) via bootstrap over task indices."""
    rng = random.Random(seed)
    n = len(composites_a)
    deltas = [b - a for a, b in zip(composites_a, composites_b, strict=False)]
    observed = sum(deltas) / max(n, 1)
    count = 0
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        if (sum(sample) / n) * (1 if observed >= 0 else -1) <= 0:
            count += 1
    return round(observed, 4), min(1.0, 2 * count / resamples)


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Returns reject decisions under Holm-Bonferroni FWER control."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    m = len(pvalues)
    rejected = [False] * m
    max_i = 0
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[idx] <= threshold:
            max_i = rank
            rejected[idx] = True
        else:
            break
    # Holm: all hypotheses before the first non-rejection are rejected
    for rank, idx in enumerate(order):
        rejected[idx] = rank <= max_i
    return rejected


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Paired model comparison")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--treatment", action="append", required=True)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="reports/comparison.json")
    args = ap.parse_args(argv)

    base = read_json(args.baseline)
    base_pass = _pass_map(base)
    treatments = [(Path(p).stem, read_json(p)) for p in args.treatment]

    comparisons = []
    pvals = []
    for name, treat in treatments:
        tp = _pass_map(treat)
        shared = sorted(set(base_pass) & set(tp))
        b01 = sum(1 for t in shared if base_pass[t] and not tp[t])  # base pass, treat fail
        b10 = sum(1 for t in shared if not base_pass[t] and tp[t])  # treat pass, base fail
        p_mcnemar = mcnemar_exact(b10, b01)
        delta, p_boot = paired_bootstrap(
            [
                r.get("composite", 0.0)
                for r in base.get("rows", [])
                if r["bench_id"] in set(base_pass) & set(tp)
            ],
            [
                r.get("composite", 0.0)
                for r in treat.get("rows", [])
                if r["bench_id"] in set(base_pass) & set(tp)
            ],
        )
        comparisons.append(
            {
                "treatment": name,
                "n_shared": len(shared),
                "base_only_passes": b01,
                "treatment_only_passes": b10,
                "pass1_base": round(sum(base_pass[t] for t in shared) / max(len(shared), 1), 4),
                "pass1_treatment": round(sum(tp[t] for t in shared) / max(len(shared), 1), 4),
                "composite_delta": delta,
                "p_mcnemar": round(p_mcnemar, 5),
                "p_bootstrap": round(p_boot, 5),
            }
        )
        pvals.append(p_mcnemar)

    decisions = holm_bonferroni(pvals, args.alpha)
    for comp, rej in zip(comparisons, decisions, strict=False):
        comp["significant_holm"] = rej

    out = {"baseline": Path(args.baseline).stem, "alpha": args.alpha, "comparisons": comparisons}
    write_json(args.out, out)
    for comp in comparisons:
        flag = "SIGNIFICANT" if comp["significant_holm"] else "n.s."
        print(
            f"{comp['treatment']}: pass@1 {comp['pass1_base']} -> {comp['pass1_treatment']} "
            f"(delta {comp['composite_delta']}, McNemar p={comp['p_mcnemar']}) [{flag}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
