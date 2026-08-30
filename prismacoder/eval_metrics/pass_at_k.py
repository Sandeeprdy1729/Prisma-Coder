"""Unbiased pass@k estimator (Chen et al. 2021, report 11.5)."""

from __future__ import annotations

from math import comb


def unbiased_pass_at_k(n: int, c: int, k: int) -> float:
    """P(at least one of k random draws from n samples passes).

    n: total samples, c: passing samples. Exact closed form:
        1 - C(n-c, k) / C(n, k)
    """
    if n <= 0 or k <= 0:
        return 0.0
    if k > n:
        k = n
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return round(1.0 - comb(n - c, k) / comb(n, k), 6)


def pass_at_k(samples: list[bool], k: int) -> float:
    """Convenience wrapper: pass@k over a boolean sample list."""
    n, c = len(samples), sum(1 for s in samples if s)
    return unbiased_pass_at_k(n, c, k)


def pass_at_k_curve(sample_passes: list[bool], ks: tuple[int, ...] = (1, 2, 5, 10)) -> dict[int, float]:
    """The report requires pass@k 'as a curve, not a point' (11.2)."""
    n = len(sample_passes)
    return {k: unbiased_pass_at_k(n, sum(sample_passes), min(k, n)) for k in ks if n > 0}
