#!/usr/bin/env python3
"""Publication-quality charts from evaluation artifacts (report Part 13).

Charts (matplotlib, no seaborn dependency):
  1. category x model pass@1 grouped bars with Wilson CIs
  2. difficulty ladder (L1/L2/L3) per model
  3. gate profile stacked bars (first-failing-gate distribution)
  4. error taxonomy Pareto bar chart
  5. pass@k curve from sample_passes
  6. composite leaderboard bar chart

Usage:
    python evaluation/plots.py --results a=benchmark/leaderboard/a.json \
        b=benchmark/leaderboard/b.json --out-dir reports/figures
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prismacoder.constants import ERROR_TAXONOMY  # noqa: E402
from prismacoder.io_utils import read_json  # noqa: E402

PALETTE = ["#2563eb", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]


def _category_rates(rows: list[dict]) -> dict[str, tuple[int, int]]:
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        agg[r.get("category", "?")][0] += 1 if r.get("passed") else 0
        agg[r.get("category", "?")][1] += 1
    return {k: (v[1], v[0]) for k, v in agg.items()}  # category -> (n, passed)


def plot_category_comparison(results: dict[str, list[dict]], out_dir: Path) -> Path:
    cats = sorted({c for rows in results.values() for c, _ in _category_rates(rows).items()})
    x = range(len(cats))
    width = 0.8 / max(len(results), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(cats) * 0.7), 4.5), constrained_layout=True)
    for i, (model, rows) in enumerate(results.items()):
        rates = _category_rates(rows)
        ys = [rates.get(c, (0, 0))[1] / max(rates.get(c, (1, 0))[0], 1) for c in cats]
        ax.bar([xi + i * width for xi in x], ys, width, label=model, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks([xi + width * (len(results) - 1) / 2 for xi in x])
    ax.set_xticklabels([c.replace("_", "\n") for c in cats], fontsize=8)
    ax.set_ylabel("pass@1")
    ax.set_ylim(0, 1)
    ax.set_title("PrismaBench pass@1 by category")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.18), ncol=min(len(results), 4), fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "category_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_level_ladder(results: dict[str, list[dict]], out_dir: Path) -> Path:
    levels = ["L1", "L2", "L3"]
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    for i, (model, rows) in enumerate(results.items()):
        ys = []
        for lv in levels:
            sel = [r for r in rows if r.get("level") == lv]
            ys.append(sum(1 for r in sel if r.get("passed")) / max(len(sel), 1))
        ax.plot(levels, ys, marker="o", label=model, color=PALETTE[i % len(PALETTE)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("pass@1")
    ax.set_title("Difficulty ladder (L1 -> L3)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.15))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "difficulty_ladder.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_gate_profile(rows: list[dict], out_dir: Path, model: str = "") -> Path:
    gate_order = ["V1_validate", "V2_generate", "V3_tsc", "V4_execute", "V5_vitest", "V0_static"]
    counts: dict[str, int] = {g: 0 for g in gate_order}
    counts["passed"] = 0
    for r in rows:
        if r.get("passed"):
            counts["passed"] += 1
        else:
            counts[r.get("first_failing_gate", "V0_static")] = (
                counts.get(r.get("first_failing_gate", "V0_static"), 0) + 1
            )
    labels = [k for k, v in counts.items() if v > 0]
    vals = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.barh(labels[::-1], vals[::-1], color="#2563eb")
    ax.set_xlabel("tasks")
    ax.set_title(f"Gate profile{(' - ' + model) if model else ''}")
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / f"gate_profile{('_' + model) if model else ''}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_taxonomy(rows: list[dict], out_dir: Path) -> Path:
    from collections import Counter

    taxonomy = Counter(r.get("error_class", "?") for r in rows if not r.get("passed"))
    if not taxonomy:
        return out_dir / "error_taxonomy.png"
    classes = [c for c, _ in taxonomy.most_common()]
    vals = [taxonomy[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(classes, vals, color="#ef4444")
    ax.set_ylabel("failed tasks")
    ax.set_title("Error taxonomy (E1-E10)")
    for i, c in enumerate(classes):
        ax.annotate(
            ERROR_TAXONOMY.get(c, "").split(" - ")[0], (i, vals[i]), ha="center", va="bottom", fontsize=7
        )
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "error_taxonomy.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_passk(rows: list[dict], out_dir: Path, model: str = "") -> Path:
    from prismacoder.eval_metrics.pass_at_k import unbiased_pass_at_k

    sample_rows = [r for r in rows if r.get("sample_passes")]
    if not sample_rows:
        return out_dir / "pass_at_k.png"
    ks = [1, 2, 3, 5, 10]
    max_n = max(len(r["sample_passes"]) for r in sample_rows)
    ks = [k for k in ks if k <= max_n] or [1]
    ys = [
        sum(unbiased_pass_at_k(len(r["sample_passes"]), sum(r["sample_passes"]), k) for r in sample_rows)
        / len(sample_rows)
        for k in ks
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    ax.plot(ks, ys, marker="o", color="#2563eb")
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title(f"pass@k curve{(' - ' + model) if model else ''}")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "pass_at_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation charts")
    ap.add_argument(
        "--results", nargs="+", required=True, help="name=path/to/results.json pairs (repeatable)"
    )
    ap.add_argument("--out-dir", default="reports/figures")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[dict]] = {}
    for spec in args.results:
        name, _, path = spec.partition("=")
        data = read_json(path)
        results[name or Path(path).stem] = data.get("rows", [])

    produced = []
    if len(results) >= 1:
        produced.append(plot_category_comparison(results, out_dir))
        produced.append(plot_level_ladder(results, out_dir))
        for name, rows in results.items():
            produced.append(plot_gate_profile(rows, out_dir, name))
            produced.append(plot_passk(rows, out_dir, name))
        first = next(iter(results.values()), [])
        if first:
            produced.append(plot_taxonomy(first, out_dir))
    for p in produced:
        print("chart:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
