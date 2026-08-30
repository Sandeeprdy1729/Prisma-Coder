#!/usr/bin/env python3
"""Error analysis over evaluation result rows (report 11.4 / Appendix D).

Groups failed tasks by:
- error taxonomy class E1-E10 (from the evaluator's classifier)
- first failing gate (V1-V5)
- category x level matrix

Emits a markdown report + json with the Pareto frontier of fixes: the
(error_class, category) pairs whose repair would recover the most tasks.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prismacoder.constants import ERROR_TAXONOMY  # noqa: E402
from prismacoder.io_utils import read_json, write_json  # noqa: E402


def analyze(rows: list[dict]) -> dict:
    failures = [r for r in rows if not r.get("passed")]
    by_class = Counter(r.get("error_class", "unclassified") for r in failures)
    by_gate = Counter(r.get("first_failing_gate", "-") for r in failures)
    matrix: dict[str, Counter] = defaultdict(Counter)
    for r in failures:
        matrix[r.get("category", "?")][r.get("error_class", "?")] += 1
    pair_counts = Counter((r.get("category", "?"), r.get("error_class", "?")) for r in failures)
    return {
        "n_tasks": len(rows),
        "n_failed": len(failures),
        "taxonomy": dict(by_class.most_common()),
        "taxonomy_legend": ERROR_TAXONOMY,
        "first_failing_gate": dict(by_gate.most_common()),
        "category_x_class": {k: dict(v) for k, v in sorted(matrix.items())},
        "pareto_fixes": [
            {"category": c, "error_class": e, "recoverable": n, "meaning": ERROR_TAXONOMY.get(e, "")}
            for (c, e), n in pair_counts.most_common(10)
        ],
    }


def render_md(analysis: dict) -> str:
    lines = [
        "# Error Analysis (E1-E10)",
        "",
        f"Tasks: {analysis['n_tasks']} | failures: {analysis['n_failed']} "
        f"({100.0 * analysis['n_failed'] / max(analysis['n_tasks'], 1):.1f}%)",
        "",
        "## Taxonomy distribution",
        "",
        "| Class | Count | Meaning |",
        "|---|---:|---|",
    ]
    for cls, n in analysis["taxonomy"].items():
        lines.append(f"| {cls} | {n} | {ERROR_TAXONOMY.get(cls, '')} |")
    lines += ["", "## First failing gate", "", "| Gate | Count |", "|---|---:|"]
    for gate, n in analysis["first_failing_gate"].items():
        lines.append(f"| {gate} | {n} |")
    lines += [
        "",
        "## Pareto frontier of fixes",
        "",
        "Repairing these (category, class) pairs recovers the most tasks:",
        "",
    ]
    for p in analysis["pareto_fixes"]:
        lines.append(
            f"- **{p['category']}** x **{p['error_class']}**: {p['recoverable']} tasks " f"- {p['meaning']}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1-E10 error analysis")
    ap.add_argument("--results", required=True, help="evaluate.py results.json")
    ap.add_argument("--out-md", default="reports/error_analysis.md")
    ap.add_argument("--out-json", default="reports/error_analysis.json")
    args = ap.parse_args(argv)

    data = read_json(args.results)
    rows = data.get("rows", [])
    analysis = analyze(rows)
    analysis["model"] = data.get("model", "?")
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out_json, analysis)
    Path(args.out_md).write_text(render_md(analysis), encoding="utf-8")
    print(f"error analysis -> {args.out_md} / {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
