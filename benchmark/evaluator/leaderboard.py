"""Leaderboard assembly (report Part 10 governance).

Aggregates ``results.json`` files under benchmark/leaderboard/ into a sorted
markdown leaderboard. The primary metric is pass@1 (full-gate mode); static
mode entries are annotated and ranked separately so proxy scores never masquerade
as gate-green numbers. Token/latency/memory accounting is mandatory (O5).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prismacoder.io_utils import read_json, write_json  # noqa: E402


def collect(results_dir: Path) -> list[dict]:
    entries = []
    for p in sorted(results_dir.glob("*.json")):
        if p.name == "eval_results.json" or p.stem.startswith("smoke"):
            pass  # still eligible; naming is not a filter
        try:
            data = read_json(p)
        except Exception:  # noqa: BLE001
            continue
        if "aggregate" not in data:
            continue
        agg = data["aggregate"]
        entries.append(
            {
                "model": data.get("model", p.stem),
                "mode": data.get("mode", "static"),
                "n_tasks": agg.get("n_tasks", 0),
                "pass@1": agg.get("pass@1", {}).get("rate", 0.0),
                "wilson95": agg.get("pass@1", {}).get("wilson95", [0.0, 0.0]),
                "pass@5": agg.get("pass@5"),
                "code_bleu": agg.get("code_bleu"),
                "latency_ms_avg": agg.get("latency_ms_avg"),
                "memory_mb_peak": agg.get("memory_mb_peak"),
                "error_taxonomy": agg.get("error_taxonomy", {}),
                "source": str(p),
            }
        )
    return entries


def render_markdown(entries: list[dict]) -> str:
    lines = [
        "# PrismaBench Leaderboard",
        "",
        "Primary metric: **pass@1** (gate-chain verdict; execution-weighted composite governs ties).",
        "Static-mode rows are ranked separately - they do not use the full V1-V5 gates.",
        "",
        "| Rank | Model | Mode | Tasks | pass@1 | 95% CI | pass@5 | CodeBLEU | Latency avg (ms) | Peak mem (MB) |",
        "|---:|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    full = sorted([e for e in entries if e["mode"] == "full_gates"], key=lambda e: -e["pass@1"])
    static = sorted([e for e in entries if e["mode"] != "full_gates"], key=lambda e: -e["pass@1"])
    rank = 0
    for e in full:
        rank += 1
        lines.append(
            f"| {rank} | {e['model']} | full | {e['n_tasks']} | {e['pass@1']:.3f} "
            f"| [{e['wilson95'][0]:.3f}, {e['wilson95'][1]:.3f}] "
            f"| {e['pass@5'] if e['pass@5'] is not None else '-'} "
            f"| {e['code_bleu'] if e['code_bleu'] is not None else '-'} "
            f"| {e['latency_ms_avg'] if e['latency_ms_avg'] is not None else '-'} "
            f"| {e['memory_mb_peak'] if e['memory_mb_peak'] is not None else '-'} |"
        )
    rank = 0
    for e in static:
        rank += 1
        lines.append(
            f"| s{rank} | {e['model']} | static | {e['n_tasks']} | {e['pass@1']:.3f} "
            f"| [{e['wilson95'][0]:.3f}, {e['wilson95'][1]:.3f}] "
            f"| {e['pass@5'] if e['pass@5'] is not None else '-'} "
            f"| {e['code_bleu'] if e['code_bleu'] is not None else '-'} "
            f"| {e['latency_ms_avg'] if e['latency_ms_avg'] is not None else '-'} "
            f"| {e['memory_mb_peak'] if e['memory_mb_peak'] is not None else '-'} |"
        )
    lines.append("")
    lines.append(
        "_Submission guide: see docs/BENCHMARK_CARD.md - latency, memory and token cost columns "
        "are mandatory for full-gate entries (O5 cost honesty)._"
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Assemble the PrismaBench leaderboard")
    ap.add_argument("--results", default="benchmark/leaderboard")
    ap.add_argument("--out", default="benchmark/leaderboard/LEADERBOARD.md")
    ap.add_argument("--json-out", default="benchmark/leaderboard/leaderboard.json")
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    entries = collect(results_dir)
    md = render_markdown(entries)
    Path(args.out).write_text(md, encoding="utf-8")
    write_json(args.json_out, {"entries": entries})
    print(f"leaderboard: {len(entries)} entries -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
