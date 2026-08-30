"""Build a dummy submission for harness smoke tests / CI.

The dummy either (a) echoes the task's context-free expectation (empty output)
or (b) replays a reference-aware heuristic when private keys are available.
CI uses (a): every task scores 0, exercising the full scoring path.

Usage:
    python benchmark/evaluator/make_dummy_submission.py \
        --tasks benchmark/public/prismabench_public.jsonl \
        --out benchmark/evaluator/dummy_submission.jsonl [--echo "schema"]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prismacoder.io_utils import read_jsonl, write_jsonl  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="benchmark/public/prismabench_public.jsonl")
    ap.add_argument("--out", default="benchmark/evaluator/dummy_submission.jsonl")
    ap.add_argument("--echo", default="", help="fixed string to emit as every output")
    args = ap.parse_args(argv)

    tasks = read_jsonl(args.tasks)
    rows = [
        {
            "bench_id": t["bench_id"],
            "output": args.echo,
            "latency_ms": 0,
            "memory_mb": 0,
        }
        for t in tasks
    ]
    n = write_jsonl(args.out, rows)
    print(f"wrote {n} dummy submissions -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
