#!/usr/bin/env python3
"""BIRD adapter: thin specialization of the Spider adapter (evidence-aware prompts,
BIRD dev.jsonl format). See spider_adapter.py for the shared mechanics.

Usage:
    python evaluation/bird_adapter.py --data bird/dev.jsonl --databases bird/dev_databases \
        --model runs/.../merged --limit 100 --out reports/bird_predictions.json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evaluation.spider_adapter import main as spider_main  # noqa: E402


def cli(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    if "--benchmark" not in argv:
        argv = ["--benchmark", "bird"] + argv
    return spider_main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
