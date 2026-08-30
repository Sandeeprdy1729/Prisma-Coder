#!/usr/bin/env python3
"""PrismaForge stage 01 - crawl GitHub for Prisma-bearing repositories.

Usage:
    python datasets/scripts/01_crawl_github.py [--out datasets] [--discover 50]

Requires GITHUB_TOKEN for anything beyond a handful of requests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.crawler import main as crawler_main  # noqa: E402


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage 01: GitHub crawl (mined track)")
    parser.add_argument("--out", default="datasets")
    parser.add_argument("--discover", type=int, default=0, help="extra repos found via GitHub code search")
    parser.add_argument("--max-ts", type=int, default=400)
    parser.add_argument("--no-ecosystem", action="store_true")
    parser.add_argument("--seed-repos", default="")
    parser.add_argument("--log-level", default="INFO")
    args, _unknown = parser.parse_known_args(argv)

    # delegate to the crawler's own CLI for exact-parity behavior
    passthrough = ["--out", args.out, "--max-ts", str(args.max_ts), "--log-level", args.log_level]
    if args.discover:
        passthrough += ["--discover", str(args.discover)]
    if args.no_ecosystem:
        passthrough.append("--no-ecosystem")
    if args.seed_repos:
        passthrough += ["--seed-repos", args.seed_repos]
    return crawler_main(passthrough)


if __name__ == "__main__":
    raise SystemExit(cli())
