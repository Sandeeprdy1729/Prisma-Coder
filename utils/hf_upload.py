#!/usr/bin/env python3
"""HuggingFace Hub upload pipeline: dataset + model + benchmark repos.

    # dataset release (train/val/test + cards + metadata)
    python utils/hf_upload.py dataset --dir datasets --repo prismacoder/PrismaForge

    # model release (merged dir + model card)
    python utils/hf_upload.py model --dir runs/prismacoder-3b-qlora-sft/merged \
        --repo prismacoder/PrismaCoder-3B

    # benchmark release (public tasks + provenance; private keys stay local)
    python utils/hf_upload.py benchmark --dir benchmark/public \
        --repo prismacoder/PrismaBench --extra benchmark/README.md:README.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATASET_FILES = (
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
    "benchmark.jsonl",
    "dpo_pairs.jsonl",
    "metadata.json",
    "schema.json",
    "DATASET_CARD.md",
)
BENCHMARK_FILES = ("prismabench_public.jsonl", "benchmark_provenance.json")


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface-hub required: pip install -e '.[hub]'")
        sys.exit(2)
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set")
        sys.exit(2)
    return HfApi()


def upload_dataset(args) -> None:
    api = _api()
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    d = Path(args.dir)
    for name in DATASET_FILES:
        p = d / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=args.repo, repo_type="dataset")
            print("uploaded:", name)
    if (d / "DATASET_CARD.md").exists():
        api.upload_file(
            path_or_fileobj=str(d / "DATASET_CARD.md"),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="dataset",
        )
    print(f"dataset release complete: https://huggingface.co/datasets/{args.repo}")


def upload_model(args) -> None:
    api = _api()
    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo, repo_type="model", folder_path=args.dir)
    card = Path(args.dir) / "MODEL_CARD.md"
    if card.exists():
        api.upload_file(
            path_or_fileobj=str(card), path_in_repo="README.md", repo_id=args.repo, repo_type="model"
        )
    print(f"model release complete: https://huggingface.co/{args.repo}")


def upload_benchmark(args) -> None:
    api = _api()
    api.create_repo(repo_id=args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    d = Path(args.dir)
    for name in BENCHMARK_FILES:
        p = d / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=args.repo, repo_type="dataset")
            print("uploaded:", name)
    for spec in args.extra or []:
        src, _, dst = spec.partition(":")
        if src and Path(src).exists():
            api.upload_file(
                path_or_fileobj=src,
                path_in_repo=dst or Path(src).name,
                repo_id=args.repo,
                repo_type="dataset",
            )
            print("uploaded:", dst or Path(src).name)
    print(f"benchmark release complete: https://huggingface.co/datasets/{args.repo}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HF Hub upload pipeline")
    sub = ap.add_subparsers(dest="kind", required=True)

    p1 = sub.add_parser("dataset")
    p1.add_argument("--dir", default="datasets")
    p1.add_argument("--repo", default="prismacoder/PrismaForge")
    p1.add_argument("--private", action="store_true")

    p2 = sub.add_parser("model")
    p2.add_argument("--dir", required=True)
    p2.add_argument("--repo", default="prismacoder/PrismaCoder-3B")
    p2.add_argument("--private", action="store_true")

    p3 = sub.add_parser("benchmark")
    p3.add_argument("--dir", default="benchmark/public")
    p3.add_argument("--repo", default="prismacoder/PrismaBench")
    p3.add_argument("--private", action="store_true")
    p3.add_argument(
        "--extra", nargs="*", default=[], help="src:dst extra files (e.g. benchmark/README.md:README.md)"
    )

    args = ap.parse_args(argv)
    {"dataset": upload_dataset, "model": upload_model, "benchmark": upload_benchmark}[args.kind](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
