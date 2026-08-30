#!/usr/bin/env python3
"""Push the PrismaCoder artifacts to the HuggingFace Hub.

    python training/scripts/push_to_hub.py --model runs/.../merged \
        --repo prismacoder/PrismaCoder-3B [--dataset datasets --dataset-repo ...]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _require_hub():
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        print("huggingface-hub required: pip install -e '.[hub]'")
        sys.exit(2)


def push_model(model_dir: str, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=model_dir)
    print(f"model pushed: https://huggingface.co/{repo_id}")


def push_dataset(dataset_dir: str, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    for name in (
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "benchmark.jsonl",
        "dpo_pairs.jsonl",
        "metadata.json",
        "schema.json",
        "DATASET_CARD.md",
    ):
        p = Path(dataset_dir) / name
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=name, repo_id=repo_id, repo_type="dataset")
    print(f"dataset pushed: https://huggingface.co/datasets/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="HuggingFace Hub release")
    ap.add_argument("--model", default="", help="merged model dir")
    ap.add_argument("--repo", default=os.environ.get("HF_USERNAME", "prismacoder") + "/PrismaCoder-3B")
    ap.add_argument("--private", action="store_true", default=False)
    ap.add_argument("--dataset", default="", help="datasets/ dir (train.jsonl ...)")
    ap.add_argument("--dataset-repo", default=os.environ.get("HF_USERNAME", "prismacoder") + "/PrismaForge")
    args = ap.parse_args(argv)

    _require_hub()
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN not set - run `huggingface-cli login` or export HF_TOKEN")
        return 2
    if args.model:
        push_model(args.model, args.repo, args.private)
    if args.dataset:
        push_dataset(args.dataset, args.dataset_repo, args.private)
    if not args.model and not args.dataset:
        print("nothing to push: pass --model and/or --dataset")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
