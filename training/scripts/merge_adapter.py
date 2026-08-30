#!/usr/bin/env python3
"""Merge a PEFT adapter into the base model and save a standalone model.

    python training/scripts/merge_adapter.py \
        --base Qwen/Qwen2.5-Coder-3B-Instruct \
        --adapter runs/prismacoder-3b-qlora-sft/final \
        --out runs/prismacoder-3b-qlora-sft/merged [--dtype bf16]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Merge LoRA/QLoRA adapter into base model")
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"loading base: {args.base} ({args.dtype})")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, trust_remote_code=True, device_map="cpu"
    )
    print(f"attaching adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)
    print("merging weights...")
    model = model.merge_and_unload()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter if (Path(args.adapter) / "tokenizer_config.json").exists() else args.base
    )
    tokenizer.save_pretrained(out)
    print(f"merged model saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
