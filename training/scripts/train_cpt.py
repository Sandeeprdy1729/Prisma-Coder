#!/usr/bin/env python3
"""CPT - continued pretraining (stage S0). Plain causal LM with packing over
the Prisma-dense corpus (mined TS + schemas + docs, 8-12B tokens, LR 8e-5).

    python training/scripts/train_cpt.py --config training/configs/cpt.yaml

The corpus file is a jsonl of {text, family} rows built from:
  - datasets/processed/clean_records.jsonl  (schemas + call-site snippets)
  - prisma/docs + prisma/website snapshots  (Apache-2.0 docs track)
Build it with `python training/scripts/build_cpt_corpus.py` (thin wrapper below
is auto-invoked when the corpus file is missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


from training.scripts.train_sft import load_config  # noqa: E402


def build_cpt_corpus(clean_file: str, docs_dir: str, out_file: str, target_rows: int = 200_000) -> int:
    """Assemble {text, family} rows: schemas, client snippets, docs, TS sources."""
    from prismacoder.io_utils import read_jsonl, write_jsonl

    rows: list[dict] = []
    if Path(clean_file).exists():
        for rec in read_jsonl(clean_file):
            schema = rec.get("schema", "")
            if schema.strip():
                rows.append(
                    {"text": "<|schema|>\n" + schema.strip() + "\n<|/schema|>", "family": "schemas_psl"}
                )
            for site in rec.get("call_sites", [])[:50]:
                if site.get("snippet"):
                    rows.append(
                        {"text": "<|client|>\n" + site["snippet"] + "\n<|/client|>", "family": "mined_ts"}
                    )
    docs = Path(docs_dir)
    if docs.exists():
        for md in docs.rglob("*.md*"):
            try:
                text = md.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for chunk in range(0, min(len(text), 24000), 4000):
                piece = text[chunk : chunk + 4000].strip()
                if len(piece) > 200:
                    rows.append({"text": piece, "family": "docs"})
            if len(rows) >= target_rows:
                break
    write_jsonl(out_file, rows)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PrismaCoder CPT trainer (S0)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--build-corpus", action="store_true", help="(re)build the CPT corpus then exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    corpus = data_cfg["train_file"]
    if args.build_corpus or not Path(corpus).exists():
        n = build_cpt_corpus(
            "datasets/processed/clean_records.jsonl", "datasets/raw/prisma__docs/files", corpus
        )
        print(f"CPT corpus: {n} rows -> {corpus}")
        if args.build_corpus:
            return 0
    if args.dry_run:
        from prismacoder.io_utils import count_lines

        print(f"[dry-run] corpus rows: {count_lines(corpus)}")
        return 0

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from datasets import Dataset

    model_cfg, tok_cfg, train_cfg = cfg["model"], cfg["tokenizer"], cfg["training"]
    tokenizer = AutoTokenizer.from_pretrained(tok_cfg.get("name_or_path", model_cfg["name_or_path"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16
    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype}
    if model_cfg.get("attn_implementation"):
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)

    from prismacoder.io_utils import read_jsonl

    rows = read_jsonl(corpus)
    ds = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=8192)

    ds = ds.map(
        tokenize, batched=True, num_proc=data_cfg.get("worker_count", 8), remove_columns=ds.column_names
    )
    ds = ds.map(lambda b: {"labels": b["input_ids"]}, batched=True)

    targs = TrainingArguments(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 20260830),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 1),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
        bf16=train_cfg.get("bf16", True),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        logging_steps=train_cfg.get("logging_steps", 20),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 2000),
        save_total_limit=train_cfg.get("save_total_limit", 4),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 8),
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(str(Path(cfg["output_dir"]) / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
