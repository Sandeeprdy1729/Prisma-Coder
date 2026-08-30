#!/usr/bin/env python3
"""SFT training (TRL SFTTrainer) - QLoRA / LoRA / full fine-tuning.

    python training/scripts/train_sft.py --config training/configs/qlora_3b.yaml

Implements report Table 12.1 S1: LR cosine 3% warmup, 3 epochs, packing on,
NEFTune alpha 5, bf16, gate-aligned sanity checks during training.
Heavy imports are deferred so `--dry-run` works on CPU-only CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section in ("model", "tokenizer", "data", "training", "hub"):
        cfg.setdefault(section, {})
    return cfg


def validate_config(cfg: dict) -> list[str]:
    """Config lint used by tests and --dry-run (no GPU required)."""
    problems = []
    t = cfg.get("training", {})
    lr = t.get("learning_rate")
    if not lr or not (1e-7 < float(lr) <= 1e-3):
        problems.append(f"learning_rate {lr} outside plausible range (1e-7, 1e-3]")
    if t.get("lr_scheduler_type") != "cosine":
        problems.append("report Table 12.1 requires cosine schedule")
    if float(t.get("warmup_ratio", 0)) != 0.03:
        problems.append("report requires warmup_ratio 0.03")
    if not cfg.get("data", {}).get("train_file"):
        problems.append("data.train_file required")
    model = cfg.get("model", {})
    if "quantization" in model and not model["quantization"].get("load_in_4bit"):
        problems.append("quantization block present but load_in_4bit is false")
    if "peft" not in cfg and "quantization" in model:
        problems.append("QLoRA requires a peft block (adapter on quantized base)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PrismaCoder SFT trainer")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--dry-run", action="store_true", help="validate config + data availability without loading torch"
    )
    ap.add_argument("--max-samples", type=int, default=0, help="override: cap train records")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    problems = validate_config(cfg)
    if problems:
        for p in problems:
            print(f"[config] {p}")
        return 2

    train_file = cfg["data"]["train_file"]
    if not Path(train_file).exists():
        print(
            f"train file missing: {train_file} - build PrismaForge first "
            f"(bash datasets/scripts/run_phase1.sh)"
        )
        return 2

    if args.dry_run:
        from prismacoder.io_utils import count_lines

        n = count_lines(train_file)
        print(f"[dry-run] config OK; train records: {n}; run_name: {cfg.get('run_name')}")
        return 0

    # ---- heavy imports (training only) ----
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from datasets import Dataset
    from prismacoder.constants import SYSTEM_PROMPT
    from training.trainer.callbacks import SanitySpec, make_sanity_callback
    from training.trainer.data import load_sft_records, to_chat_example

    model_cfg, tok_cfg, data_cfg = cfg["model"], cfg["tokenizer"], cfg["data"]
    train_cfg = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(tok_cfg.get("name_or_path", model_cfg["name_or_path"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"trust_remote_code": model_cfg.get("trust_remote_code", True)}
    dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16
    model_kwargs["torch_dtype"] = dtype
    if model_cfg.get("attn_implementation"):
        try:
            import flash_attn  # noqa: F401

            model_kwargs["attn_implementation"] = model_cfg["attn_implementation"]
        except ImportError:
            model_kwargs["attn_implementation"] = "sdpa"

    quant = model_cfg.get("quantization")
    if quant and quant.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)

    peft_cfg = cfg.get("peft")
    if peft_cfg:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if quant:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            task_type=peft_cfg.get("task_type", "CAUSAL_LM"),
            r=peft_cfg.get("r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            target_modules=peft_cfg.get("target_modules"),
            bias=peft_cfg.get("bias", "none"),
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    train_records, val_records = load_sft_records(
        data_cfg["train_file"], data_cfg.get("validation_file"), limit=args.max_samples or None
    )

    with_reasoning = True  # ~10% carry plan-spans (attached in stage 04)

    def _format(rec: dict) -> dict:
        ex = to_chat_example(
            rec,
            tokenizer,
            system_prompt=data_cfg.get("system_prompt", SYSTEM_PROMPT),
            with_reasoning=with_reasoning and "reasoning" in rec,
        )
        return {"text": tokenizer.decode(ex["input_ids"])}

    train_ds = Dataset.from_list([{"text": _format(r)["text"]} for r in train_records])
    val_ds = Dataset.from_list([{"text": _format(r)["text"]} for r in val_records]) if val_records else None

    sanity = cfg.get("generation_sanity", {}) or {}
    sanity_examples = train_records[: sanity.get("num_examples", 2)]

    sft_cfg = SFTConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 20260830),
        num_train_epochs=train_cfg.get("num_train_epochs", 3),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 8),
        bf16=train_cfg.get("bf16", True),
        tf32=train_cfg.get("tf32", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps" if val_ds is not None else "no",
        eval_steps=train_cfg.get("eval_steps", 250),
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        group_by_length=train_cfg.get("group_by_length", True),
        max_seq_length=data_cfg.get("max_seq_length", 4096),
        packing=data_cfg.get("packing", True),
        neftune_noise_alpha=data_cfg.get("neftune_noise_alpha", 5),
        completion_only_loss=False,  # loss handled via packing + prompt masking in data.py
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )
    if sanity.get("enabled"):
        trainer.add_callback(
            make_sanity_callback(
                tokenizer,
                model,
                sanity_examples,
                SanitySpec(
                    enabled=True,
                    every_steps=sanity.get("every_steps", 500),
                    num_examples=sanity.get("num_examples", 2),
                    max_new_tokens=sanity.get("max_new_tokens", 512),
                    temperature=sanity.get("temperature", 0.2),
                ),
            )
        )

    trainer.train()
    trainer.save_model(str(Path(cfg["output_dir"]) / "final"))

    hub_cfg = cfg.get("hub", {})
    if hub_cfg.get("push_to_hub") and not args.no_push:
        trainer.push_to_hub(repo_id=hub_cfg.get("repo_id"), private=hub_cfg.get("private", True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
