#!/usr/bin/env python3
"""DPO training (TRL DPOTrainer) - stage S2 (report Table 12.1).

    python training/scripts/train_dpo.py --config training/configs/dpo_qlora.yaml

Trains on verifier-labeled pairs (chosen = gate-green, rejected = strongest
gate-failing candidate). beta 0.05, LR 5e-7, ref = S1 model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


from training.scripts.train_sft import load_config, validate_config  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PrismaCoder DPO trainer (S2)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-pairs", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    problems = validate_config(cfg)
    beta = cfg.get("training", {}).get("beta")
    if beta is None or abs(float(beta) - 0.05) > 1e-9:
        problems.append(f"DPO beta must be 0.05 per Table 12.1 (got {beta})")
    if problems:
        for p in problems:
            print(f"[config] {p}")
        return 2

    pairs_file = cfg["data"]["train_file"]
    if not Path(pairs_file).exists():
        print(f"DPO pairs missing: {pairs_file} - produced by datasets stage 08")
        return 2
    if args.dry_run:
        from prismacoder.io_utils import count_lines

        print(f"[dry-run] config OK; DPO pairs: {count_lines(pairs_file)}")
        return 0

    # ---- heavy imports ----
    import torch
    from peft import LoraConfig, prepare_model_for_kbit_training  # noqa: F401
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    from datasets import Dataset
    from prismacoder.constants import SYSTEM_PROMPT
    from prismacoder.io_utils import read_jsonl
    from training.trainer.data import to_dpo_example

    model_cfg, tok_cfg, data_cfg, train_cfg = (cfg["model"], cfg["tokenizer"], cfg["data"], cfg["training"])
    tokenizer = AutoTokenizer.from_pretrained(tok_cfg.get("name_or_path", model_cfg["name_or_path"]))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if model_cfg.get("torch_dtype") == "bfloat16" else torch.float16
    model_kwargs: dict = {"trust_remote_code": True, "torch_dtype": dtype}
    quant = model_cfg.get("quantization")
    if quant and quant.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
    if quant:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = cfg.get("peft")
    peft_config = None
    if peft_cfg:
        peft_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=peft_cfg.get("r", 16),
            lora_alpha=peft_cfg.get("lora_alpha", 32),
            lora_dropout=peft_cfg.get("lora_dropout", 0.05),
            target_modules=peft_cfg.get("target_modules"),
            bias="none",
        )

    rows = read_jsonl(pairs_file)
    if args.max_pairs:
        rows = rows[: args.max_pairs]
    examples = [to_dpo_example(r, tokenizer, SYSTEM_PROMPT) for r in rows]
    train_ds = Dataset.from_list(examples)

    dpo_cfg = DPOConfig(
        output_dir=cfg["output_dir"],
        run_name=cfg.get("run_name"),
        seed=cfg.get("seed", 20260830),
        num_train_epochs=train_cfg.get("num_train_epochs", 1),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 32),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
        beta=float(train_cfg.get("beta", 0.05)),
        loss_type=train_cfg.get("loss_type", "sigmoid"),
        max_prompt_length=data_cfg.get("max_prompt_length", 2048),
        max_completion_length=data_cfg.get("max_completion_length", 2048),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="no",
        save_strategy=train_cfg.get("save_strategy", "steps"),
        save_steps=train_cfg.get("save_steps", 400),
        save_total_limit=train_cfg.get("save_total_limit", 3),
        report_to=train_cfg.get("report_to", ["tensorboard"]),
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=train_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(str(Path(cfg["output_dir"]) / "final"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
