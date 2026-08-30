# Training (Phase 3)

Stage plan from the research report (Table 12.1):

| Stage | Config | Data | Objective | Key HP |
|---|---|---|---|---|
| S0 CPT | `configs/cpt.yaml` | 8-12B tok Prisma-dense TS+PSL+docs | next-token | LR 8e-5 peak, bs 4M tok, seq 8k |
| S1 SFT | `configs/qlora_3b.yaml` (primary) / `lora_3b.yaml` / `full_ft.yaml` | 120k verified triples + 10k gold | cross-entropy on artifacts | LR 1e-4 (adapter) / 1.5e-5 (full); 3 ep; bs 128; packing; NEFTune 5 |
| S2 DPO | `configs/dpo_qlora.yaml` | 30k verifier-labeled pairs | DPO | beta 0.05; ref = S1; LR 5e-7; 1-2 ep; bs 64 |
| S3 GRPO | optional, roadmap | T1-T12 rollouts | GRPO group 8 | LR 1e-6; KL 0.02; R = 0.25·V1+0.15·V2+0.20·V3+0.40·V5 |

All schedules: cosine with 3% warmup, bf16.

## Quickstart

```bash
pip install -e ".[training]"

# 0) sanity: validate configs against the dataset
python training/scripts/train_sft.py --config training/configs/qlora_3b.yaml --dry-run

# 1) SFT - QLoRA on a single 24GB GPU (or a Kaggle T4 pair; see kaggle/)
accelerate launch --config_file training/configs/accelerate/single_gpu.yaml \
    training/scripts/train_sft.py --config training/configs/qlora_3b.yaml

# 2) merge adapter
python training/scripts/merge_adapter.py \
    --base Qwen/Qwen2.5-Coder-3B-Instruct \
    --adapter runs/prismacoder-3b-qlora-sft/final \
    --out runs/prismacoder-3b-qlora-sft/merged

# 3) DPO on verifier-labeled pairs
python training/scripts/train_dpo.py --config training/configs/dpo_qlora.yaml

# 4) release
python training/scripts/push_to_hub.py --model runs/prismacoder-3b-qlora-sft/merged \
    --repo prismacoder/PrismaCoder-3B --dataset datasets --dataset-repo prismacoder/PrismaForge
python training/scripts/export_gguf.py --model runs/prismacoder-3b-qlora-sft/merged \
    --out exports/PrismaCoder-3B.gguf --quant q8_0
```

## DeepSpeed / multi-GPU

- `configs/deepspeed/zero2.json` - 8x A100 SFT (ZeRO-2, recommended)
- `configs/deepspeed/zero3.json` - full fine-tuning on tighter clusters
- `configs/accelerate/deepspeed_zero2.yaml` - accelerate launcher profile

## Notes

- `mixture_soft_ordering` (report 12.3): epoch 1 oversamples L1 (60%), epoch 3
  oversamples L3 + repair - implemented in `training/trainer/data.py:SoftOrderingMixer`.
- Sanity generation during training runs the static symbol-grounding check so
  the gate profile is visible in the training log (no Node needed mid-run).
- Full-gate evaluation of checkpoints happens post-hoc via
  `benchmark/evaluator/evaluate.py --full-gates`.
