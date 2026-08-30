# Reproducibility

Every headline number in the project must be rebuildable from this repository
with pinned inputs. This file is the pin list.

## Toolchain pins

| Tool | Version | Where used |
|---|---|---|
| Node.js | >= 18.18 (CI: 20) | V1-V5 gates, sandbox |
| prisma CLI | 6.2.1 (`prismacoder/verification/sandbox.py`) | V1/V2/V4 |
| TypeScript | 5.6.3 | V3 |
| vitest | 2.1.8 | V5 |
| tsx | 4.19.2 | V4 seed/migration runner |
| Python | 3.10-3.12 (CI: 3.11) | everything |
| torch / transformers / trl / peft / bitsandbytes | `pyproject.toml [training]` minimums; Kaggle notebook pins exact versions (4.46.3 / 0.13.2 / 0.12.2 / 0.44.1) | training |
| llama.cpp | pinned commit in `training/scripts/export_gguf.py` | GGUF export |
| Official Spider & BIRD eval scripts | pinned at their repos; adapter emits their native formats (`evaluation/*_adapter.py`) | cross-benchmarks |

## Seeds

- Everything defaults to `DEFAULT_SEED = 20260830` (`prismacoder/constants.py`).
- The synthetic generator is deterministic per seed: same seed => identical
  corpus (asserted by `tests/dataset_tests/test_dataset.py`).
- PrismaBench is deterministic per seed; the generator asserts the exact
  Table 10.1 composition at scale 1.0.

## Canonical build commands

```bash
# PrismaForge (full / smoke)
bash datasets/scripts/run_phase1.sh
SMOKE=1 bash datasets/scripts/run_phase1.sh

# PrismaBench
python benchmark/generator/generate_benchmark.py --scale 1.0 --seed 20260830

# Training (S1 QLoRA primary; see training/README.md for S0/S2)
python training/scripts/train_sft.py --config training/configs/qlora_3b.yaml

# Evaluation
python benchmark/evaluator/evaluate.py --tasks benchmark/public/prismabench_public.jsonl \
    --private-keys benchmark/private/private_keys.jsonl --submission sub.jsonl \
    --model <name> --full-gates --out benchmark/leaderboard/<name>_results.json
python benchmark/evaluator/leaderboard.py
python evaluation/error_analysis.py --results benchmark/leaderboard/<name>_results.json
python evaluation/compare_models.py --baseline a.json --treatment b.json
```

## Environment manifest

Every released result ships: raw per-task JSON, seeds, git commit,
`pip freeze` + `node --version` output, and the verifier version
(`verification.verifier_version` in each record). The evaluate.py
`results.json` carries `mode` (full_gates vs static) - static-mode numbers
must be labeled as such in any publication.

## Cross-benchmark protocol

Spider (dev) and BIRD (dev) predictions are produced with the adapters and
scored with each benchmark's **official** evaluation scripts (execution
accuracy semantics are theirs, not ours). Pin the eval-script commit in your
release notes; our adapters only standardize prompting + prediction format.
