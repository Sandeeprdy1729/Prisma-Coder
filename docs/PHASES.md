# Implementation Phases

The repository is organized into the five delivery phases requested for the
project. Each phase is independently runnable; phases map to the directory
layout as follows.

## Phase 1 - Dataset collection & verification (PrismaForge)

**Entry**: `bash datasets/scripts/run_phase1.sh` (or `SMOKE=1 ...`)

| Step | Implementation |
|---|---|
| Crawl | `prismacoder/crawler.py`, `github_api.py`, `sources.py` -> `datasets/scripts/01_crawl_github.py` |
| Extract | `prismacoder/extractor.py` -> `02_extract_corpus.py` (schema.prisma, migrations, CRUD/transactions/aggregations/groupBy/raw-SQL, repository/service/API layers) |
| Clean | `license_filter.py`, `secrets.py`, `pii.py`, `quality.py`, `version_norm.py`, `dedup.py` -> `03_clean_corpus.py` |
| Synthetic | `synthetic/` (vocab, schema_gen, instruction_gen, crud_gen, relation_gen, migration_gen, task_variants, augment, generator) -> `04_generate_synthetic.py`, `05_augment.py` |
| Verify | `verification/` (sandbox, prisma_gate, ts_gate, db_gate, test_gate, verifier) -> `06_verify_all.py` |
| Finalize | `07_split_dataset.py`, `08_finalize_dataset.py` -> `datasets/{train,validation,test,benchmark,dpo_pairs}.jsonl` + `metadata.json` + `schema.json` |

## Phase 2 - Benchmark generation (PrismaBench)

**Entry**: `python benchmark/generator/generate_benchmark.py --scale 1.0`

| Step | Implementation |
|---|---|
| Task spec | `benchmark/generator/task_specs.py` (T1-T12 x counts = 1,200; L1-L3 rules; gates per category) |
| Build | `benchmark/generator/generate_benchmark.py` (public/private split, canaries, provenance) |
| Evaluate | `benchmark/evaluator/evaluate.py`, `scoring` via `prismacoder.eval_metrics` + `constants.COMPOSITE_WEIGHTS` |
| Leaderboard | `benchmark/evaluator/leaderboard.py` -> `benchmark/leaderboard/LEADERBOARD.md` |

## Phase 3 - Training pipeline (QLoRA/LoRA + Kaggle notebook)

| Step | Implementation |
|---|---|
| Configs | `training/configs/{qlora_3b,lora_3b,full_ft,dpo_qlora,cpt}.yaml`, `accelerate/`, `deepspeed/` |
| Trainers | `training/scripts/train_sft.py`, `train_dpo.py`, `train_cpt.py`; `training/trainer/{data,collators,callbacks}.py` |
| Ship | `merge_adapter.py`, `export_gguf.py`, `push_to_hub.py` |
| Notebook | `kaggle/prismacoder_qlora_kaggle.ipynb` (install -> data -> 4-bit -> QLoRA -> train -> static eval -> checkpoints -> merge -> push -> GGUF) |

## Phase 4 - Evaluation harness

| Step | Implementation |
|---|---|
| Metrics | `prismacoder/eval_metrics/` (exact_match, pass_at_k, bleu, code_bleu, tree_edit_distance, registry + Wilson CI) |
| Benchmarks | `benchmark/evaluator/evaluate.py`; `evaluation/spider_adapter.py`, `evaluation/bird_adapter.py` |
| Analysis | `evaluation/error_analysis.py` (E1-E10 Pareto), `evaluation/plots.py`, `evaluation/compare_models.py` (McNemar/bootstrap/Holm) |
| Notebook | `notebooks/03_evaluation.ipynb` |

## Phase 5 - Inference, deployment, and Hugging Face release

| Step | Implementation |
|---|---|
| Core | `prismacoder/inference/pipeline.py` + `repair.py` |
| Serving | `inference/server/app.py` + `schemas.py` (FastAPI REST/SSE), `inference/gradio_app.py`, `inference/hf_space/app.py` |
| Docker | `docker/Dockerfile`, `Dockerfile.verifier`, `docker-compose.yml`, `entrypoint.sh`, `verifier/worker.py` |
| Release | `utils/hf_upload.py`, `docs/MODEL_CARD.md`, `docs/DATASET_CARD.md`, `docs/BENCHMARK_CARD.md`, `docs/RELEASE_CHECKLIST.md`, `docs/CITATION.md`, `.github/workflows/release.yml` |
| CI | `.github/workflows/ci.yml` (lint, 87 tests, pipeline smoke, canary firewall, node-gates) |
