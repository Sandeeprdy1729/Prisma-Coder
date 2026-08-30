<div align="center">

# PrismaCoder

**The first open-source Text-to-Prisma LLM project.**

`PrismaForge` (execution-verified dataset) · `PrismaBench` (1,200-task benchmark) · `PrismaCoder-3B` (Qwen2.5-Coder-3B fine-tune)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://img.shields.io/badge/CI-github--actions-blue)](.github/workflows/ci.yml)
[![Benchmark](https://img.shields.io/badge/Benchmark-PrismaBench%20v1.0-8b5cf6)](benchmark/)

</div>

---

PrismaCoder turns natural language into **schema.prisma (PSL)** and **Prisma Client TypeScript** that *compiles, validates, and executes*. It is the complete, reproducible implementation of the research program: a dataset pipeline (PrismaForge), an execution-based benchmark (PrismaBench), the training recipe (CPT -> SFT -> DPO), an evaluation harness, inference services, and the full release tooling.

There was, before this project, **no academic Text-to-Prisma benchmark, dataset, or fine-tuned model** - the task has to track paired relation fields, typed include/select shapes, nested writes, transaction semantics, and PSL's attribute grammar, which is a sharper test of typed composition than SQL.

## Repository map (every folder explained)

| Folder | What lives there |
|---|---|
| `prismacoder/` | The core Python package. `schema.py` (canonical record contract), `psl.py` (PSL parser/renderer), `crawler.py` + `github_api.py` + `sources.py` (mined track), `extractor.py` (schema + client call-site census), `license_filter.py` / `secrets.py` / `pii.py` / `quality.py` / `version_norm.py` / `dedup.py` (cleaning), `synthetic/` (grammar sampler, perturbation engine, reference composers, bug injector, augmentation), `verification/` (the V1-V5 gate chain + static fallback), `eval_metrics/` (exact-match, pass@k, BLEU, CodeBLEU, tree edit distance, Wilson CIs), `inference/` (pipeline + repair loop) |
| `datasets/` | The PrismaForge build workspace: `raw/` crawl snapshots, `processed/` extractions + clean corpus, `synthetic/` generated pools, `verified/` post-gate pools, `splits/` database-level splits, `scripts/01-08` the pipeline stages, plus the finalized release set (`train.jsonl`, `metadata.json`, `schema.json`, ...) |
| `benchmark/` | PrismaBench: `generator/` (T1-T12 task specs + seeded builder), `public/` tasks + provenance/canaries, `private/` reference keys (maintainers), `evaluator/` (scoring harness + leaderboard), `leaderboard/` results |
| `training/` | `configs/` (QLoRA/LoRA/full/DPO/CPT YAMLs, accelerate + DeepSpeed configs), `scripts/` (train_sft, train_dpo, train_cpt, merge_adapter, export_gguf, push_to_hub), `trainer/` (data formatting, collators, gate-aligned callbacks) |
| `kaggle/` | The production Kaggle notebook: end-to-end QLoRA (install -> data -> train -> eval -> merge -> push -> GGUF) |
| `notebooks/` | `02_inference.ipynb` (NL -> validated artifact with repair), `03_evaluation.ipynb` (PrismaBench + error analysis + charts) |
| `evaluation/` | Cross-benchmark adapters (Spider, BIRD), E1-E10 error analysis, charts (matplotlib), paired model comparison (McNemar + bootstrap + Holm) |
| `inference/` | `server/` (FastAPI REST + SSE streaming), `gradio_app.py` (UI), `hf_space/` (self-contained Space app) |
| `docker/` | Inference image (Py+Node, gates enabled), verifier sandbox image, compose stack (api + postgres + verifier) |
| `docs/` | Architecture, base-model selection, model/dataset/benchmark cards, reproducibility, release checklist, citation |
| `tests/` | unit / dataset / benchmark / training / verification suites (87 tests; CI runs them without GPU or network) |
| `utils/` | HF upload pipeline, environment checker |

## Quickstart

```bash
pip install -e ".[dev,eval]"
python utils/env_check.py          # toolchain report

# 1) Build PrismaForge end-to-end (or: SMOKE=1 bash datasets/scripts/run_phase1.sh)
bash datasets/scripts/run_phase1.sh

# 2) Build PrismaBench (1,200 tasks, deterministic)
python benchmark/generator/generate_benchmark.py --scale 1.0 --seed 20260830

# 3) Train (QLoRA; see training/README.md for the S0-S3 stage plan)
python training/scripts/train_sft.py --config training/configs/qlora_3b.yaml

# 4) Evaluate
python benchmark/evaluator/evaluate.py --submission sub.jsonl --model my-model --full-gates
python benchmark/evaluator/leaderboard.py

# 5) Serve
docker compose -f docker/docker-compose.yml up -d
```

## The verification chain (the project's core idea)

Every dataset record and every benchmark score flows through the gates:

```
V1 prisma validate -> V2 prisma generate -> V3 tsc --noEmit -> V4 db push/seed -> V5 vitest
      E1/E2                  E3                     E4                E5                E6-E9 (+probes)
```

The same gates are reused three ways: as the **data filter** (only gate-green records train),
as the **DPO/GRPO reward** (gate differentials label preference pairs), and as the
**repair-loop oracle** at inference time. Without Node, a static
symbol-grounding validator (V0) keeps the pipeline runnable everywhere and marks
records `unverified` pending a full-gate re-run.

## Provenance, licensing, contamination

- License gate: permissive-only (MIT / Apache-2.0 / BSD / ISC); GPL/AGPL excluded by policy.
- Secret scan + PII scrub on every artifact; HIGH-severity findings drop the record.
- Three-granularity dedup: exact hash, MinHash-LSH (Jaccard 0.80), model-graph isomorphism.
- Contamination firewall: benchmark provenance hashes are excluded at split time; every
  benchmark task embeds a canary string; CI asserts canaries never appear in training splits.
- Two of the 12 domains are withheld from all public training data (OOD reporting).

## Documentation

| Doc | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | system walkthrough + dataflow diagrams |
| [docs/BASE_MODEL.md](docs/BASE_MODEL.md) | why Qwen2.5-Coder-3B (comparison table) |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | model card for PrismaCoder-3B |
| [docs/DATASET_CARD.md](docs/DATASET_CARD.md) | PrismaForge card (schema, splits, audit) |
| [docs/BENCHMARK_CARD.md](docs/BENCHMARK_CARD.md) | PrismaBench card (tasks, scoring, governance, submission guide) |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | pinned toolchain, seeds, exact build commands |
| [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) | the release gate |
| [docs/CITATION.md](docs/CITATION.md) | BibTeX |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs lint + 87 tests + an end-to-end
pipeline smoke on every PR; the node-gates job additionally exercises the real
V1-V5 chain on a generated sample.

## License

Apache-2.0. Dataset artifacts additionally respect the per-file licenses of
mined sources (permissive-only, enforced by the license gate).
