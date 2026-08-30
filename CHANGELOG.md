# Changelog

All notable changes are documented here. Versioning follows the dataset
convention from the research report: v1.x additive, v2.0 breaking.

## [1.0.0] - 2026-08-30

Initial public implementation of the research program.

### Added - Phase 1: PrismaForge (dataset collection & verification)
- `prismacoder/` core package: canonical record schema + JSON Schema, PSL
  parser/renderer, GitHub crawler (seed list + code-search discovery),
  deterministic artifact extractor (CRUD/relations/transactions/aggregations/
  raw-SQL census + service/repository/API layer detection)
- Cleaning: license gate (permissive-only), gitleaks-style secret scan, PII
  scrub, Prisma v7 version normalization, quality filter, three-granularity
  dedup (exact / MinHash-LSH 0.80 / model-graph isomorphism)
- Synthetic engine: 12 domain archetypes, grammar sampler (3-14 models, key
  styles, soft-delete, tenancy, audit), perturbation engine (10-op matrix),
  persona verbalizer + teacher hook, reference composers for T1-T12, bug
  injector (10 patterns, E1-E10 aligned), augmentation (rename/shuffle/
  distractor/paraphrase), DPO preference pairs + reasoning plan-spans
- Verification: V1-V5 gate chain in a pinned Node sandbox (prisma
  validate/generate, tsc, db push + migration dry-run, vitest JSON),
  docker verifier service, LightValidator static fallback, E1-E10 classifier
- Dataset finalize: database-level splits, contamination firewall, release
  set (train/validation/test/benchmark/dpo jsonl + metadata + schema.json)

### Added - Phase 2: PrismaBench (benchmark generation)
- Table 10.1 task specs (T1-T12, counts summing to 1,200), mechanical L1-L3
  difficulty rules, 12 domains (2 OOD-withheld)
- Seeded benchmark builder (public/private split, provenance hashes, canaries)
- Evaluator (gate-based pass@1, unbiased pass@k, CodeBLEU/TED similarity,
  efficiency columns, composite 0.20/0.60/0.10/0.10), Wilson CIs, leaderboard

### Added - Phase 3: Training
- QLoRA/LoRA/full SFT configs + DPO + CPT (Table 12.1 hyperparameters)
- accelerate + DeepSpeed ZeRO-2/3 configs
- TRL SFTTrainer/DPOTrainer scripts, merge/GGUF/push-to-hub tooling
- Kaggle notebook: end-to-end QLoRA with HF-token secrets, hub download or
  deterministic mini-corpus fallback, static gate eval, merge, push, GGUF

### Added - Phase 4: Evaluation harness
- Metrics: exact match, pass@k curve, BLEU, CodeBLEU, tree edit distance,
  Wilson CIs, aggregation (per-category/level/gate + E1-E10)
- Spider/BIRD adapters (official-eval-native prediction formats)
- Error analysis (Pareto of fixes), charts, paired comparison
  (McNemar + bootstrap + Holm-Bonferroni)
- Inference + evaluation notebooks

### Added - Phase 5: Inference, deployment, release
- PrismaCoderPipeline (generate -> validate -> repair loop with gate
  diagnostics), FastAPI server (REST + SSE + validate endpoints), Gradio UI,
  self-contained HF Space app
- Docker: inference image (Py+Node, gates enabled), verifier sandbox image,
  compose stack with postgres lane
- HF upload pipeline (dataset/model/benchmark), docs suite (architecture,
  base-model choice, model/dataset/benchmark cards, reproducibility pins,
  release checklist, citation), CI (lint + 87 tests + pipeline smoke +
  canary firewall + node-gates) and release workflows
