# Model Card: PrismaCoder-3B

## Model details

- **Model**: PrismaCoder-3B (QLoRA SFT on Qwen2.5-Coder-3B-Instruct; DPO variant noted in revision tags)
- **Version**: v1.0.0
- **Developed by**: PrismaCoder Team
- **License**: Apache-2.0 (base model: Qwen2.5-Coder-3B-Instruct, Apache-2.0)
- **Architecture**: Qwen2.5 dense decoder, 3.09B params, 32k context; LoRA r=16 alpha=32 on q/k/v/o + gate/up/down projections, merged at release
- **Languages**: PSL (Prisma Schema Language), TypeScript (Prisma Client), English
- **Release format**: HF safetensors (bf16) + GGUF (q8_0) export
- **Training date**: 2026

## Intended use

**Primary**: generating `schema.prisma` and Prisma Client TypeScript from
natural-language requests, optionally grounded on an existing schema
provided in the prompt. Ideal operating point: single-artifact generation
with the validate-and-repair loop (see `prismacoder/inference/repair.py`),
which is how all headline metrics are defined.

**In scope**: CRUD, complex reads (include/select/pagination), nested writes,
interactive transactions, aggregations/groupBy, schema edits, migration
plans with destructive-risk labels, repo-aware completion, bug fixing,
performance-safe rewrites (N+1 avoidance), security-hardened raw-SQL
replacement.

**Out of scope**: general-purpose coding beyond the Prisma stack;
non-TypeScript clients; MongoDB provider idioms are under-represented;
production migration execution without human review.

## Training data

PrismaForge v1.0 - 120k verified instruction triples (four-gate chain:
validate/generate/tsc/execute/vitest) + verifier-labeled DPO pairs.
Domains: 12 archetypes, 2 withheld from training (OOD eval). License
permissive-only; secret/PII scrubbed; three-granularity dedup. Full card:
`DATASET_CARD.md`.

## Hyperparameters (headline)

SFT: 3 epochs, LR 1e-4 (adapter) cosine with 3% warmup, effective batch 128,
packing on, NEFTune alpha 5, bf16. DPO: beta 0.05, LR 5e-7, ref = SFT model.
The exact stage plan (S0 CPT -> S1 SFT -> S2 DPO) is Table 12.1 of the report
and `training/configs/*`.

## Evaluation

**PrismaBench v1.0** (1,200 tasks, T1-T12, L1-L3, execution-weighted
composite; gate-chain pass@1 primary). Report per-category, per-level,
per-gate numbers with Wilson 95% CIs; pass@k curves; E1-E10 taxonomy
distribution; latency/memory mandatory columns. Program goal (report O2):
>= 60% pass@1 with repair on L1-L2. Current results: see
`benchmark/leaderboard/LEADERBOARD.md`.

## Limitations and risks

- **Schema grounding is not a guarantee**: unprovided symbols may still be
  hallucinated under paraphrase pressure; the LightValidator catches most
  grounding failures, execution gates catch more.
- **Wrong-but-compiling semantics** (E6) requires tests or human review;
  do not ship generated migrations unreviewed - destructive-risk labels are
  advisory, not a safety mechanism.
- **Post-cutoff Prisma features** (post Nov-2025) may be missed; private
  rotating split measures this drift annually.
- **Bias**: training corpora are GitHub-mined and synthetic over 12 domain
  archetypes; naming/style reflect TypeScript community conventions.

## Usage

```python
from prismacoder.inference.pipeline import PrismaCoderPipeline
pipe = PrismaCoderPipeline.from_local("prismacoder/PrismaCoder-3B")
result = pipe.run("List the 20 most recent orders with users, newest first, "
                  "excluding cancelled.", schema=SCHEMA)
print(result.artifact, result.status)
```

## Citation

See `docs/CITATION.md`.
