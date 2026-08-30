# Dataset Card: PrismaForge v1.0

## Summary

PrismaForge is an open, execution-verified corpus for Text-to-Prisma:
**120,000 SFT instruction triples** (target composition, Table 9.1 of the
report), **30,000 verifier-labeled DPO pairs**, a 1,500-record public dev
set, 2,000 gold human-verified anchors, and a 600-record private held-out
set. Every training record passed (or is explicitly marked
`unverified` pending) the four-gate chain: prisma validate -> prisma
generate -> tsc -> database execution (+ vitest where tests exist).

## Tracks

| Track | Source | Verification |
|---|---|---|
| mined | GitHub repos containing schema.prisma (seed list + code-search expansion), call-site census via deterministic extraction | license gate, secret/PII scan, V-gates on schemas |
| official | prisma org, create-t3-app, Redwood, Blitz, trpc, Amplication, ZenStack, Wasp | golden seeds / eval anchors, not diluted into bulk SFT |
| synthetic | 12-domain grammar sampler + perturbation engine; mechanical reference composers | full four-gate chain |
| docs | prisma/docs + website (Apache-2.0) | CPT only |
| augmented | persona paraphrase, entity rename, schema shuffle, distractor insertion | re-verified, tagged |

## Record schema

Canonical contract: `datasets/schema.json` (JSON Schema draft 2020-12),
implemented in `prismacoder/schema.py`. One record = (source, domain, task
[type/category/difficulty/instruction/context/reference/persona/tests],
verification, difficulty_scores, style_tags, meta). DPO pairs additionally
carry `preference {chosen, rejected, basis}`; ~10% of SFT records carry a
mechanical `reasoning.plan_span`.

## Composition controls

- **Task families** (Table 9.1): schema tasks 35k / client CRUD 25k /
  nested-agg-txn 30k / repo-aware+repair 20k / docs-misc 10k.
- **Difficulty** is mechanical: model count, relation depth, nesting, ops,
  NL ambiguity (five axes recorded per record).
- **Domains**: 12 archetypes; `health` and `multi_tenant_hr` are withheld
  from public training (OOD reporting).
- **Personas**: junior / pm / expert phrasings varied per epoch (soft-ordering
  mixture, report 12.3).

## Cleaning pipeline (stage 03)

1. License gate - permissive only (MIT, Apache-2.0, BSD-2/3, ISC); copyleft
   and unlicensed content excluded by policy.
2. Secret scan - gitleaks-equivalent rules; HIGH findings drop the record,
   MEDIUM findings redact in place.
3. PII scrub - emails/phones/coords/IPs replaced with structural
   placeholders (example-domain emails kept).
4. Version normalization - legacy syntax normalized to Prisma v7 dialect;
   original version tagged.
5. Quality filter - PSL structural sanity (paired relations, keys,
   datasource/generator), TS brace balance, TODO-stub rejection.
6. Dedup - exact hash; MinHash-LSH at 0.80 Jaccard; model-graph isomorphism
   signatures (corpus-wide, not per-track).

## Splits and contamination

- Splits are **database-level**: all records from one source schema live in
  exactly one split (`datasets/scripts/07_split_dataset.py`).
- **Firewall**: records whose provenance hash appears in
  `benchmark/public/benchmark_provenance.json` are dropped from training.
- **Canaries**: benchmark tasks embed `PRISMABENCH-CANARY-*`; CI asserts
  they never appear in training splits.
- **Temporal hygiene**: post-cutoff repos are tagged and routed to the
  private-refresh pipeline.

## Human audit

Gold records follow the report 9.9 rubric (request fidelity, schema realism,
reference idiomaticity, test strength, metadata accuracy; 1-4 per dimension,
release requires mean >= 3.4 and no dimension < 3, kappa target 0.75+).
The rubric ships with the dataset release.

## Maintenance

- Data: `prismacoder` org on HuggingFace (`PrismaForge`).
- Quarterly refresh: re-crawl -> same gates -> additive v1.x releases;
  breaking schema-format changes are v2.0.
- Build: reproducible from `datasets/scripts/run_phase1.sh` with pinned
  seeds (default 20260830); `metadata.json` in every release carries the
  full manifest.
- Contact / erroneous-content reports: repository issues (see
  docs/RELEASE_CHECKLIST.md for the takedown/rebuild protocol).
