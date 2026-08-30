# `datasets/` - PrismaForge build workspace

Every file here (except scripts) is a **build artifact** - reproducible from
`datasets/scripts/run_phase1.sh` with pinned seeds. Nothing needs to be
committed except the finalized release set.

## Directory map

| Path              | Contents                                                                 | Rebuilt by |
|-------------------|--------------------------------------------------------------------------|------------|
| `raw/`            | Immutable crawl snapshots per repo (`meta.json`, `tree.json`, `files/`)   | stage 01   |
| `processed/`      | Extraction records + clean corpus + cleaning report                       | 02, 03     |
| `synthetic/`      | Synthetic pool, augmented pool, mined pairs + build reports               | 04, 05     |
| `verified/`       | Pools after the V1-V5 verification sweep (or static `unverified` pass)    | 06         |
| `splits/`         | Database-level train/validation/test split + manifest                     | 07         |
| `scripts/`        | The pipeline itself (committed; the only source-of-truth code here)       | -          |

## Final release set (root of `datasets/`, emitted by stage 08)

- `train.jsonl`, `validation.jsonl`, `test.jsonl` - canonical records (see `schema.json`)
- `benchmark.jsonl` - dev-eval rows in the public shape (reference withheld)
- `dpo_pairs.jsonl` - verifier-labeled preference pairs (S2 DPO stage)
- `metadata.json` - build manifest: seeds, mixture, domains, difficulties
- `schema.json` - JSON Schema (draft 2020-12) of the canonical record

## Pipeline

```bash
# full build (needs GITHUB_TOKEN + Node >= 18.18)
bash datasets/scripts/run_phase1.sh

# smoke run (no network, no node; ~600 records, static verification)
SMOKE=1 bash datasets/scripts/run_phase1.sh
```

Stage contracts (each stage reads only prior-stage outputs):

```
01 crawl        raw/{repo}/...
02 extract      processed/extraction_records.jsonl
03 clean        processed/clean_records.jsonl          license+secrets+PII+v7+quality+dedup
04 synthetic    synthetic/synthetic_pool.jsonl          seeded grammar sampler
05 augment      synthetic/augmented_pool.jsonl, mined_pairs.jsonl
06 verify       verified/*.verified.jsonl               V1-V5 (or V0 static fallback)
07 split        splits/{train,validation,test}.jsonl    database-level, firewall-enforced
08 finalize     train.jsonl ... metadata.json           release set
```

The contamination firewall (stage 07) excludes any record whose provenance hash
appears in `benchmark/public/benchmark_provenance.json`, enforcing the report's
eval-firewall (9.8.1).
