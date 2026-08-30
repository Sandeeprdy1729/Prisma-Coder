# PrismaBench (benchmark/)

PrismaBench v1.0 - the first execution-based benchmark for Text-to-Prisma.
1,200 tasks, 12 categories (T1-T12), 3 mechanical difficulty levels (L1-L3),
12 domains. Design: report Part 10. Full card: `docs/BENCHMARK_CARD.md`.

## Layout

| Path | Contents |
|---|---|
| `generator/` | task spec (Table 10.1), difficulty rules, seeded generator |
| `public/` | `prismabench_public.jsonl` (reference-withheld tasks), `benchmark_provenance.json` (firewall + canaries) |
| `private/` | `private_keys.jsonl` - references + gate requirements. Maintainers only; never commit to public releases. |
| `evaluator/` | `evaluate.py` (scoring harness), `leaderboard.py`, `make_dummy_submission.py` |
| `leaderboard/` | `results.json` per system + generated `LEADERBOARD.md` |

## Build

```bash
# full benchmark (1,200 tasks, deterministic under --seed)
python benchmark/generator/generate_benchmark.py --scale 1.0 --seed 20260830

# smoke (60 tasks)
python benchmark/generator/generate_benchmark.py --scale 0.05
```

The generator asserts the exact Table 10.1 composition at `--scale 1.0`.

## Evaluate a system

Submission format (`jsonl`, one row per task):

```json
{"bench_id": "pb-t4-0001", "output": "```ts\n...```", "latency_ms": 812, "memory_mb": 940}
```

For pass@k, provide `samples: [out1, out2, ...]` instead of `output`.

```bash
# 1) score (static mode without Node; add --full-gates with Node >= 18.18)
python benchmark/evaluator/make_dummy_submission.py \
    --tasks benchmark/public/prismabench_public.jsonl \
    --out my_submission.jsonl
python benchmark/evaluator/evaluate.py \
    --tasks benchmark/public/prismabench_public.jsonl \
    --private-keys benchmark/private/private_keys.jsonl \
    --submission my_submission.jsonl --model my-model \
    --out benchmark/leaderboard/my_model_results.json --full-gates

# 2) leaderboard
python benchmark/evaluator/leaderboard.py --results benchmark/leaderboard \
    --out benchmark/leaderboard/LEADERBOARD.md
```

## Scoring

- **pass@1** (primary): full gate-chain verdict per task; unbiased pass@k via
  Chen et al. closed form when samples are provided.
- **Composite** = 0.20*compile + 0.60*execute + 0.10*style + 0.10*efficiency.
- Per-category, per-level, per-gate breakdowns with Wilson 95% CIs; failures
  auto-classified into the E1-E10 taxonomy.

## Governance

- canary strings are embedded in every task (`PRISMABENCH-CANARY-*`) and checked
  against training builds in CI (`tests/dataset_tests/test_splits.py`).
- the private 50% rotates annually; the 2 withheld OOD domains never appear in
  public training data (enforced by `SyntheticGenerator._allowed_domains`).
- provenance hashes of every task are published in `benchmark_provenance.json`;
  PrismaForge's split stage refuses records matching them.
