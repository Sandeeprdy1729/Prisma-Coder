# Benchmark Card: PrismaBench v1.0

## Motivation

Text-to-SQL benchmarks do not measure ORM competence: Text-to-Prisma requires
paired relation fields, typed include/select shapes, nested writes with
transaction semantics, and PSL's attribute grammar - a sharper test of typed
composition. Prior to PrismaBench there was **no Text-to-Prisma benchmark
of any kind**. PrismaBench v1.0 provides 1,200 execution-scored tasks with
per-gate diagnostics, contamination controls, and a maintained leaderboard.

## Task composition (report Table 10.1)

| Task | Description | Artifact | Count | Levels | Scoring gates |
|---|---|---|---:|---|---|
| T1 | Schema from NL (green-field) | PSL | 120 | L1-L3 | V1, V2 (+migrate) |
| T2 | Schema edit from change request | PSL diff | 120 | L1-L3 | V1, V2 + semantic diff |
| T3 | Migration plan + destructive-risk label | PSL+SQL+risk | 90 | L2-L3 | V1, V2, migration dry-run |
| T4 | Client CRUD | TS | 150 | L1-L2 | V3, V4, V5 |
| T5 | Complex reads (include/select/pagination) | TS | 140 | L1-L3 | V3, V4, V5 |
| T6 | Nested writes + interactive transactions | TS | 120 | L2-L3 | V3, V4, V5 (+isolation) |
| T7 | Aggregations (aggregate/groupBy/count) | TS | 100 | L2-L3 | V3, V4, V5 |
| T8 | Repo-aware completion | TS multi-file | 120 | L2-L3 | V3, V4, V5 |
| T9 | Bug fix from failing code + diagnostic | TS/PSL | 100 | L1-L3 | V-chain to green |
| T10 | Cross-file reasoning (data+API+types) | TS multi-file | 60 | L3 | V3, V4, V5 |
| T11 | Performance-aware queries (N+1 probes) | TS+index | 60 | L2-L3 | V3-V5 + query-count probe |
| T12 | Security-hardened (parameterization, tenant scoping) | TS | 20 | L3 | V3-V5 + security probes |

Difficulty is mechanical (report 10.3): L1 <= 6 models / depth 1-2; L2 6-10
models / depth 2-3; L3 10-14 models / depth up to 4 + distractors or
destructive operations. 12 domains, each 8-12% of tasks, every domain present
at every level; 2 domains are withheld from public data entirely (OOD subset).

## Scoring

- **pass@1** (primary): gate-chain verdict; unbiased pass@k (Chen et al.)
  over submitted samples.
- **Composite** = 0.20*compile + 0.60*execute + 0.10*style + 0.10*efficiency
  - execution-dominated by design (report 10.1).
- **Diagnostics**: per-category, per-level, per-gate pass rates with Wilson
  95% CIs; failures auto-classified into E1-E10 (Appendix D).
- **Efficiency columns (mandatory)**: latency (TTFT + end-to-end at pinned
  serving config), peak memory, token cost per task.
- **Robustness**: score delta under paraphrase / entity rename / schema
  shuffle / distractor insertion (Dr.Spider-style families).
- **Statistics**: McNemar + paired bootstrap for head-to-head; Holm-Bonferroni
  across grids; minimum detectable effect ~3.5-4 pts at n=1,200.

## Submission guide

1. Build/load the public tasks (`benchmark/public/prismabench_public.jsonl`).
2. Produce a submission JSONL:
   `{"bench_id": ..., "output": "...", "latency_ms": ..., "memory_mb": ...}`
   (or `samples: [...]` for pass@k).
3. `python benchmark/evaluator/evaluate.py --submission ... --model ... \
   --full-gates --out benchmark/leaderboard/<name>_results.json`
   (full gates require Node >= 18.18; static-mode entries are ranked
   separately and never conflated with gate-green numbers).
4. `python benchmark/evaluator/leaderboard.py` regenerates
   `LEADERBOARD.md`.

Single-shot-plus-repair is the default track; agentic multi-call tracks are
opt-in and reported separately (report O5: cost honesty).

## Governance

- **Contamination**: canary strings in every task; n-gram decontamination
  audit against public corpora; provenance hashes published for the
  training-side firewall; private 50% rotating annually with post-cutoff
  schemas.
- **OOD**: 2 withheld domains reported separately from public domains.
- **Reproducibility**: pinned toolchain (see docs/REPRODUCIBILITY.md),
  containerized harness, deterministic seeds, raw per-task JSON shipped with
  every released result.
- **Flake budget**: harness flake rate < 1% or the release is blocked.
- **Disclosure**: coordinated-disclosure process for harness vulnerabilities.

## Limitations

- Execution lanes cover sqlite by default and postgres when provisioned;
  mysql/sqlserver lanes are skipped (reported as skipped, not failed).
- Style/efficiency components are heuristic (similarity + query-count
  probes); execution remains 60% of the composite.
- T12 security probes are spot-probes by construction - passing T12 is not a
  security guarantee for downstream deployments.
