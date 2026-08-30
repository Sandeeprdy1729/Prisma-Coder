# Release checklist (v1.0)

## Dataset (PrismaForge)

- [ ] Full crawl executed with `GITHUB_TOKEN` (mined track >= 15k validating schemas target)
- [ ] Stage 03 clean report reviewed: drops attributed (license/secrets/PII/quality/dedup)
- [ ] Full-gate verification sweep re-run with Node (no `unverified` records left in train)
- [ ] Database-level split manifest checked (no database spans two splits)
- [ ] Contamination firewall applied; canary check passes in CI
- [ ] `metadata.json` complete (mixture, domains, difficulties, seeds, invalid-skipped)
- [ ] Gold-annotation rubric audit: mean >= 3.4, no dimension < 3, kappa >= 0.75
- [ ] 2 OOD domains absent from all public files (grep-verified)
- [ ] `DATASET_CARD.md` + `schema.json` uploaded with the release

## Benchmark (PrismaBench)

- [ ] Generator run at scale 1.0; Table 10.1 composition assertion passed
- [ ] Private keys committed only to the private storage (never the public repo)
- [ ] Provenance hashes + canaries published in `benchmark_provenance.json`
- [ ] Evaluator dry-run on dummy submission; leaderboard renders
- [ ] Harness flake rate < 1% across 5 repeated runs on a 100-task subset
- [ ] `BENCHMARK_CARD.md` submitted with the first public leaderboard

## Model (PrismaCoder-3B)

- [ ] S1 SFT converged (eval loss plateau; sanity gate profile logged)
- [ ] S2 DPO gate-profile regression check (no easy-task collapse)
- [ ] PrismaBench full-gate numbers: per-category/per-level/per-gate + Wilson CIs + E1-E10
- [ ] pass@1 (repair loop) on L1-L2 >= 60% for the O2 goal, else documented shortfall
- [ ] Efficiency columns reported (latency, memory, tokens) at pinned serving config
- [ ] Model card filled with measured numbers (not targets)
- [ ] Adapter merged; safetensors + GGUF q8_0 artifacts uploaded
- [ ] License check: base model + training data licenses propagate cleanly (Apache-2.0)

## Infrastructure

- [ ] CI green on the release commit (lint, 87 tests, pipeline smoke, canary check)
- [ ] Docker images build; verifier `/health` smoke passes
- [ ] `pip install prismacoder` works from a clean env (sdist + wheel)
- [ ] PyPI publish (OIDC) + GitHub release with changelog
- [ ] HuggingFace dataset/model/benchmark repos live with cards
- [ ] Docs updated (REPRODUCIBILITY pins, ARCHITECTURE if changed)

## Governance

- [ ] Erroneous-content takedown protocol link published (dataset card)
- [ ] Contact channel for contamination reports open
- [ ] Private split rotation date recorded (annual)
