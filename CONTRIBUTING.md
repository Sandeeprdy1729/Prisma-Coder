# Contributing

Thanks for helping build the first open-source Text-to-Prisma stack.

## Ground rules

1. **The gates decide.** Dataset records, benchmark tasks, and model outputs
   are only as good as the V1-V5 chain says they are. Never bypass, mock, or
   weaken a gate to make a number look better; skipped gates must be
   reported as skipped.
2. **Mechanics write answers.** Reference solutions are composed
   deterministically from the sampler's internal representation and must pass
   the gates. LLMs may rewrite *requests* (persona/teacher hooks), never
   *references*.
3. **Reproducibility is a contract.** New pipeline behavior must be
   seed-stable and recorded in the build manifest. No non-replicable service
   in the training path.
4. **License hygiene.** Permissive-only sources; the license gate is policy,
   not a suggestion.

## Development setup

```bash
git clone https://github.com/prismacoder/PrismaCoder && cd PrismaCoder
pip install -e ".[dev,eval]"
pre-commit run --all-files   # ruff + black (config in pyproject.toml)
pytest -m "not gpu and not node and not network"   # 87 tests, no GPU needed
```

With Node >= 18.18 installed you can also run the node-gates suite
(`pytest -m node` / the CI `node-gates` job).

## PR checklist

- [ ] `make lint` and `make test` pass locally
- [ ] New pipeline stages have tests (unit + a fixture-based happy path)
- [ ] New dataset fields are added to `prismacoder/schema.py` AND
      `datasets/schema.json` AND `datasets/scripts/08_finalize_dataset.py`
- [ ] Any new subprocess call has a timeout and captured output
- [ ] Docs updated (README table, relevant card, REPRODUCIBILITY pins)

## Areas where help is most valuable

- Real Vitest/seed bundles for mined records (V5 coverage beyond smoke tests)
- MySQL/SQLServer lanes for the V4 execution gate
- Gold annotation (the report 9.9 rubric) and kappa reporting tooling
- A GRPO trainer for stage S3 (reward wiring exists in `prismacoder/constants.py`)
- Additional domain archetypes (must keep the 2 OOD domains withheld)
