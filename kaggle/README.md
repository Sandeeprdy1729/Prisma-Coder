# `kaggle/` - the production Kaggle notebooks

Three notebooks, all zero-manual-edit (`Run All` works top to bottom):

## `prismacoder_end2end_kaggle.ipynb` - **the WHOLE pipeline** (data -> benchmark -> training -> eval)

One notebook, one session: the complete PrismaCoder production loop.

1. installs the pinned stack (transformers 4.52.3, trl 0.19.1, peft 0.16.0, bitsandbytes 0.46.1),
2. pulls `HF_TOKEN` (and optionally `GITHUB_TOKEN`) from Kaggle secrets,
3. locates the repo automatically (mounted Dataset / `PrismaCoder.zip` / clone),
4. **Part A - PrismaForge**: repo stages 01-06 via exact CLI parity - GitHub crawl
   (mined track, token-gated; `--offline` fallback) -> extract -> clean ->
   **synthetic track** (`SYNTH_LIMIT`, default 12k, scales to the report's 120k) ->
   augment -> verification sweep (LightValidator V0; optional tiny V1-V5 demo),
5. **Part B - PrismaBench**: seeded build at `BENCH_SCALE` (1.0 = the full 1,200-task
   Table 10.1 composition, assertion-checked), public + private keys + canaries +
   provenance, plus an internal reference-carrying dev build for in-notebook scoring;
   harness smoke: dummy submission scores 0.0 through the full evaluator,
6. **Part C - finalize**: stages 07-08 (database-level splits, DPO negatives routing,
   release set) - runs *after* the benchmark so the **contamination firewall** has the
   provenance registry,
7. **Part D - training with checkpoints**: SFT (QLoRA, completion-only loss, soft
   ordering) -> DPO (static gate-differential + on-policy mined pairs) -> GRPO
   (online RL, verifier reward) -> iterative RL self-play round. **Every stage saves
   step-checkpoints (adapter + optimizer + scheduler) and auto-resumes via
   `get_last_checkpoint`**; a utility cell restores checkpoints from a previous
   session's attached output (12h-limit survival recipe in the intro),
8. **Part E - evaluation**: the final model generates submissions on the benchmark
   built in Part B and is scored by the repo evaluator (pass@1 + Wilson CI,
   exact match / CodeBLEU / tree similarity, composite, E1-E10 taxonomy,
   per-category table) + stage-wise gate pass@1 chart,
9. **Part F - ship**: HF push (model + optionally the PrismaForge release files) + GGUF export.

Every part is switchable in Cell 2 (`RUN_DATA_CREATE / RUN_BENCH / RUN_TRAIN /
RUN_SFT / RUN_DPO / RUN_GRPO / RUN_ITERATIVE_RL / RUN_BENCH_EVAL / PUSH_TO_HUB / RUN_GGUF`).

**Usage**: import -> GPU T4 x2 -> Internet ON -> add secrets -> `Run All`. Defaults fit
one 12h session; or split into two sessions using the checkpoint resume recipe.

## `prismacoder_sft_dpo_grpo_kaggle.ipynb` - the full alignment pipeline (SFT -> DPO -> GRPO -> RL)

The alignment stages only, starting from data on the HF Hub (or an offline deterministic
mini-corpus built by the repo's synthetic generator). Same trainer stack and stage
switches as Part D of the end-to-end notebook. Use it when PrismaForge is already
built/published and you only need training + RL.

## `prismacoder_qlora_kaggle.ipynb` - SFT-only quick notebook

The single-stage QLoRA SFT run (install -> data -> train -> static gate eval -> merge
-> push -> GGUF). Use it for a fast first pass or when you only need the imitation
stage; the end-to-end notebook supersedes it for the report recipe.

## Notes

- Kaggle T4s have no bf16 and no flash-attn: the notebooks auto-select fp16 + SDPA.
- The V0 static gate is the no-Node fallback; V0-green records are `unverified` until a
  full-gate (V1-V5) re-run with `benchmark/evaluator/evaluate.py --full-gates`.
  Empty artifacts fail V0 with error class E7 (they can never pass a full gate).
- Step-checkpoints keep optimizer state (true resume); `save_total_limit=3` keeps the
  rolling window small enough for the ~20 GB `/kaggle/working` quota.
- Merges run on CPU; prefer T4 x2 sessions (29 GB system RAM) over P100 (13 GB).
