#!/usr/bin/env bash
# PrismaForge - full Phase 1 build (report Part 9).
#
# Requirements:
#   - GITHUB_TOKEN set (5000 req/h) for the mined track
#   - Node.js >= 18.18 on PATH for the full V1-V5 gates
#     (without Node the sweep degrades to static validation and marks
#      records `unverified`; a full-gate re-run is required before release)
#
# Usage:
#   bash datasets/scripts/run_phase1.sh                 # full scale (~120k records)
#   SMOKE=1 bash datasets/scripts/run_phase1.sh         # smoke run (~600 records, no crawl)
set -euo pipefail
cd "$(dirname "$0")/../.."

SEED="${SEED:-20260830}"
LIMIT="${LIMIT:-120000}"

echo "== PrismaForge Phase 1 =="
if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT=600
  CRAWL_ARGS="--offline"
  echo "(smoke mode: limit=$LIMIT, crawl skipped)"
else
  CRAWL_ARGS="${CRAWL_ARGS:-}"
fi

echo "-- stage 01: crawl ------------------------------------------"
python3 datasets/scripts/01_crawl_github.py --out datasets $CRAWL_ARGS

echo "-- stage 02: extract ----------------------------------------"
python3 datasets/scripts/02_extract_corpus.py --raw datasets/raw --out datasets/processed

echo "-- stage 03: clean ------------------------------------------"
python3 datasets/scripts/03_clean_corpus.py --input datasets/processed/extraction_records.jsonl --out datasets/processed

echo "-- stage 04: synthetic --------------------------------------"
python3 datasets/scripts/04_generate_synthetic.py --limit "$LIMIT" --seed "$SEED" --out datasets/synthetic

echo "-- stage 05: augment ----------------------------------------"
python3 datasets/scripts/05_augment.py --seed "$SEED" --out datasets/synthetic

echo "-- stage 06: verify -----------------------------------------"
python3 datasets/scripts/06_verify_all.py --out datasets/verified

echo "-- stage 07: split ------------------------------------------"
python3 datasets/scripts/07_split_dataset.py --seed "$SEED" --out datasets/splits

echo "-- stage 08: finalize ---------------------------------------"
python3 datasets/scripts/08_finalize_dataset.py --splits datasets/splits --out datasets --tag v1.0

echo "== Phase 1 complete. Release set: datasets/train.jsonl (+validation/test/benchmark/dpo) =="
