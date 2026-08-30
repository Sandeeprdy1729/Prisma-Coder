# PrismaCoder - top-level orchestration.
# Every stage below is idempotent and safe to re-run.

PYTHON      ?= python3
PIP         ?= $(PYTHON) -m pip
SEED        ?= 20260830
SMOKE_N     ?= 40          # dataset size for `make smoke`
BENCH_SCALE ?= 0.05        # fraction of the 1,200-task benchmark built by `make benchmark`

.PHONY: help install install-dev lint format test smoke phase1 phase2 phase3 eval serve docker-build docker-up clean tree

help:
	@echo "PrismaCoder - open-source Text-to-Prisma LLM"
	@echo "  make install      install core package"
	@echo "  make install-dev  install with dev/eval extras"
	@echo "  make lint         ruff + black --check"
	@echo "  make test         pytest unit + dataset + benchmark tests"
	@echo "  make smoke        end-to-end pipeline on a tiny config (no GPU/node needed)"
	@echo "  make phase1       full PrismaForge build (network + node required)"
	@echo "  make phase2       full PrismaBench build (1,200 tasks)"
	@echo "  make phase3       launch SFT QLoRA training (GPU required)"
	@echo "  make eval         evaluate a submission on PrismaBench"
	@echo "  make serve        start the FastAPI inference server"
	@echo "  make docker-build build the inference + verifier images"
	@echo "  make docker-up    start api + postgres + verifier stack"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,eval]"

lint:
	ruff check prismacoder datasets benchmark training evaluation inference utils tests
	black --check prismacoder datasets benchmark training evaluation inference utils tests

format:
	black prismacoder datasets benchmark training evaluation inference utils tests
	ruff check --fix prismacoder datasets benchmark training evaluation inference utils tests

test:
	pytest -m "not gpu and not node and not network" tests

smoke:
	$(PYTHON) datasets/scripts/04_generate_synthetic.py --limit $(SMOKE_N) --seed $(SEED) --out datasets/synthetic
	$(PYTHON) datasets/scripts/06_verify_all.py --input datasets/synthetic/synthetic_pool.jsonl --light-only
	$(PYTHON) datasets/scripts/07_split_dataset.py --synthetic datasets/verified/synthetic_pool.verified.jsonl --out datasets/splits --seed $(SEED)
	$(PYTHON) datasets/scripts/08_finalize_dataset.py --splits datasets/splits --out datasets --tag smoke
	$(PYTHON) benchmark/generator/generate_benchmark.py --scale $(BENCH_SCALE) --seed $(SEED) --out benchmark/public
	$(PYTHON) benchmark/evaluator/make_dummy_submission.py --out benchmark/evaluator/dummy_submission.jsonl
	$(PYTHON) benchmark/evaluator/evaluate.py --tasks benchmark/public/prismabench_public.jsonl --submission benchmark/evaluator/dummy_submission.jsonl --out benchmark/leaderboard/smoke_results.json
	$(PYTHON) benchmark/evaluator/leaderboard.py --results benchmark/leaderboard --out benchmark/leaderboard/LEADERBOARD.md

phase1:  ## Full PrismaForge build. Requires GITHUB_TOKEN and Node.js >= 18.18.
	bash datasets/scripts/run_phase1.sh

phase2:
	$(PYTHON) benchmark/generator/generate_benchmark.py --scale 1.0 --seed $(SEED) --out benchmark/public

phase3:
	$(PYTHON) training/scripts/train_sft.py --config training/configs/qlora_3b.yaml

eval:
	$(PYTHON) benchmark/evaluator/evaluate.py --tasks benchmark/public/prismabench_public.jsonl \
	--submission $(SUB) --out benchmark/leaderboard/eval_results.json
	$(PYTHON) benchmark/evaluator/leaderboard.py --results benchmark/leaderboard --out benchmark/leaderboard/LEADERBOARD.md

serve:
	$(PYTHON) -m uvicorn inference.server.app:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -f docker/Dockerfile -t prismacoder/inference:latest .
	docker build -f docker/Dockerfile.verifier -t prismacoder/verifier:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

tree:
	$(PYTHON) utils/env_check.py --tree

clean:
	rm -rf datasets/synthetic/* datasets/splits/* benchmark/leaderboard/*.json sandbox/ .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
