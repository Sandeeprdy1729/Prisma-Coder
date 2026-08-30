"""Project-wide constants. Single source of truth, mirrors the research report (Parts 9-12)."""

from __future__ import annotations

# --- Versions ---------------------------------------------------------------
PRISMA_VERSION = "7"  # Prisma 7: released Nov 2025, Rust-free client [verified]
NODE_MIN_VERSION = (18, 18)  # minimum Node for prisma CLI v7
TOOLCHAIN = {
    "typescript": "5.6.3",
    "prisma": "7.10.0",  # CLI used inside the sandbox (Rust-free; supports the `prisma-client` generator)
    "@prisma/client": "7.10.0",
    "vitest": "2.1.8",
    "tsx": "4.19.2",
}

# --- Dataset (PrismaForge) --------------------------------------------------
DATASET_NAME = "PrismaForge"
DATASET_VERSION = "1.0.0"
SFT_TOTAL = 120_000  # verified instruction triples (report Table 9.1)
SFT_MIXTURE = {  # task-family -> record budget
    "schema_tasks": 35_000,  # T1/T2/T3
    "client_crud": 25_000,  # T4
    "nested_agg_txn": 30_000,  # T5-T7
    "repo_repair": 20_000,  # T8/T9
    "docs_misc": 10_000,  # T10-T12
}
DPO_PAIRS = 30_000
DEV_PUBLIC = 1_500
GOLD_VERIFIED = 2_000
PRIVATE_HELDOUT = 600
CPT_TOKEN_BUDGET = (8_000_000_000, 12_000_000_000)  # 8-12B tokens

# Near-duplicate thresholds (report 9.2: three dedup granularities)
DEDUP_EXACT = True
DEDUP_MINHASH_JACCARD = 0.80  # MinHash-LSH threshold
DEDUP_ISOMORPHISM = True  # AST/model-graph signature
MINHASH_PERMUTATIONS = 128
MINHASH_SHINGLE_K = 5

# Allowed / denied licenses (report 9.1: permissive only; copyleft excluded by policy)
ALLOWED_LICENSES = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC")
DENIED_LICENSES = ("GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0", "SSPL-1.0", "EUPL-1.2")

# --- Benchmark (PrismaBench) ------------------------------------------------
BENCHMARK_NAME = "PrismaBench"
BENCHMARK_VERSION = "1.0.0"
BENCHMARK_TOTAL = 1_200
# report Table 10.1 - category -> task count (sums to 1,200)
BENCH_TASK_COUNTS = {
    "T1_schema_greenfield": 120,
    "T2_schema_edit": 120,
    "T3_migration_plan": 90,
    "T4_client_crud": 150,
    "T5_complex_reads": 140,
    "T6_nested_writes_txn": 120,
    "T7_aggregations": 100,
    "T8_repo_completion": 120,
    "T9_bug_fix": 100,
    "T10_cross_file": 60,
    "T11_performance": 60,
    "T12_security": 20,
}
assert sum(BENCH_TASK_COUNTS.values()) == BENCHMARK_TOTAL

# Difficulty mechanics (report 10.3)
DIFFICULTY_LEVELS = ("L1", "L2", "L3")
LEVEL_RULES = {  # models bounds, relation depth, nesting depth
    "L1": {"models": (2, 6), "rel_depth": (1, 2), "nesting": 1},
    "L2": {"models": (6, 10), "rel_depth": (2, 3), "nesting": 2},
    "L3": {"models": (10, 14), "rel_depth": (3, 4), "nesting": 3},
}

# 12 domain archetypes (report 9.4); 2 of these are withheld from public data for OOD eval
DOMAINS = (
    "ecommerce",
    "saas_billing",
    "social",
    "cms",
    "fintech_ledger",
    "health",
    "logistics",
    "iot_telemetry",
    "lms",
    "analytics",
    "multi_tenant_hr",
    "marketplace",
)
OOD_WITHHELD_DOMAINS = ("health", "multi_tenant_hr")  # never released in public training data

# --- Verifier gates ---------------------------------------------------------
GATE_NAMES = ("V1_validate", "V2_generate", "V3_tsc", "V4_execute", "V5_vitest")
GATE_ORDER = {"V1_validate": 1, "V2_generate": 2, "V3_tsc": 3, "V4_execute": 4, "V5_vitest": 5}
# GRPO reward weights (report 12.2): R = 0.25*V1 + 0.15*V2 + 0.20*V3 + 0.40*V5 (+style, -hack)
GRPO_REWARD_WEIGHTS = {
    "V1_validate": 0.25,
    "V2_generate": 0.15,
    "V3_tsc": 0.20,
    "V4_execute": 0.00,
    "V5_vitest": 0.40,
}

# Composite scoring for PrismaBench (execution-dominated, report 10.1/11.2)
COMPOSITE_WEIGHTS = {"compile": 0.20, "execute": 0.60, "style": 0.10, "efficiency": 0.10}

# --- Error taxonomy (report Table 11.3) -------------------------------------
ERROR_TAXONOMY = {
    "E1": "Syntax/grammar - malformed PSL or TS syntax error (caught by V1/tsc parse)",
    "E2": "Schema semantics - unpaired relation fields, bad defaults, duplicate names (V1)",
    "E3": "Codegen gate - schema valid but generator fails (V2)",
    "E4": "Type errors - wrong include shape, missing non-null handling (V3)",
    "E5": "Runtime error - P2002/P2003/P2025, null deref (V4/V5)",
    "E6": "Wrong semantics - passes gates but behavior inverted/over-broad (tests+audit)",
    "E7": "Incomplete - request partially satisfied (test coverage+audit)",
    "E8": "Unsafe - string-concat raw SQL, missing tenant scoping, unlabeled destructive migration",
    "E9": "Inefficient - N+1, fetch-then-filter (query-count probe)",
    "E10": "Unidiomatic - valid but style-inverted naming, enum misuse, antipattern includes",
}

# --- Task category ids (shared with benchmark + synthetic) -------------------
CATEGORY_IDS = tuple(BENCH_TASK_COUNTS.keys())

# Request persona tags (report 9.7: junior / pm / expert phrasings)
PERSONAS = ("junior", "pm", "expert", "none")

# --- Seed / reproducibility --------------------------------------------------
DEFAULT_SEED = 20260830

# --- Style: system prompt used across SFT/DPO/inference ----------------------
SYSTEM_PROMPT = (
    "You are PrismaCoder, an expert in the Prisma ORM. You write schema.prisma (PSL) and "
    "TypeScript Prisma Client code that compiles, validates, and executes correctly. "
    "Output only the requested artifact inside a fenced code block. Use Prisma v7 conventions."
)
