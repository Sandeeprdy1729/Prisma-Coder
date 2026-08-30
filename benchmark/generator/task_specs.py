"""PrismaBench task specification (report Table 10.1).

Category definitions with per-category difficulty spreads and scoring-gate
requirements. Counts sum to exactly 1,200.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prismacoder.constants import BENCH_TASK_COUNTS, COMPOSITE_WEIGHTS


@dataclass(frozen=True)
class TaskSpec:
    tid: str  # T1..T12
    category_id: str  # T1_schema_greenfield etc.
    artifact: str  # PSL | TS | PSL+SQL
    levels: tuple[str, ...]  # allowed difficulty spread
    gates: tuple[str, ...]  # scoring gates (beyond static)
    description: str


TASK_SPECS: tuple[TaskSpec, ...] = (
    TaskSpec(
        "T1",
        "T1_schema_greenfield",
        "PSL",
        ("L1", "L2", "L3"),
        ("V1_validate", "V2_generate", "V4_execute"),
        "Schema from NL description (green-field)",
    ),
    TaskSpec(
        "T2",
        "T2_schema_edit",
        "PSL diff",
        ("L1", "L2", "L3"),
        ("V1_validate", "V2_generate", "V4_execute", "semantic_diff"),
        "Schema edit from NL change request",
    ),
    TaskSpec(
        "T3",
        "T3_migration_plan",
        "PSL+SQL+risk",
        ("L2", "L3"),
        ("V1_validate", "V2_generate", "migration_dry_run"),
        "Migration plan incl. destructive-risk label",
    ),
    TaskSpec(
        "T4",
        "T4_client_crud",
        "TS",
        ("L1", "L2"),
        ("V3_tsc", "V4_execute", "V5_vitest"),
        "Client CRUD: find/create/update/delete/upsert",
    ),
    TaskSpec(
        "T5",
        "T5_complex_reads",
        "TS",
        ("L1", "L2", "L3"),
        ("V3_tsc", "V4_execute", "V5_vitest"),
        "Include/select projections, nested filters, pagination, distinct",
    ),
    TaskSpec(
        "T6",
        "T6_nested_writes_txn",
        "TS",
        ("L2", "L3"),
        ("V3_tsc", "V4_execute", "V5_vitest", "isolation_check"),
        "Nested writes and interactive transactions",
    ),
    TaskSpec(
        "T7",
        "T7_aggregations",
        "TS",
        ("L2", "L3"),
        ("V3_tsc", "V4_execute", "V5_vitest"),
        "aggregate / groupBy / count with having-style filters",
    ),
    TaskSpec(
        "T8",
        "T8_repo_completion",
        "TS multi-file",
        ("L2", "L3"),
        ("V3_tsc", "V4_execute", "V5_vitest"),
        "Repo-aware completion: implement service/repository function",
    ),
    TaskSpec(
        "T9",
        "T9_bug_fix",
        "TS|PSL",
        ("L1", "L2", "L3"),
        ("V3_tsc", "V4_execute", "V5_vitest"),
        "Bug fix: given failing code + diagnostic, repair to green",
    ),
    TaskSpec(
        "T10",
        "T10_cross_file",
        "TS multi-file",
        ("L3",),
        ("V3_tsc", "V4_execute", "V5_vitest"),
        "Cross-file reasoning: data layer + API route + types coherence",
    ),
    TaskSpec(
        "T11",
        "T11_performance",
        "TS+index",
        ("L2", "L3"),
        ("V3_tsc", "V4_execute", "V5_vitest", "query_count_probe"),
        "Performance-aware query task (avoid N+1, pushdown, index awareness)",
    ),
    TaskSpec(
        "T12",
        "T12_security",
        "TS",
        ("L3",),
        ("V3_tsc", "V4_execute", "V5_vitest", "security_probes"),
        "Security-hardened task: parameterize raw SQL, tenant scoping",
    ),
)

SPEC_BY_ID = {s.category_id: s for s in TASK_SPECS}


def difficulty_weights(tid: str) -> dict[str, float]:
    """Per-category level distribution (report 10.2/10.3)."""
    if tid == "T10":
        return {"L1": 0.0, "L2": 0.0, "L3": 1.0}
    if tid == "T12":
        return {"L1": 0.0, "L2": 0.0, "L3": 1.0}
    if tid in ("T3", "T6", "T7", "T8", "T11"):
        return {"L1": 0.0, "L2": 0.65, "L3": 0.35}
    if tid == "T4":
        return {"L1": 0.7, "L2": 0.3, "L3": 0.0}
    return {"L1": 0.4, "L2": 0.4, "L3": 0.2}


def build_plan() -> list[tuple[str, str, int]]:
    """Expand (category, level, count) triples covering all 1,200 tasks."""
    plan: list[tuple[str, str, int]] = []
    for spec in TASK_SPECS:
        total = BENCH_TASK_COUNTS[spec.category_id]
        weights = difficulty_weights(spec.tid)
        levels = [lv for lv in spec.levels]
        w = [weights.get(lv, 0.0) for lv in levels]
        alloc = _largest_remainder(total, w)
        for lv, n in zip(levels, alloc, strict=False):
            if n > 0:
                plan.append((spec.category_id, lv, n))
    return plan


def _largest_remainder(total: int, weights: list[float]) -> list[int]:
    base = [int(total * w) for w in weights]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (total * weights[i]) - base[i], reverse=True)
    for i in range(rem):
        base[order[i % len(order)]] += 1
    return base


# Composite scoring (report 10.1/11.2): execution-dominated
def composite_score(parts: dict[str, float]) -> float:
    """parts: {compile, execute, style, efficiency} in [0,1]."""
    total = sum(COMPOSITE_WEIGHTS[k] * parts.get(k, 0.0) for k in COMPOSITE_WEIGHTS)
    return round(total, 4)


def bench_task_id(tid: str, seq: int) -> str:
    return f"pb-{tid.lower()}-{seq:04d}"


def public_task_shape(task: dict[str, Any]) -> dict[str, Any]:
    """Public shape: instruction + context only; reference/tests withheld."""
    return {
        "bench_id": task["bench_id"],
        "category": task["category"],
        "level": task["level"],
        "domain": task["domain"],
        "instruction": task["instruction"],
        "context": task["context"],
        "input": task.get("input", ""),
        "canary": task.get("canary", ""),
        "reference": None,
        "tests": None,
    }
