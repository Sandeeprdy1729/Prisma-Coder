"""Canonical PrismaForge / PrismaBench record schema.

Every record produced anywhere in the project conforms to this contract
(mirrors Listing 9.1 of the research report). Validation is dependency-free
so the whole pipeline can run on the stdlib alone.
"""

from __future__ import annotations

import re
from typing import Any

# Task-type vocabulary (T-ids from report Table 10.1 / dataset mixture Table 9.1)
TASK_TYPES = (
    "nl2prisma",  # T1: NL -> PSL schema
    "schema_edit",  # T2: edit PSL per change request
    "migration",  # T3: migration plan + destructive-risk label
    "crud",  # T4
    "complex_read",  # T5
    "nested_write",  # T6
    "aggregation",  # T7
    "repo_completion",  # T8
    "bug_fix",  # T9
    "cross_file",  # T10
    "performance",  # T11
    "security",  # T12
    "nl_explain",  # Prisma -> NL (auxiliary SFT variant)
    "qa",  # conceptual QA (auxiliary SFT variant)
    "completion",  # code completion (auxiliary SFT variant)
)

TRACKS = ("mined", "synthetic", "official", "docs", "augmented")
LICENSES = ("MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "UNKNOWN")
PROVIDERS = ("sqlite", "postgres", "mysql")
PERSONAS = ("junior", "pm", "expert", "none")

_ID_RE = re.compile(r"^(pf|pb)-[a-z0-9]+(-[a-z0-9]+)*$")
_BENCH_ID_RE = re.compile(r"^pb-(t[1-9]|t1[0-2])-\d{4}$")


class RecordSchemaError(ValueError):
    """Raised when a record violates the canonical schema."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RecordSchemaError(msg)


def validate_record(rec: dict[str, Any], *, strict: bool = False) -> list[str]:
    """Validate a record dict. Returns a list of human-readable problems (empty = valid).

    With ``strict=True`` the call raises :class:`RecordSchemaError` on the first problem.
    """
    problems: list[str] = []
    rid = rec.get("id", "")
    if not (isinstance(rid, str) and (_ID_RE.match(rid) or _BENCH_ID_RE.match(rid))):
        problems.append(f"id '{rid}' must match 'pf-*' or 'pb-tN-NNNN'")

    task = rec.get("task") or {}

    if task.get("type") not in TASK_TYPES:
        problems.append(f"task.type '{task.get('type')}' not in {TASK_TYPES}")
    if task.get("difficulty") not in ("L1", "L2", "L3", "easy", "medium", "hard"):
        problems.append(f"task.difficulty '{task.get('difficulty')}' invalid")
    cats = task.get("categories", [])
    if not isinstance(cats, list) or not cats:
        problems.append("task.categories must be a non-empty list")
    for c in cats:
        if c not in CATEGORY_IDS_:
            problems.append(f"task.categories contains unknown id '{c}'")

    if not (task.get("instruction") or "").strip():
        problems.append("task.instruction must be a non-empty string")
    ctx = task.get("context") or {}
    if not (ctx.get("schema") or "").strip():
        problems.append("task.context.schema must contain the PSL schema")
    files = ctx.get("files", [])
    if not isinstance(files, list):
        problems.append("task.context.files must be a list")

    if rec.get("reference") is None and (task.get("reference") is None) and "benchmark" not in rid:
        problems.append("reference (gold artifact) is required outside benchmark public releases")

    src = rec.get("source") or {}
    if src.get("track") not in TRACKS:
        problems.append(f"source.track '{src.get('track')}' not in {TRACKS}")
    if src.get("track") == "mined" and not src.get("origin_repo"):
        problems.append("mined records require source.origin_repo")
    if src.get("license") not in LICENSES:
        problems.append(f"source.license '{src.get('license')}' not in {LICENSES}")

    ver = rec.get("verification") or {}
    if ver.get("status") not in ("verified", "unverified", "failed", "pending"):
        problems.append("verification.status must be verified|unverified|failed|pending")

    if rec.get("domain") not in DOMAINS_:
        problems.append(f"domain '{rec.get('domain')}' not in the 12 archetypes")

    if problems and strict:
        raise RecordSchemaError(problems[0])

    return problems


def canonicalize_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and normalize field order for a record. Idempotent."""
    out = {
        "id": rec["id"],
        "schema_version": rec.get("schema_version", "1.0"),
        "source": {
            "track": rec.get("source", {}).get("track", "synthetic"),
            "origin_repo": rec.get("source", {}).get("origin_repo", ""),
            "url": rec.get("source", {}).get("url", ""),
            "license": rec.get("source", {}).get("license", "UNKNOWN"),
            "commit": rec.get("source", {}).get("commit", ""),
            "path": rec.get("source", {}).get("path", ""),
            "provenance_hash": rec.get("source", {}).get("provenance_hash", ""),
        },
        "domain": rec["domain"],
        "task": {
            "type": rec["task"]["type"],
            "category": rec["task"].get("category", ""),
            "difficulty": rec["task"].get("difficulty", "L1"),
            "categories": rec["task"].get("categories", []),
            "instruction": rec["task"]["instruction"],
            "context": rec["task"].get("context", {}),
            "input": rec["task"].get("input", ""),
            "reference": rec.get("task", {}).get("reference", rec.get("reference", "")),
            "persona": rec["task"].get("persona", "none"),
            "tests": rec["task"].get("tests", {}),
            "destructive_risk": rec["task"].get("destructive_risk", False),
        },
        "verification": {
            "status": rec.get("verification", {}).get("status", "unverified"),
            "gates": rec.get("verification", {}).get("gates", {}),
            "runtime_ms": rec.get("verification", {}).get("runtime_ms", 0),
            "verifier_version": rec.get("verification", {}).get("verifier_version", ""),
        },
        "difficulty_scores": {
            "models": rec.get("difficulty_scores", {}).get("models", 0),
            "relations": rec.get("difficulty_scores", {}).get("relations", 0),
            "nesting": rec.get("difficulty_scores", {}).get("nesting", 0),
            "ops": rec.get("difficulty_scores", {}).get("ops", 0),
            "ambiguity": rec.get("difficulty_scores", {}).get("ambiguity", 0),
        },
        "style_tags": rec.get("style_tags", []),
        "meta": {
            "created": rec.get("meta", {}).get("created", ""),
            "generator_version": rec.get("meta", {}).get("generator_version", ""),
            "seed": rec.get("meta", {}).get("seed", 0),
            "tags": rec.get("meta", {}).get("tags", []),
            "canary": rec.get("meta", {}).get("canary", ""),
        },
    }
    # Preference/reasoning pairs carry an extra block (SFT auxiliary variants)
    if "preference" in rec:
        out["preference"] = rec["preference"]  # {"chosen": ..., "rejected": ..., "basis": ...}
    if "reasoning" in rec:
        out["reasoning"] = rec["reasoning"]  # {"plan_span": ...}
    return out


# Populated from constants at import time (kept separate to avoid a circular import).
from prismacoder.constants import (  # noqa: E402
    BENCH_TASK_COUNTS,
    CATEGORY_IDS,
    DIFFICULTY_LEVELS,
    DOMAINS,
)

CATEGORY_IDS_ = tuple(BENCH_TASK_COUNTS.keys())
DOMAINS_ = DOMAINS

__all__ = [
    "TASK_TYPES",
    "TRACKS",
    "LICENSES",
    "PROVIDERS",
    "PERSONAS",
    "RecordSchemaError",
    "validate_record",
    "canonicalize_record",
    "DIFFICULTY_LEVELS",
    "CATEGORY_IDS",
    "DOMAINS",
]
