"""Quality filtering + broken-schema detection (report 9.2 'validate probe').

Stages (all cheap, all static - compilation happens later in the V-gates):
1. ``check_schema``      - PSL structural sanity: block balance, datasource,
                           generator, >=1 model, paired relation fields, id field.
2. ``check_source_code`` - TS sanity: brace/paren balance, prisma import present
                           when client calls exist, no TODO-riddled stubs.
3. ``quality_score``     - weighted score used for stratification and to bound
                           low-value samples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from prismacoder.psl import parse_schema

RE_MODEL = re.compile(r"(?m)^model\s+\w+\s*\{")
RE_RELATION_ATTR = re.compile(r"@relation\(")
RE_TODO = re.compile(r"(?i)\b(TODO|FIXME|XXX|HACK)\b")


@dataclass
class CheckResult:
    ok: bool
    problems: list[str]
    score: float = 1.0  # 0..1 quality score


def check_schema(schema_text: str) -> CheckResult:
    problems: list[str] = []

    if schema_text.count("{") != schema_text.count("}"):
        problems.append("unbalanced braces")
        return CheckResult(False, problems, 0.0)

    try:
        parsed = parse_schema(schema_text)
    except ValueError as e:
        problems.append(f"unparseable PSL: {e}")
        return CheckResult(False, problems, 0.0)

    if not parsed.models:
        problems.append("no model blocks parseable")
        return CheckResult(False, problems, 0.0)

    if not parsed.datasource:
        problems.append("missing datasource block")

    provider = parsed.datasource.get("provider", "")
    if provider and provider not in (
        "sqlite",
        "postgresql",
        "postgres",
        "mysql",
        "sqlserver",
        "mongodb",
        "cockroachdb",
    ):
        problems.append(f"unknown datasource provider '{provider}'")

    if not parsed.generator:
        problems.append("missing generator block")

    model_names = set(parsed.model_names())
    for m in parsed.models:
        if not m.fields:
            problems.append(f"model {m.name} has no fields")
            continue
        if m.primary_key() is None:
            problems.append(f"model {m.name} has no @id / @@id")
        for f in m.relation_fields():
            if f.type not in model_names:
                problems.append(f"relation field {m.name}.{f.name} points to unknown model {f.type}")
                continue
            if f.is_list:
                continue  # back-relation list side
            target = parsed.get_model(f.type)
            if target is not None and "@relation(" not in " ".join(f.attributes):
                problems.append(f"relation field {m.name}.{f.name} missing @relation attribute")
            if target is not None:
                # pairing heuristic: the target must have a scalar side or an opposite relation
                opp = [tf for tf in target.relation_fields() if tf.type == m.name]
                scalars = target.scalar_fields()
                if not opp and len(scalars) < 1:
                    problems.append(f"unpaired relation {m.name}.{f.name} -> {target.name}")

    # duplicate field names within a model
    for m in parsed.models:
        names = [f.name for f in m.fields]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            problems.append(f"model {m.name} duplicate fields: {sorted(dupes)}")

    score = 1.0 - min(0.3 * len(problems), 1.0)
    return CheckResult(not problems, problems, max(score, 0.0))


def check_source_code(code: str, *, expect_prisma: bool = True) -> CheckResult:
    problems: list[str] = []
    for open_c, close_c in (("{", "}"), ("(", ")"), ("[", "]")):
        if code.count(open_c) != code.count(close_c):
            problems.append(f"unbalanced {open_c}{close_c}")
    if (
        expect_prisma
        and re.search(r"\bprisma\b|\bPrismaClient\b", code) is None
        and re.search(r"\.\s*(findMany|create|update|delete|upsert|aggregate|groupBy)\s*\(", code)
    ):
        problems.append("client calls present but no prisma client reference")
    if len(RE_TODO.findall(code)) > 6:
        problems.append("stub-like file (too many TODO/FIXME markers)")
    if code.strip() == "":
        problems.append("empty source file")
    return CheckResult(not problems, problems, 1.0 if not problems else 0.4)


def quality_score_record(rec: dict[str, Any]) -> float:
    """Composite 0..1 score over schema + code + metadata completeness."""
    checks = [check_schema(rec.get("schema", ""))]
    for code in (rec.get("source_codes") or {}).values():
        checks.append(check_source_code(code))
    meta_ok = 1.0 if (rec.get("license") not in (None, "", "UNKNOWN")) else 0.5
    base = sum(c.score for c in checks) / max(len(checks), 1)
    return round(0.85 * base + 0.15 * meta_ok, 4)


def filter_corpus(
    records: list[dict[str, Any]], *, min_score: float = 0.6
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split records into (accepted, rejected-with-reasons)."""
    accepted, rejected = [], []
    for rec in records:
        chk = check_schema(rec.get("schema", ""))
        score = quality_score_record(rec)
        if chk.ok and score >= min_score:
            rec["quality_score"] = score
            accepted.append(rec)
        else:
            rejected.append(
                {"id": rec.get("id", rec.get("repo", "?")), "problems": chk.problems, "score": score}
            )
    return accepted, rejected
