"""Migration-plan generation (T3): schema change -> updated PSL + SQL migration
+ destructive-risk label (report Table 10.1: 'PSL + SQL + risk label').

Reference SQL is composed from the perturbation op list, so every statement is
mechanically derivable from the recorded change metadata (no LLM in the loop).
Destructive = op drops a column/table or changes a column type lossily.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from prismacoder.psl import parse_schema
from prismacoder.synthetic.crud_gen import GeneratedTask
from prismacoder.synthetic.instruction_gen import TemplateVerbalizer, humanize_model
from prismacoder.synthetic.schema_gen import SampledSchema

VERBALIZER = TemplateVerbalizer()

DESTRUCTIVE_OPS = {"remove_field", "retype_field", "swap_key_style"}
OP_DESCRIPTIONS = {
    "add_field": "add a new field",
    "remove_field": "remove an existing field",
    "retype_field": "change a field's type",
    "toggle_optionality": "make a field optional or required",
    "add_unique": "put a unique constraint on a field",
    "add_index": "add an index",
    "add_composite_unique": "add a composite unique constraint",
    "swap_key_style": "switch the primary key strategy",
    "add_enum_field": "introduce an enum-typed field",
    "change_cascade": "change the cascade rule of a relation",
}


@dataclass
class MigrationTask:
    instruction: str
    reference_schema: str
    reference_sql: str
    destructive: bool
    change_desc: str
    persona: str


def _sql_type(prisma_type: str, dialect: str) -> str:
    pg = {
        "String": "TEXT",
        "Int": "INTEGER",
        "Float": "DOUBLE PRECISION",
        "Boolean": "BOOLEAN",
        "DateTime": "TIMESTAMP(3)",
        "BigInt": "BIGINT",
        "Json": "JSONB",
        "Decimal": "DECIMAL(65,30)",
    }
    sq = {
        "String": "TEXT",
        "Int": "INTEGER",
        "Float": "REAL",
        "Boolean": "BOOLEAN",
        "DateTime": "DATETIME",
        "BigInt": "BIGINT",
        "Json": "TEXT",
        "Decimal": "DECIMAL",
    }
    my = {
        "String": "VARCHAR(191)",
        "Int": "INTEGER",
        "Float": "DOUBLE",
        "Boolean": "BOOLEAN",
        "DateTime": "DATETIME(3)",
        "BigInt": "BIGINT",
        "Json": "JSON",
        "Decimal": "DECIMAL(65,30)",
    }
    table = {"postgresql": pg, "postgres": pg, "sqlite": sq, "mysql": my}
    return table.get(dialect, sq).get(prisma_type, "TEXT")


def _table_name(model_name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()


def _diff_models(before: str, after: str) -> dict[str, dict]:
    """Mechanical column diff between two schema versions (field-name keyed)."""
    before_p, after_p = parse_schema(before), parse_schema(after)
    diffs: dict[str, dict] = {}
    for am in after_p.models:
        bm = before_p.get_model(am.name)
        added = [
            f for f in am.scalar_fields() if bm is None or f.name not in {x.name for x in bm.scalar_fields()}
        ]
        removed = (
            [f for f in bm.scalar_fields() if f.name not in {x.name for x in am.scalar_fields()}]
            if bm
            else []
        )
        retyped = []
        if bm:
            bmap = {x.name: x for x in bm.scalar_fields()}
            for f in am.scalar_fields():
                if f.name in bmap and bmap[f.name].type != f.type:
                    retyped.append(f)
        if added or removed or retyped:
            diffs[am.name] = {
                "added": added,
                "removed": removed,
                "retyped": retyped,
                "indexes": [a for a in am.block_attributes if a.startswith("@@index")],
            }
    return diffs


def gen_migration_task(
    sampled: SampledSchema, after_psl: str, ops: list[str], rng: random.Random
) -> MigrationTask:
    before_psl = sampled.render()
    diffs = _diff_models(before_psl, after_psl)
    dialect = (
        "postgresql"
        if sampled.provider in ("postgresql", "postgres")
        else ("mysql" if sampled.provider == "mysql" else "sqlite")
    )

    stmts: list[str] = []
    destructive = any(op in DESTRUCTIVE_OPS for op in ops)
    for model_name, d in diffs.items():
        table = _table_name(model_name)
        for f in d["added"]:
            not_null = " NOT NULL" if not f.is_optional else ""
            default = ""
            if f.default == "now()" and f.type == "DateTime":
                default = " DEFAULT CURRENT_TIMESTAMP"
            elif f.default is not None:
                default = f" DEFAULT {f.default}"
            stmts.append(
                f'ALTER TABLE "{table}" ADD COLUMN "{f.name}" '
                f"{_sql_type(f.type, dialect)}{not_null}{default};"
            )
        for f in d["removed"]:
            stmts.append(f'ALTER TABLE "{table}" DROP COLUMN "{f.name}";')
            destructive = True
        for f in d["retyped"]:
            stmts.append(
                f'ALTER TABLE "{table}" ALTER COLUMN "{f.name}" TYPE ' f"{_sql_type(f.type, dialect)};"
            )
            destructive = True
        for idx in d["indexes"]:
            cols = re.findall(r"\[([^\]]+)\]", idx)
            if cols:
                col_names = ", ".join(f'"{c.strip()}"' for c in cols[0].split(","))
                stmts.append(f'CREATE INDEX "{table}_{len(stmts)}_idx" ON "{table}" ({col_names});')

    change_desc = "; ".join(OP_DESCRIPTIONS.get(op, op) for op in ops)
    models_touched = ", ".join(humanize_model(m) for m in sorted(diffs)[:3]) or "the schema"
    instruction = (
        f"Apply this change to {models_touched}: {change_desc}. "
        f"Return the complete updated schema.prisma and a {dialect} migration. "
        + (
            "Flag the change as destructive: it may lose data."
            if destructive
            else "The migration must be non-destructive; label it as such."
        )
    )

    return MigrationTask(
        instruction=instruction,
        reference_schema=after_psl,
        reference_sql="\n".join(stmts) or "-- no table-level changes required",
        destructive=destructive,
        change_desc=change_desc,
        persona="pm",
    )


def to_generated_task(mt: MigrationTask, category: str = "T3_migration_plan") -> GeneratedTask:
    reference = (
        "// schema.prisma\n"
        + mt.reference_schema.strip()
        + "\n\n// migration.sql\n"
        + mt.reference_sql.strip()
        + f"\n\n// destructiveRisk: {'DESTRUCTIVE' if mt.destructive else 'NON_DESTRUCTIVE'}"
    )
    return GeneratedTask(
        "migration",
        category,
        mt.instruction,
        reference,
        mt.persona,
        ["migration", "destructive_risk" if mt.destructive else "safe_migration"],
    )
