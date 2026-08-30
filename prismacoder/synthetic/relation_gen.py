"""Nested writes, interactive transactions, aggregations (T6 / T7).

All references are composed over the sampler's internal representation:
relation fields and FK scalar names come from the parsed models, so emitted
client calls match the schema. Transaction references use the interactive
form (``tx`` parameter) with an explicit failure path so V5 tests can assert
rollback semantics.
"""

from __future__ import annotations

import random

from prismacoder.psl import Field, Model
from prismacoder.synthetic.crud_gen import GeneratedTask, _client_import, _snake_prop
from prismacoder.synthetic.instruction_gen import TemplateVerbalizer, humanize_model
from prismacoder.synthetic.schema_gen import SampledSchema

VERBALIZER = TemplateVerbalizer()


def _fk_scalar_for(s: SampledSchema, child: Model, parent: Model) -> Field | None:
    candidates = [
        f
        for f in child.scalar_fields()
        if f.name.lower() == f"{_snake(parent.name)}id" or f.name.lower().endswith(f"{_snake(parent.name)}id")
    ]
    return candidates[0] if candidates else None


# --- T6a: nested writes ---------------------------------------------------------
def gen_nested_create(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    for m in rng.sample(s.models, len(s.models)):
        list_rels = [f for f in m.relation_fields() if f.is_list]
        if not list_rels:
            continue
        rf = rng.choice(list_rels)
        child = s.model(rf.type)
        if child is None:
            continue
        req = s.required_scalars(child.name)
        create_payload = ",\n".join(f"            {f.name}: data.{_camel(rf.name)}_{f.name}," for f in req)
        iface = ",\n".join(f"  {_camel(rf.name)}_{f.name}: {ts_type(f)};" for f in req)
        ref = f"""{_client_import()}
export interface Create{m.name}With{_camel(rf.type)}Input {{
{iface}
}}

export async function create{m.name}With{_camel(rf.type)}(data: Create{m.name}With{_camel(rf.type)}Input) {{
  return prisma.{_snake_prop(m.name)}.create({{
    data: {{
      {rf.name}: {{
        create: {{
{create_payload}
        }},
      }},
    }},
    include: {{ {rf.name}: true }},
  }});
}}
"""
        instruction = VERBALIZER.nested_write(m.name, rf.type, "create")
        return GeneratedTask(
            "nested_write",
            "T6_nested_writes_txn",
            instruction,
            ref,
            persona,
            ["nested_write", "atomic_create"],
        )
    return _fallback(s, rng, "nested_write")


def gen_connect_or_create(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    pairs = []
    for m in s.models:
        for rf in m.relation_fields():
            if rf.is_list:
                uniq = s.unique_scalars(rf.type)
                if uniq:
                    pairs.append((m, rf, rng.choice(uniq)))
    if not pairs:
        return _fallback(s, rng, "nested_write")
    m, rf, key = rng.choice(pairs)
    from prismacoder.synthetic.crud_gen import id_scalar_type

    owner_id_type = id_scalar_type(s, m.name)
    ref = f"""{_client_import()}
export async function attach{_camel(rf.type)}To{_camel(m.name)}(
  {m.name[:1].lower() + m.name[1:]}Id: {owner_id_type},
  {key.name}: {ts_type(key)},
) {{
  return prisma.{_snake_prop(m.name)}.update({{
    where: {{ id: {m.name[:1].lower() + m.name[1:]}Id }},
    data: {{
      {rf.name}: {{
        connectOrCreate: {{
          where: {{ {key.name} }},
          create: {{ {key.name} }},
        }},
      }},
    }},
    include: {{ {rf.name}: true }},
  }});
}}
"""
    instruction = VERBALIZER.nested_write(m.name, rf.type, "connectOrCreate")
    return GeneratedTask(
        "nested_write",
        "T6_nested_writes_txn",
        instruction,
        ref,
        persona,
        ["connect_or_create", "idempotent_attach"],
    )


# --- T6b: interactive transactions ----------------------------------------------
def gen_transaction_transfer(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    """Atomic value transfer between two records of the same or sibling models."""
    candidates = []
    for m in s.models:
        amount = next(
            (
                f
                for f in m.scalar_fields()
                if "amount" in f.name.lower() or "balance" in f.name.lower() or "cents" in f.name.lower()
            ),
            None,
        )
        if amount:
            candidates.append((m, amount))
    if not candidates:
        return _fallback(s, rng, "nested_write")
    m, amount_f = rng.choice(candidates)
    key = s.unique_scalars(m.name) or [f for f in m.scalar_fields() if "@id" in f.attributes]
    key_f = key[0] if key else None
    if key_f is None:
        return _fallback(s, rng, "nested_write")
    ref = f"""{_client_import()}
export class InsufficientFundsError extends Error {{
  constructor() {{
    super("insufficient funds");
    this.name = "InsufficientFundsError";
  }}
}}

export async function transfer{_camel(m.name)}Balance(
  fromId: {ts_type(key_f)},
  toId: {ts_type(key_f)},
  amount: number,
) {{
  return prisma.$transaction(async (tx) => {{
    const from = await tx.{_snake_prop(m.name)}.findUniqueOrThrow({{ where: {{ {key_f.name}: fromId }} }});
    if (Number(from.{amount_f.name}) < amount) {{
      throw new InsufficientFundsError();
    }}
    await tx.{_snake_prop(m.name)}.update({{
      where: {{ {key_f.name}: fromId }},
      data: {{ {amount_f.name}: {{ decrement: amount }} }},
    }});
    const to = await tx.{_snake_prop(m.name)}.update({{
      where: {{ {key_f.name}: toId }},
      data: {{ {amount_f.name}: {{ increment: amount }} }},
    }});
    return to;
  }});
}}
"""
    instruction = VERBALIZER.transaction([m.name, m.name], amount_f.name)
    return GeneratedTask(
        "nested_write",
        "T6_nested_writes_txn",
        instruction,
        ref,
        persona,
        ["interactive_transaction", "rollback_guard"],
    )


def gen_multi_model_transaction(
    s: SampledSchema, rng: random.Random, persona: str = "expert"
) -> GeneratedTask:
    """Create an audit/log record in the same transaction as a primary mutation."""
    m = rng.choice(s.models)
    req = s.required_scalars(m.name)
    audit = next(
        (
            a
            for a in s.models
            if a.name.lower().startswith(("audit", "event", "shipmentevent", "audit event"))
        ),
        None,
    )
    if audit is None or not req:
        return gen_transaction_transfer(s, rng, persona)
    audit_req = s.required_scalars(audit.name)
    nl = chr(10)  # newline (kept out of f-strings for pre-3.12 compatibility)
    iface_lines = nl.join(f"  {f.name}: {ts_type(f)};" for f in req)
    audit_action_field = audit.name[:1].lower() + audit.name[1:] + "Action"
    iface_audit = (f"  {audit_action_field}: string;") if audit_req else ""
    payload_lines = nl.join(f"        {f.name}: data.{f.name}," for f in req)
    audit_payload_lines = (
        nl.join(
            f"        {f.name}: "
            + (
                "data." + audit_action_field
                if f.type == "String" and f.name in ("action", "kind")
                else default_literal(f)
            )
            + ","
            for f in audit_req[:3]
        )
        if audit_req
        else "        at: new Date(),"
    )
    ref = f"""{_client_import()}
export async function create{_camel(m.name)}WithAudit(data: {{
{iface_lines}
{iface_audit}
}}) {{
  return prisma.$transaction(async (tx) => {{
    const created = await tx.{_snake_prop(m.name)}.create({{
      data: {{
{payload_lines}
      }},
    }});
    await tx.{_snake_prop(audit.name)}.create({{
      data: {{
{audit_payload_lines}
      }},
    }});
    return created;
  }});
}}
"""
    instruction = (
        f"Create a {humanize_model(m.name)} and write a corresponding "
        f"{humanize_model(audit.name)} record in one atomic $transaction - "
        "if the audit write fails, the primary create must roll back."
    )
    return GeneratedTask(
        "nested_write",
        "T6_nested_writes_txn",
        instruction,
        ref,
        persona,
        ["interactive_transaction", "audit_trail"],
    )


# --- T7: aggregations -----------------------------------------------------------
def gen_aggregate(s: SampledSchema, rng: random.Random, persona: str = "pm") -> GeneratedTask:
    candidates = []
    for m in s.models:
        for f in m.scalar_fields():
            if f.type in ("Int", "Float", "BigInt") and f.name not in ("id",) and "id" not in f.name.lower():
                candidates.append((m, f))
    if not candidates:
        return _fallback(s, rng, "aggregation")
    m, f = rng.choice(candidates)
    op = rng.choice(["avg", "sum", "min", "max"])
    where_hint = ""
    rel = next((rf for rf in m.relation_fields() if not rf.is_list), None)
    if rel and rng.random() < 0.4:
        where_hint = f"\n      where: {{ {rel.name}: {{ is: {{}} }} }},"
    ref = f"""{_client_import()}
export async function stats{m.name}{op.title()}() {{
  const result = await prisma.{_snake_prop(m.name)}.aggregate({{
{where_hint}      _{op}: {{ {f.name}: true }},
  }});
  return result._{op}?.{f.name};
}}
"""
    instruction = VERBALIZER.aggregation(m.name, f.name, None)
    return GeneratedTask(
        "aggregation", "T7_aggregations", instruction, ref, persona, ["aggregate", "numeric_summary"]
    )


def gen_group_by(s: SampledSchema, rng: random.Random, persona: str = "pm") -> GeneratedTask:
    candidates = []
    for m in s.models:
        group_fields = [
            f
            for f in m.scalar_fields()
            if f.type == "String"
            and f.name.lower()
            in ("status", "kind", "role", "direction", "severity", "metric", "category", "visibility")
        ]
        count_target = next(
            (f for f in m.scalar_fields() if f.type in ("Int", "Float") and f.name not in ("id",)), None
        )
        if group_fields:
            candidates.append((m, group_fields[0], count_target))
    if not candidates:
        return gen_aggregate(s, rng, persona)
    m, g, ct = rng.choice(candidates)
    sum_block = f",\n      _sum: {{ {ct.name}: true }}" if ct else ""
    ref = f"""{_client_import()}
export async function {g.name}Breakdown() {{
  return prisma.{_snake_prop(m.name)}.groupBy({{
    by: [{g.name}]{sum_block},
    _count: {{ _all: true }},
    orderBy: {{ _count: {{ {g.name}: "desc" }} }},
  }});
}}
"""
    instruction = VERBALIZER.aggregation(m.name, ct.name if ct else "count", g.name)
    return GeneratedTask(
        "aggregation", "T7_aggregations", instruction, ref, persona, ["group_by", "count", "dashboard"]
    )


# --- helpers ---------------------------------------------------------------------
def ts_type(f: Field) -> str:
    mapping = {
        "String": "string",
        "Int": "number",
        "Float": "number",
        "BigInt": "bigint",
        "Boolean": "boolean",
        "DateTime": "Date",
        "Decimal": "number",
        "Json": "unknown",
    }
    base = mapping.get(f.type, "unknown")
    return base + (" | null" if f.is_optional else "")


def default_literal(f: Field) -> str:
    if f.type == "DateTime":
        return "new Date()"
    if f.type == "String":
        return f'"{f.name}"'
    if f.type == "Boolean":
        return "false"
    return "0"


def _camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _snake(name: str) -> str:
    import re as _re

    return _re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _fallback(s: SampledSchema, rng: random.Random, kind: str) -> GeneratedTask:
    from prismacoder.synthetic.crud_gen import gen_find_many

    t = gen_find_many(s, rng)
    return GeneratedTask(
        kind,
        "T6_nested_writes_txn" if kind == "nested_write" else "T7_aggregations",
        t.instruction,
        t.reference,
        "junior",
        ["fallback"],
    )


RELATION_GENERATORS = (gen_nested_create, gen_connect_or_create)
TXN_GENERATORS = (gen_transaction_transfer, gen_multi_model_transaction)
AGG_GENERATORS = (gen_aggregate, gen_group_by)
