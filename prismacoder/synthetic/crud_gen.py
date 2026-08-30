"""CRUD + complex-read reference generators (T4 / T5).

Every generator composes the reference from the sampled schema's internal
representation, so field names/types in the emitted TypeScript always exist in
the schema. The output is a repository-function module ready for the V3/V4/V5
gates (tsc, execution, vitest).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from prismacoder.psl import Field, Model
from prismacoder.synthetic.instruction_gen import TemplateVerbalizer, humanize_field, humanize_model
from prismacoder.synthetic.schema_gen import SampledSchema

VERBALIZER = TemplateVerbalizer()

FILTERABLE_STRING = ("email", "slug", "username", "sku", "code", "number", "name", "title", "status", "kind")


@dataclass
class GeneratedTask:
    task_type: str  # crud | complex_read | ...
    category: str  # T4_client_crud etc.
    instruction: str
    reference: str  # TypeScript artifact
    persona: str
    style_tags: list[str]


def _client_import() -> str:
    return (
        'import { PrismaClient } from "../app/generated/prisma";\n\n' "const prisma = new PrismaClient();\n"
    )


def _ts_type_of(f: Field) -> str:
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
    suffix = " | null" if f.is_optional else ""
    return base + suffix


def _first_filterable(m: Model, rng: random.Random) -> Field | None:
    cands = [
        f
        for f in m.scalar_fields()
        if f.name.lower() in FILTERABLE_STRING or f.type in ("String", "Boolean", "Int")
    ]
    return rng.choice(cands) if cands else None


def _order_field(m: Model, rng: random.Random) -> Field | None:
    cands = [f for f in m.scalar_fields() if f.type == "DateTime" and f.name != "updatedAt"] or [
        f for f in m.scalar_fields() if f.type == "Int"
    ]
    return rng.choice(cands) if cands else None


# --- T4: core CRUD ------------------------------------------------------------
def gen_find_many(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    m = rng.choice(s.models)
    where_f = _first_filterable(m, rng)
    order_f = _order_field(m, rng)
    take = rng.choice([5, 10, 20, 50])

    where_block = ""
    iface_first = "  take?: number;"
    if where_f:
        op = "contains" if where_f.type == "String" else "equals"
        where_block = (
            f"\n      where: args.{where_f.name} "
            f"? {{ {where_f.name}: {{ {op}: args.{where_f.name} }} }} : undefined,"
        )
        iface_first = f"  {where_f.name}?: {_ts_type_of(where_f)};\n  take?: number;"
    order_block = ""
    if order_f:
        order_block = f'\n      orderBy: {{ {order_f.name}: "desc" }},'

    ref = f"""{_client_import()}
export interface List{m.name}Args {{
{iface_first}
}}

export async function list{m.name}(args: List{m.name}Args) {{
  return prisma.{_snake_prop(m.name)}.findMany({{{where_block}{order_block}
      take: args.take ?? {take},
  }});
}}
"""
    instruction = VERBALIZER.crud_list(
        m.name, where_f.name if where_f else None, order_f.name if order_f else None, take
    )
    return GeneratedTask(
        "crud", "T4_client_crud", instruction, ref, persona, ["filtered_list"] if where_f else ["plain_list"]
    )


def gen_find_unique(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    m = rng.choice(s.models)
    uniq = s.unique_scalars(m.name)
    if not uniq:
        return gen_find_many(s, rng, persona)
    f = rng.choice(uniq)
    ref = f"""{_client_import()}
export async function get{m.name}({f.name}: {_ts_type_of(f)}) {{
  return prisma.{_snake_prop(m.name)}.findUnique({{
    where: {{ {f.name} }},
  }});
}}
"""
    return GeneratedTask(
        "crud", "T4_client_crud", VERBALIZER.crud_get(m.name, f.name), ref, persona, ["find_unique"]
    )


def gen_create(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    m = rng.choice(s.models)
    required = s.required_scalars(m.name)
    if not required:
        return gen_find_many(s, rng, persona)
    payload = ",\n".join(f"      {f.name}: data.{f.name}," for f in required)
    iface = ",\n".join(f"  {f.name}: {_ts_type_of(f)};" for f in required)
    rel_fields = [f for f in m.relation_fields() if not f.is_list]
    fk = ""
    if rel_fields:
        fk = f"{rel_fields[0].name}Id"
        if fk in {f.name for f in m.fields} and fk not in [f.name for f in required]:
            iface += f"\n  {fk}: number;"
            payload += f"\n      {fk}: data.{fk},"
    ref = f"""{_client_import()}
export interface Create{m.name}Input {{
{iface}
}}

export async function create{m.name}(data: Create{m.name}Input) {{
  return prisma.{_snake_prop(m.name)}.create({{
    data: {{
{payload}
    }},
  }});
}}
"""
    fields = [f.name for f in required] + (
        [f"{fk}"] if rel_fields and f"{fk}" in {f.name for f in m.fields} else []
    )
    return GeneratedTask(
        "crud", "T4_client_crud", VERBALIZER.crud_create(m.name, fields), ref, persona, ["create"]
    )


def gen_update(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    m = rng.choice(s.models)
    uniq = s.unique_scalars(m.name)
    updatable = [f for f in m.scalar_fields() if "@id" not in f.attributes and f.name not in ("createdAt",)]
    if not uniq or not updatable:
        return gen_find_many(s, rng, persona)
    key = rng.choice(uniq)
    fields = rng.sample(updatable, min(len(updatable), rng.randint(1, 3)))
    payload = ",\n".join(f"      {f.name}: patch.{f.name}," for f in fields)
    iface = ",\n".join(f"  {f.name}{'?:' if f.is_optional else ':'} {_ts_type_of(f)};" for f in fields)
    ref = f"""{_client_import()}
export interface Update{m.name}Patch {{
{iface}
}}

export async function update{m.name}({key.name}: {_ts_type_of(key)}, patch: Update{m.name}Patch) {{
  return prisma.{_snake_prop(m.name)}.update({{
    where: {{ {key.name} }},
    data: {{
{payload}
    }},
  }});
}}
"""
    return GeneratedTask(
        "crud",
        "T4_client_crud",
        VERBALIZER.crud_update(m.name, key.name, [f.name for f in fields]),
        ref,
        persona,
        ["update"],
    )


def gen_delete(s: SampledSchema, rng: random.Random, persona: str = "junior") -> GeneratedTask:
    m = rng.choice(s.models)
    uniq = s.unique_scalars(m.name)
    if not uniq:
        return gen_find_many(s, rng, persona)
    key = rng.choice(uniq)
    ref = f"""{_client_import()}
export async function delete{m.name}({key.name}: {_ts_type_of(key)}) {{
  return prisma.{_snake_prop(m.name)}.delete({{
    where: {{ {key.name} }},
  }});
}}
"""
    return GeneratedTask(
        "crud", "T4_client_crud", VERBALIZER.crud_delete(m.name, key.name), ref, persona, ["delete"]
    )


def gen_upsert(s: SampledSchema, rng: random.Random, persona: str = "pm") -> GeneratedTask:
    m = rng.choice(s.models)
    uniq = s.unique_scalars(m.name)
    required = s.required_scalars(m.name)
    if not uniq or not required:
        return gen_find_many(s, rng, persona)
    key = rng.choice(uniq)
    payload = ",\n".join(f"        {f.name}: data.{f.name}," for f in required)
    iface = ",\n".join(f"  {f.name}: {_ts_type_of(f)};" for f in required)
    ref = f"""{_client_import()}
export interface Upsert{m.name}Input {{
{iface}
}}

export async function upsert{m.name}({key.name}: {_ts_type_of(key)}, data: Upsert{m.name}Input) {{
  return prisma.{_snake_prop(m.name)}.upsert({{
    where: {{ {key.name} }},
    update: {{ {", ".join(f"{f.name}: data.{f.name}" for f in required)} }},
    create: {{
{payload}
    }},
  }});
}}
"""
    return GeneratedTask(
        "crud", "T4_client_crud", VERBALIZER.upsert(m.name, key.name), ref, persona, ["upsert"]
    )


# --- T5: complex reads ----------------------------------------------------------
def gen_include_read(s: SampledSchema, rng: random.Random, persona: str = "pm") -> GeneratedTask:
    pairs = []
    for m in s.models:
        for rf in m.relation_fields():
            if rf.is_list or rng.random() < 0.5:
                pairs.append((m, rf))
    if not pairs:
        return gen_find_many(s, rng, persona)
    m, rf = rng.choice(pairs)
    order_f = _order_field(m, rng)
    take = rng.choice([10, 20, 50])
    order_block = f'\n      orderBy: {{ {order_f.name}: "desc" }},' if order_f else ""
    ref = f"""{_client_import()}
export async function list{m.name}With{_singular(rf.type).title()}() {{
  return prisma.{_snake_prop(m.name)}.findMany({{
    include: {{ {rf.name}: true }},{order_block}
    take: {take},
  }});
}}
"""
    instruction = VERBALIZER.complex_read(
        m.name, rf.type, order_field=order_f.name if order_f else None, take=take
    )
    return GeneratedTask(
        "complex_read", "T5_complex_reads", instruction, ref, persona, ["include_projection"]
    )


def gen_select_projection(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    m = rng.choice(s.models)
    scalars = m.scalar_fields()
    if len(scalars) < 2:
        return gen_find_many(s, rng, persona)
    picked = rng.sample(scalars, min(len(scalars), rng.randint(2, 4)))
    proj = ",\n".join(f"      {f.name}: true," for f in picked)
    ref = f"""{_client_import()}
export async function project{m.name}() {{
  return prisma.{_snake_prop(m.name)}.findMany({{
    select: {{
{proj}
    }},
  }});
}}
"""
    instruction = (
        f"Query {humanize_model(m.name)} returning only "
        f"{', '.join(humanize_field(f.name) for f in picked)} - use a select projection, "
        "not fetch-then-filter."
    )
    return GeneratedTask(
        "complex_read",
        "T5_complex_reads",
        instruction,
        ref,
        persona,
        ["select_projection", "n_plus_one_avoided"],
    )


def gen_cursor_pagination(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    m = rng.choice(s.models)
    key = s.unique_scalars(m.name)
    if not key:
        return gen_include_read(s, rng, persona)
    k = rng.choice(key)
    ref = f"""{_client_import()}
export async function page{m.name}(cursor: {_ts_type_of(k)} | undefined, take: number = 20) {{
  return prisma.{_snake_prop(m.name)}.findMany({{
    take,
    skip: cursor ? 1 : 0,
    cursor: {{ {k.name}: cursor as never }},
    orderBy: {{ {k.name}: "asc" }},
  }});
}}
"""
    instruction = (
        f"Implement cursor-based pagination over {humanize_model(m.name)} keyed on "
        f"{humanize_field(k.name)} with page size {20}. Use cursor + skip 1 semantics."
    )
    return GeneratedTask("complex_read", "T5_complex_reads", instruction, ref, persona, ["cursor_pagination"])


# --- helpers ----------------------------------------------------------------------
def id_scalar_type(s: SampledSchema, model_name: str) -> str:
    """'number' | 'string' - the TS type of a model's scalar id (V3-safe)."""
    m = s.model(model_name)
    for f in (m.fields if m else []):
        if "@id" in f.attributes:
            return "string" if f.type == "String" else "number"
    return "number"


def _snake_prop(model_name: str) -> str:
    """Model name -> prisma client property (camelCase)."""
    return model_name[:1].lower() + model_name[1:]


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") else name


CRUD_GENERATORS = (gen_find_many, gen_find_unique, gen_create, gen_update, gen_delete, gen_upsert)
READ_GENERATORS = (gen_include_read, gen_select_projection, gen_cursor_pagination)
