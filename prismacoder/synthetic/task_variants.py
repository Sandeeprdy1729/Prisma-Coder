"""Task variants (T1, T2, T8-T12 + auxiliary SFT variants).

Contents:
- ``BugInjector``      : 10 controlled bug patterns (E1-E10 aligned) producing
                         bug-fix tasks and hard negatives / DPO rejected arms.
- ``gen_nl2prisma``    : T1 green-field schema from a verbalized description.
- ``gen_schema_edit``  : T2 edit pairs from the perturbation engine.
- ``gen_completion``   : prefix-code-completion variant.
- ``gen_repo_completion`` / ``gen_cross_file``: T8 / T10 multi-file context.
- ``gen_performance``  : T11 N+1 avoidance.
- ``gen_security``     : T12 raw-SQL parameterization + tenant scoping.
- ``gen_nl_explain``   : Prisma -> NL (auxiliary).
- ``gen_qa``           : conceptual QA bank (auxiliary, coverage for T10-T12 docs mix).
- ``gen_chain_edit``   : 2-3 sequential edit requests over one schema.
- ``make_preference_pair`` / ``make_reasoning``: DPO + plan-span records.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any

from prismacoder.synthetic.crud_gen import GeneratedTask, _snake_prop
from prismacoder.synthetic.instruction_gen import TemplateVerbalizer, humanize_model
from prismacoder.synthetic.schema_gen import SampledSchema

VERBALIZER = TemplateVerbalizer()


# ============================================================================
# Bug injection (bug-fix tasks + hard negatives)
# ============================================================================
@dataclass(frozen=True)
class BugPattern:
    code: str
    error_class: str  # E1-E10
    diagnostic: str  # the mechanical diagnostic shown to the model
    severity_hint: str


BUG_PATTERNS: tuple[BugPattern, ...] = (
    BugPattern(
        "wrong_field",
        "E4",
        "TS2339: Property does not exist on type. " "The referenced field is not part of the model.",
        "compile",
    ),
    BugPattern(
        "missing_await",
        "E5",
        "The promise is not awaited; the function returns a Promise, " "not the row. Callers read undefined.",
        "runtime",
    ),
    BugPattern(
        "inverted_order",
        "E6",
        "Unit tests expect the newest record first, but the list " "comes back oldest-first.",
        "semantics",
    ),
    BugPattern(
        "missing_where",
        "E6",
        "updateMany/deleteMany affected every row instead of one - " "missing where clause.",
        "semantics",
    ),
    BugPattern(
        "wrong_include",
        "E4",
        "TS2769: include path does not match the schema's relation " "field name.",
        "compile",
    ),
    BugPattern(
        "n_plus_one",
        "E9",
        "Query-count probe fired: one query per row inside a loop. " "Fetch relations with include instead.",
        "efficiency",
    ),
    BugPattern(
        "sql_injection",
        "E8",
        "Security probe: $queryRawUnsafe interpolates user input. "
        "Use a tagged template with $queryRaw or parameterize.",
        "security",
    ),
    BugPattern(
        "no_transaction",
        "E6",
        "Two writes are not atomic; a crash between them leaves " "inconsistent state. Wrap in $transaction.",
        "semantics",
    ),
    BugPattern(
        "null_assertion",
        "E4",
        "TS2531: Object is possibly 'null'. The lookup can return " "null; handle it.",
        "compile",
    ),
    BugPattern(
        "unique_violation",
        "E5",
        "P2002: Unique constraint failed on create. Use upsert or " "catch P2002.",
        "runtime",
    ),
)


class BugInjector:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def inject(self, reference: str, schema_models: list[str]) -> tuple[str, BugPattern] | None:
        """Apply ONE controlled bug to a reference artifact. Returns (buggy, pattern)."""
        patterns = list(BUG_PATTERNS)
        self.rng.shuffle(patterns)
        for p in patterns:
            out = self._apply(reference, p, schema_models)
            if out is not None and out != reference:
                return out, p
        return None

    def _apply(self, code: str, p: BugPattern, models: list[str]) -> str | None:
        if p.code == "wrong_field":
            m = re.search(r"(?:findMany|findUnique|create|update)\s*\(", code)
            if not m:
                return None
            return (
                code.replace("take: ", "taken: ", 1)
                if "take: " in code
                else re.sub(r"(orderBy: \{ \w+:)", r"orderBy: { orderAtX:", code, count=1)
            )

        if p.code == "missing_await":
            m = re.search(r"return prisma\.\w+\.\w+\(", code)
            if not m:
                return None
            return code.replace("return prisma.", "return void prisma.", 1)

        if p.code == "inverted_order":
            return (
                re.sub(r'(orderBy: \{ \w+: )"desc"', r'\1"asc"', code)
                if '"desc"' in code
                else re.sub(r'(orderBy: \{ \w+: )"asc"', r'\1"desc"', code)
            )

        if p.code == "missing_where":
            return (
                re.sub(r"(deleteMany\(\{\n)(\s*where: \{[^}]*\},\n)?", r"\1", code, count=1)
                if "deleteMany" in code
                else (
                    re.sub(r"(updateMany\(\{\n)(\s*where: \{[^}]*\},\n)?", r"\1", code, count=1)
                    if "updateMany" in code
                    else None
                )
            )

        if p.code == "wrong_include":
            m = re.search(r"include: \{ (\w+): true \}", code)
            if not m or len(models) < 2:
                return None
            wrong = next((x for x in models if x.lower() != m.group(1)), None)
            return (
                code.replace(m.group(0), f"include: {{ {wrong[:1].lower() + wrong[1:] + 'X'}: true }}", 1)
                if wrong
                else None
            )

        if p.code == "n_plus_one":
            if "for (const" in code or "findMany" not in code:
                return None
            inject = (
                "  for (const row of rows) {\n"
                "    row.related = await prisma."
                + _snake_prop(models[0])
                + ".findUnique({ where: { id: row.id } });\n  }\n"
            )
            return (
                code.replace("return prisma.", "const rows = await prisma.", 1) + inject
                if "const rows" not in code
                else None
            )

        if p.code == "sql_injection":
            if "$queryRaw" not in code:
                return None
            return re.sub(
                r"\$queryRaw`SELECT.*?`",
                r"$queryRawUnsafe(`SELECT 1 FROM User WHERE id = ${userId}`)",
                code,
                flags=re.S,
            )

        if p.code == "no_transaction":
            if "$transaction" in code or "increment" not in code:
                return None
            return re.sub(r"prisma\.\$(transaction)", r"prisma.\1", code)  # no-op guard

        if p.code == "null_assertion":
            m = re.search(r"(const \w+ = await prisma\.\w+\.findUnique\([^)]*\);)", code)
            if not m:
                return None
            return (
                code.replace(m.group(1), m.group(1) + "\n  return row!.id;", 1)
                if "row!" not in code
                else None
            )

        if p.code == "unique_violation":
            m = re.search(r"(create\(\{\s*data: \{)", code)
            if not m or "@unique" not in " ".join(models):
                return None
            return code.replace(m.group(1), m.group(1), 1)  # structural no-op: pattern kept rare
        return None


def gen_bug_fix(
    s: SampledSchema, rng: random.Random, reference_task: GeneratedTask, persona: str = "junior"
) -> GeneratedTask | None:
    inj = BugInjector(rng)
    bug = inj.inject(reference_task.reference, s.model_names())
    if bug is None:
        return None
    buggy, pattern = bug
    instruction = (
        f"The following {reference_task.task_type} implementation fails:\n\n"
        f"```\n{buggy.strip()}\n```\n\n"
        f"Diagnostic: {pattern.diagnostic}\n\n"
        "Return the corrected implementation only."
    )
    return GeneratedTask(
        "bug_fix",
        "T9_bug_fix",
        instruction,
        reference_task.reference,
        persona,
        ["bug_fix", f"class_{pattern.error_class}"],
    )


# ============================================================================
# T1 / T2 / chain edits
# ============================================================================
def describe_schema_nl(s: SampledSchema) -> str:
    """Green-field verbalization of a sampled schema (T1)."""
    lines = [
        f"Design a Prisma schema for a {s.domain.replace('_', ' ')} application "
        f"backed by {('PostgreSQL' if s.provider in ('postgresql', 'postgres') else s.provider.title())}."
    ]
    for m in s.models:
        rel_targets = sorted({rf.type for rf in m.relation_fields()})
        scalars = ", ".join(f.name for f in m.scalar_fields()[:6])
        desc = f"- {m.name}: fields {scalars}"
        if rel_targets:
            desc += "; related to " + ", ".join(rel_targets)
        lines.append(desc)
    if s.enums:
        lines.append("Use these enums: " + ", ".join(e.name for e in s.enums) + ".")
    if s.multi_tenant:
        lines.append(f"Scope records by {s.tenant_field} for multi-tenancy.")
    if s.soft_delete:
        lines.append("Records support soft delete via deletedAt.")
    return "\n".join(lines)


def gen_nl2prisma(s: SampledSchema, rng: random.Random, persona: str = "pm") -> GeneratedTask:
    return GeneratedTask(
        "nl2prisma",
        "T1_schema_greenfield",
        describe_schema_nl(s),
        s.render(),
        persona,
        ["greenfield_schema", f"key_{s.key_style}"],
    )


def gen_schema_edit(
    before: str, after: str, change_desc: str, model_name: str, persona: str = "pm"
) -> GeneratedTask:
    instruction = (
        f"Here is the current schema excerpt for {humanize_model(model_name)}:\n\n"
        f"```prisma\n{before.strip()}\n```\n\n"
        f"Change request: {change_desc}.\n\nReturn the complete updated schema.prisma."
    )
    return GeneratedTask(
        "schema_edit", "T2_schema_edit", instruction, after.strip(), persona, ["schema_edit"]
    )


def gen_chain_edit(s: SampledSchema, rng: random.Random, n_steps: int = 2) -> list[GeneratedTask]:
    """Sequential edits: step k receives the result of step k-1 as context."""
    from prismacoder.synthetic.schema_gen import PerturbationEngine

    eng = PerturbationEngine(rng)
    tasks: list[GeneratedTask] = []
    current = s.render()
    current_s = s
    for i in range(n_steps):
        before, after, meta = eng.perturb(current_s)
        desc = "; ".join(meta["ops"])
        ctx = current if i == 0 else tasks[-1].reference
        t = GeneratedTask(
            "schema_edit",
            "T2_schema_edit",
            f"Apply change step {i + 1} of {n_steps} to the schema below: {desc}.\n\n"
            f"```prisma\n{ctx.strip()}\n```\n\nReturn the complete updated schema.prisma.",
            after.strip(),
            "pm",
            ["chain_edit", f"step_{i + 1}"],
        )
        tasks.append(t)
        current = after
        try:
            from prismacoder.psl import parse_schema as _ps

            parsed = _ps(after)
            current_s = SampledSchema(
                domain=current_s.domain,
                provider=current_s.provider,
                models=parsed.models,
                enums=parsed.enums,
                key_style=current_s.key_style,
                soft_delete=current_s.soft_delete,
                multi_tenant=current_s.multi_tenant,
                tenant_field=current_s.tenant_field,
                audit_fields=current_s.audit_fields,
            )
        except Exception:
            break
    return tasks


# ============================================================================
# T8 / T10: repository-aware + cross-file
# ============================================================================
def gen_repo_completion(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    m = rng.choice(s.models)
    scalars = m.scalar_fields()[:4]
    repo_stub = (
        "// repositories/" + _snake(m.name) + ".repository.ts\n"
        'import { PrismaClient } from "../app/generated/prisma";\n\n'
        "export class " + m.name + "Repository {\n"
        "  constructor(private readonly prisma: PrismaClient) {}\n\n"
        "  async findById(id: number) {\n"
        f"    return this.prisma.{_snake_prop(m.name)}.findUnique({{ where: {{ id }} }});\n"
        "  }\n\n"
        "  // TODO: implement below\n"
        "  async listRecent(take: number) {\n"
        "    // your implementation here\n"
        "  }\n"
        "}\n"
    )
    ref = repo_stub.replace(
        "  // TODO: implement below\n  async listRecent(take: number) {\n    // your implementation here\n  }",
        "  async listRecent(take: number) {\n"
        f"    return this.prisma.{_snake_prop(m.name)}.findMany({{\n"
        + (f'      orderBy: {{ {scalars[-1].name}: "desc" }},\n' if scalars else "")
        + "      take,\n    });\n  }",
    )
    instruction = (
        f"Implement the `listRecent` method of the repository class below, following the "
        f"file's existing conventions ({m.name} model, Prisma Client injected as `this.prisma`).\n\n"
        f"```ts\n{repo_stub.strip()}\n```"
    )
    return GeneratedTask(
        "repo_completion",
        "T8_repo_completion",
        instruction,
        ref,
        persona,
        ["repository_layer", "convention_following"],
    )


def gen_cross_file(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    m = rng.choice(s.models)
    fields = m.scalar_fields()[:3]
    types = {"String": "string", "Int": "number", "Float": "number", "Boolean": "boolean", "DateTime": "Date"}
    iface = "\n".join(f"  {f.name}: {types.get(f.type, 'unknown')};" for f in fields)
    req = s.required_scalars(m.name)
    payload = ",\n".join(f"      {f.name}: body.{f.name}," for f in req[:3]) or "      createdAt: new Date(),"
    context = (
        "// lib/validations.ts\n"
        f"export interface {m.name}Dto {{\n{iface}\n}}\n\n"
        "// app/api/" + _snake(m.name) + "/route.ts (Next.js App Router)\n"
        f'import {{ NextResponse }} from "next/server";\n'
        'import { prisma } from "@/lib/prisma";\n\n'
        "export async function POST(request: Request) {\n"
        f"  const body = (await request.json()) as {m.name}Dto;\n"
        "  // implement: persist via Prisma and return the created row\n"
        "}\n"
    )
    ref = context.replace(
        "  // implement: persist via Prisma and return the created row",
        f"  const created = await prisma.{_snake_prop(m.name)}.create({{\n    data: {{\n{payload}\n    }},\n  }});\n"
        "  return NextResponse.json(created, { status: 201 });",
    )
    instruction = (
        f"The data layer defines `{m.name}Dto` and the route handler is a stub. "
        f"Implement POST so it validates nothing twice (trust the DTO), persists via "
        f"Prisma, and returns 201 with the created {humanize_model(m.name)[:-1]}.\n\n"
        f"```ts\n{context.strip()}\n```"
    )
    return GeneratedTask(
        "cross_file", "T10_cross_file", instruction, ref, persona, ["multi_file", "api_coherence"]
    )


# ============================================================================
# T11 / T12
# ============================================================================
def gen_performance(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    parent, child = None, None
    for m in s.models:
        for rf in m.relation_fields():
            if rf.is_list:
                parent, child = m, rf.type
                break
        if parent:
            break
    if parent is None:
        return gen_cross_file(s, rng, persona)
    bad = (
        f"const parents = await prisma.{_snake_prop(parent.name)}.findMany();\n"
        "for (const p of parents) {\n"
        f"  p.children = await prisma.{_snake_prop(child)}.findMany();\n"
        "}\nreturn parents;"
    )
    good = (
        f"return prisma.{_snake_prop(parent.name)}.findMany({{\n"
        f"  include: {{ {next(rf.name for rf in parent.relation_fields() if rf.is_list)}: true }},\n}});"
    )
    ref = good
    instruction = (
        "The query-per-row implementation below trips the N+1 probe (one query per row). "
        f"Rewrite it as a single findMany using include.\n\n```ts\n{bad.strip()}\n```"
    )
    return GeneratedTask(
        "performance",
        "T11_performance",
        instruction,
        ref,
        persona,
        ["n_plus_one_avoided", "include_pushdown"],
    )


def gen_security(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    tenant_f = s.tenant_field or "organizationId"
    m = next((x for x in s.models if any(f.name == tenant_f for f in x.scalar_fields())), s.models[0])
    filterable = next((f for f in m.scalar_fields() if f.type == "String"), None)
    filter_name = filterable.name if filterable else "id"
    bad = (
        f"const rows = await prisma.$queryRawUnsafe(\n"
        f"  `SELECT * FROM {_snake(m.name)}s WHERE {tenant_f} = '${{tenantId}}' AND {filter_name} = '${{q}}'`\n"
        ");"
    )
    good = (
        f"export async function listScoped{_camel(m.name)}(tenantId: string, q: string) {{\n"
        f"  const rows = await prisma.{_snake_prop(m.name)}.findMany({{\n"
        f"    where: {{ {tenant_f}: tenantId, {filter_name}: {{ contains: q }} }},\n"
        "  });\n"
        "  return rows;\n"
        "}"
    )
    instruction = (
        f"Security review flagged the raw query below (string interpolation -> SQL injection, "
        f"and it must always be scoped to a tenant via {tenant_f}). Rewrite it safely with "
        f"the Prisma Client API.\n\n```ts\n{bad.strip()}\n```"
    )
    return GeneratedTask(
        "security",
        "T12_security",
        instruction,
        good,
        persona,
        ["parameterized", "tenant_scoped", "no_raw_interpolation"],
    )


# ============================================================================
# Auxiliary variants: Prisma->NL, QA, completion
# ============================================================================
def gen_nl_explain(task: GeneratedTask, persona: str = "junior") -> GeneratedTask:
    """Prisma -> NL: explain what an artifact does (mechanical summary of the code)."""
    ops = sorted(
        set(
            re.findall(
                r"\.(findMany|findUnique|findFirst|create|update|upsert|delete|aggregate|groupBy|\$transaction)\(",
                task.reference,
            )
        )
    )
    includes = re.findall(r"include: \{ (\w+): true \}", task.reference)
    explanation = (
        f"This module calls {', '.join(ops) if ops else 'Prisma Client read operations'} "
        + (f" and includes related {', '.join(includes)}" if includes else "")
        + ". It is exported as typed async functions over the Prisma Client singleton."
    )
    return GeneratedTask(
        "nl_explain",
        "T10_cross_file",
        f"Explain in one paragraph what this Prisma Client module does. Do not invent "
        f"behavior that is not in the code.\n\n```ts\n{task.reference.strip()}\n```",
        explanation,
        persona,
        ["prisma_to_nl"],
    )


QA_BANK: tuple[tuple[str, str], ...] = (
    (
        "What does prisma validate check that prisma format does not?",
        "format normalizes layout and whitespace only; validate checks semantic integrity: paired "
        "relation fields, valid attribute arguments, enum references, and datasource consistency.",
    ),
    (
        "When must you use connectOrCreate instead of connect?",
        "When the related record may not exist yet: connect throws if the where matches nothing, while "
        "connectOrCreate falls back to create with the provided payload.",
    ),
    (
        "What is the difference between $transaction with an array and with a function?",
        "The array form runs sequential queries with a shared timeout; the interactive form takes an async "
        "function receiving tx and supports conditional logic, with the timeout counting against the whole callback.",
    ),
    (
        "Why can a Prisma relation require two fields in the schema?",
        "Prisma requires both a scalar foreign-key field and a relation field pointing at the other model; "
        "the relation field is virtual (not stored) and references the FK via @relation(fields: [...]).",
    ),
    (
        "What error does Prisma throw on a unique constraint violation, and how do you handle it?",
        "P2002 (PrismaClientKnownRequestError). Catch it and branch on e.code, or prevent it with upsert.",
    ),
    (
        "How does cursor pagination differ from skip/take pagination?",
        "cursor uses a stable key (usually a unique id) with skip: 1 to start after the cursor; it avoids "
        "the O(offset) scan cost and stays consistent when rows are inserted mid-list.",
    ),
    (
        "When does groupBy require `orderBy` to be a _count/_aggregate form?",
        "When ordering by an aggregate instead of the grouped-by columns, orderBy must address the aggregate, "
        'e.g. orderBy: { _count: { status: "desc" } }.',
    ),
    (
        "What is the default cascade behavior of a required relation in Prisma?",
        "For required relations the default referential action onDelete is Restrict (with foreignKeys "
        "enforced by the database); optional relations default to SetNull.",
    ),
    (
        "Why should $queryRawUnsafe be avoided in application code?",
        "It interpolates values outside the driver's parameterization, enabling SQL injection; $queryRaw "
        "tagged templates parameterize automatically and should be the default, with executeRaw for writes.",
    ),
    (
        "What does @@index give you that @unique does not?",
        "@@index creates a non-unique secondary index for read performance; @unique enforces uniqueness "
        "(which also creates an index but with a different semantic and error behavior).",
    ),
    (
        "How do implicit and explicit many-to-many relations differ?",
        "Implicit m-n is managed by Prisma with an auto-created join table (not queryable directly); explicit "
        "m-n models the join table as a real model, allowing extra columns and direct queries.",
    ),
    (
        "What is the difference between findUnique and findFirst?",
        "findUnique requires a filter on a unique field(s) and returns at most one row; findFirst accepts any "
        "filter with orderBy and returns the first match.",
    ),
    (
        "When is updateMany preferred over update?",
        "When the filter targets non-unique fields or you want bulk semantics with a count result; update "
        "requires a unique where and throws P2025 when the row is missing.",
    ),
    (
        "What does P2025 signal?",
        "An operation expected a row that does not exist - e.g. update/delete/findUniqueOrThrow on a missing "
        "record, or a required relation connect to a non-existent id.",
    ),
    (
        "How do you run a raw query with parameters safely?",
        "Use $queryRaw with a tagged template: SELECT * FROM t WHERE id = ${id}. Prisma turns interpolations "
        "into prepared-statement parameters.",
    ),
    (
        "What is the purpose of the @relation attribute's fields/references pair?",
        "fields lists the local scalar FK columns; references lists the referenced fields on the target model; "
        "together they define how the join is materialized in the database.",
    ),
    (
        "Why does Prisma generate a nullable type for optional relations?",
        "An optional relation (no required FK) may be absent at runtime, so generated types reflect T | null "
        "unless the FK is non-nullable.",
    ),
    (
        "How do you atomically decrement a counter?",
        "Pass an increment/decrement object in data, e.g. data: { stock: { decrement: qty } } - the database "
        "performs the arithmetic server-side, avoiding read-modify-write races.",
    ),
    (
        "What is the effect of deleting a model with onDelete: Cascade?",
        "The database deletes dependent rows (via FK cascade) when the referenced row is deleted; useful for "
        "child records, dangerous for unlabeled destructive migrations.",
    ),
    (
        "How do you type the payload of findUnique with include?",
        "Use Prisma.validator or Prisma.UserGetPayload<{ include: { posts: true } }> to derive the exact "
        "return type from the query shape.",
    ),
    (
        "What is the difference between $executeRaw and $executeRawUnsafe?",
        "$executeRaw is the parameterized tagged-template form; $executeRawUnsafe takes a string and raw "
        "params, skipping template safety - use only for dynamically built statements after validation.",
    ),
    (
        "How does Prisma's `select` interact with `include`?",
        "include is sugar for select on a relation plus all scalars; a select that mentions relation fields "
        "fully replaces include - they are not merged when both are present.",
    ),
    (
        "When should you model soft delete instead of relying on database delete?",
        "When audit/recovery requirements exist: add deletedAt DateTime?, filter reads (middleware or every "
        "query), and treat hard delete as a background/admin operation.",
    ),
    (
        "What is Prisma's default id strategy and when would you change it?",
        "Default is autoincrement Int (or Int @default(autoincrement())); switch to uuid()/cuid() strings for "
        "distributed systems, URL-safe public ids, or merge-safe offline writes.",
    ),
)


def gen_qa(rng: random.Random) -> GeneratedTask:
    q, a = rng.choice(QA_BANK)
    return GeneratedTask("qa", "T10_cross_file", q, a, "none", ["qa", "docs"])


def gen_completion(s: SampledSchema, rng: random.Random, persona: str = "expert") -> GeneratedTask:
    from prismacoder.synthetic.crud_gen import gen_create, gen_include_read

    base = gen_create(s, rng, persona) if rng.random() < 0.5 else gen_include_read(s, rng, persona)
    lines = base.reference.splitlines()
    cut = max(3, len(lines) // 2)
    prefix = "\n".join(lines[:cut])
    suffix = "\n".join(lines[cut:])
    instruction = (
        f"Complete the implementation. The file continues from the cursor marker `<CURSOR>`.\n\n"
        f"```ts\n{prefix}\n<CURSOR>\n{suffix}\n```"
    )
    return GeneratedTask(
        "completion",
        "T8_repo_completion",
        instruction,
        base.reference,
        persona,
        ["infill", "code_completion"],
    )


# ============================================================================
# Preference / reasoning pairs (DPO + plan-span)
# ============================================================================
def make_preference_pair(task: GeneratedTask, buggy: str, basis: str) -> dict[str, Any]:
    return {
        "prompt": task.instruction,
        "chosen": task.reference,
        "rejected": buggy,
        "basis": basis,  # e.g. "V3_tsc differential" / "V5_vitest differential"
        "task_type": task.task_type,
        "category": task.category,
    }


def make_reasoning(task: GeneratedTask, plan_span: str) -> dict[str, Any]:
    return {
        "plan_span": plan_span,
        "artifact": task.reference,
        "task_type": task.task_type,
        "category": task.category,
    }


def derive_plan_span(task: GeneratedTask) -> str:
    """Mechanical plan sketch from the task shape (no LLM)."""
    steps = ["Read the schema and identify the target model and its relations."]
    if task.task_type in ("crud", "complex_read"):
        steps.append("Choose the client operation matching the request (findMany/findUnique/create/update).")
        steps.append("Shape the payload: where/orderBy/take or data; prefer include/select pushdown.")
    elif task.task_type == "nested_write":
        steps.append("Locate the relation field and decide nested create vs connect vs connectOrCreate.")
        steps.append("If multiple writes must be atomic, wrap them in an interactive $transaction.")
    elif task.task_type == "aggregation":
        steps.append("Pick aggregate (scalar summary) vs groupBy (per-bucket summary).")
        steps.append("Wire orderBy/_count semantics and shape the return type.")
    elif task.task_type == "bug_fix":
        steps.append("Reproduce the diagnostic; map it to the failing gate (compile/runtime/semantics).")
        steps.append("Apply the minimal correction that satisfies the request and the schema.")
    elif task.task_type in ("schema_edit", "nl2prisma", "migration"):
        steps.append("Diff the requested change against the current models (fields, keys, relations).")
        steps.append("Emit valid PSL; flag destructive operations explicitly.")
    steps.append(
        "Verify: field names exist in the schema, nullability handled, no N+1, no raw interpolation."
    )
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _camel(name: str) -> str:
    return name[:1].lower() + name[1:]


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


VARIANT_GENERATORS = ()
