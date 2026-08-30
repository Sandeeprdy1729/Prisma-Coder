"""Instruction verbalization (report 9.7: 'teachers write questions, mechanics
write answers').

``TemplateVerbalizer`` produces deterministic, persona-varied requests that
mention only schema-visible symbols - the mechanical baseline every record
passes through. ``TeacherHook`` is the optional upgrade path where an open
teacher model (Qwen3-235B-A22B class / DeepSeek-V3 class per the report)
rewrites requests with richer phrasing; the hook is called with the reference
solution and schema and must return an instruction string. Records produced by
a teacher are tagged ``meta.tags += ['teacher-verbalized']``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from prismacoder.constants import PERSONAS

TeacherHook = Callable[[str, str], str]  # (template_instruction, schema_psl) -> instruction


def _snake_to_words(name: str) -> str:
    return name.replace("_", " ").strip()


def humanize_model(model_name: str) -> str:
    """'OrderItem' -> 'order items', 'User' -> 'users'."""
    words = re.findall(r"[A-Z][a-z0-9]*", model_name) or [model_name]
    head = " ".join(w.lower() for w in words)
    return head + "s" if not head.endswith("s") else head


def humanize_field(field_name: str) -> str:
    return _snake_to_words(re.sub(r"(?<!^)(?=[A-Z])", " ", field_name).lower())


@dataclass
class PersonaPhrasing:
    junior: str
    pm: str
    expert: str


class TemplateVerbalizer:
    """Persona-varied request templates. Every template references only
    symbols passed in explicitly (models/fields) - no hallucinated names."""

    @staticmethod
    def phrasings(action: str, object_desc: str, detail: str) -> PersonaPhrasing:
        obj = object_desc
        return PersonaPhrasing(
            junior=f"Write a Prisma Client function that {action} {obj}. {detail}".strip(),
            pm=f"We need the data layer to {action} {obj}. {detail}".strip(),
            expert=f"Prisma Client: {action} {obj}. {detail}".strip(),
        )

    def crud_list(
        self, model: str, where_field: str | None, order_field: str | None, take: int | None
    ) -> str:
        obj = humanize_model(model)
        parts = []
        if where_field:
            parts.append(f"filtered by {humanize_field(where_field)}")
        if order_field:
            parts.append(
                f"newest first by {humanize_field(order_field)}"
                if order_field.lower().endswith("at")
                else f"ordered by {humanize_field(order_field)}"
            )
        if take:
            parts.append(f"limited to {take} results")
        detail = "Use findMany with a where clause." if where_field else "Use findMany."
        p = self.phrasings("list", obj, detail)
        tail = (" " + ", ".join(parts) + ".") if parts else ""
        return p.junior + tail

    def crud_get(self, model: str, unique_field: str) -> str:
        p = self.phrasings(
            "fetch a single",
            humanize_model(model[:-1] if model.endswith("s") else model),
            f"Look it up by {humanize_field(unique_field)} with findUnique.",
        )
        return p.junior

    def crud_create(self, model: str, fields: list[str]) -> str:
        field_list = ", ".join(humanize_field(f) for f in fields)
        p = self.phrasings(
            "create a new",
            humanize_model(model[:-1] if model.endswith("s") else model),
            f"Accept {field_list} as input and use create.",
        )
        return p.junior

    def crud_update(self, model: str, unique_field: str, fields: list[str]) -> str:
        field_list = ", ".join(humanize_field(f) for f in fields)
        p = self.phrasings(
            "update an existing",
            humanize_model(model[:-1] if model.endswith("s") else model),
            f"Find it by {humanize_field(unique_field)}, update {field_list}, use update.",
        )
        return p.junior

    def crud_delete(self, model: str, unique_field: str) -> str:
        p = self.phrasings(
            "remove a",
            humanize_model(model[:-1] if model.endswith("s") else model),
            f"Delete it by {humanize_field(unique_field)} with delete.",
        )
        return p.junior

    def upsert(self, model: str, unique_field: str) -> str:
        p = self.phrasings(
            "create or update a",
            humanize_model(model[:-1] if model.endswith("s") else model),
            f"Match on {humanize_field(unique_field)} with upsert.",
        )
        return p.pm

    def complex_read(
        self,
        model: str,
        include_model: str,
        *,
        order_field: str | None,
        take: int | None,
        distinct: bool = False,
    ) -> str:
        obj = humanize_model(model)
        detail = f"Include each record's related {humanize_model(include_model)} via include."
        if order_field:
            detail += f" Order by {humanize_field(order_field)} descending."
        if take:
            detail += f" Take {take}."
        if distinct:
            detail += " Apply distinct on the queried field."
        p = self.phrasings("query", obj, detail)
        return p.pm

    def nested_write(self, parent: str, child: str, mode: str) -> str:
        rel_desc = {
            "connect": f"connect an existing {humanize_model(child[:-1] if child.endswith('s') else child)}",
            "create": f"create a nested {humanize_model(child[:-1] if child.endswith('s') else child)}",
            "connectOrCreate": f"connect-or-create a {humanize_model(child[:-1] if child.endswith('s') else child)}",
        }
        p = self.phrasings(
            "create a new",
            humanize_model(parent[:-1] if parent.endswith("s") else parent),
            f"In the same write, {rel_desc.get(mode, rel_desc['create'])} using nested writes.",
        )
        return p.junior

    def transaction(self, models: list[str], amount_field: str) -> str:
        names = " and ".join(humanize_model(m[:-1] if m.endswith("s") else m) for m in models[:2])
        p = self.phrasings(
            "move value between two",
            f"{names} records atomically",
            f"Use an interactive $transaction; debit one {humanize_field(amount_field)} and credit "
            "the other, throwing on failure so both writes roll back.",
        )
        return p.expert

    def aggregation(self, model: str, agg_field: str, group_field: str | None) -> str:
        obj = humanize_model(model)
        if group_field:
            p = self.phrasings(
                "summarize",
                obj,
                f"Use groupBy on {humanize_field(group_field)} with an aggregate over "
                f"{humanize_field(agg_field)}.",
            )
        else:
            p = self.phrasings(
                "summarize", obj, f"Use aggregate to compute an average of {humanize_field(agg_field)}."
            )
        return p.pm

    def schema_edit(self, model: str, change_desc: str) -> str:
        p = self.phrasings(
            "modify the schema for",
            humanize_model(model),
            change_desc + " Return the complete updated schema.prisma.",
        )
        return p.pm

    def migration(self, change_desc: str, destructive: bool) -> str:
        risk = (
            "WARNING: this change may destroy data - label the destructive risk explicitly."
            if destructive
            else "The change must be non-destructive."
        )
        p = self.phrasings("migrate the database", "", change_desc + " Produce the migration SQL and " + risk)
        return p.pm


class TeacherVerbalizer:
    """LLM teacher adapter (offline in CI; requires teacher endpoint at build time)."""

    def __init__(self, hook: TeacherHook, name: str = "teacher"):
        self.hook = hook
        self.name = name

    def verbalize(self, template_instruction: str, schema_psl: str) -> tuple[str, str]:
        """Returns (instruction, persona_tag). Persona is 'expert' for teacher output."""
        out = self.hook(template_instruction, schema_psl)
        if not isinstance(out, str) or not out.strip():
            raise ValueError("teacher hook returned an empty instruction")
        return out.strip(), "expert"


def persona_pick(rng: Any) -> str:
    return rng.choice([p for p in PERSONAS if p != "none"] + ["expert"])


def render_persona(p: PersonaPhrasing, persona: str) -> str:
    return {"junior": p.junior, "pm": p.pm, "expert": p.expert}.get(persona, p.junior)
