"""Minimal, dependency-free PSL (Prisma Schema Language) parser + renderer.

Used by: dedup (model-graph signatures), quality filtering, the synthetic
generators, tree-edit-distance metrics, and the hallucination detector
(symbol-table grounding). Supports the Prisma v7 dialect subset that the
corpus actually contains: models, enums, composite `@@id`/`@@unique`/
`@@index`, relation attributes, views are passed through as blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

SCALAR_TYPES = {
    "String",
    "Int",
    "BigInt",
    "Float",
    "Decimal",
    "Boolean",
    "DateTime",
    "Json",
    "Bytes",
    "Char",
    "Vector",
    "Unsupported",
}

_BLOCK_RE = re.compile(r"^(model|enum|generator|datasource|view)\s+(\w+)\s*\{", re.M)


@dataclass
class Field:
    name: str
    type: str
    is_list: bool = False
    is_optional: bool = False
    attributes: list[str] = dc_field(default_factory=list)  # e.g. ["@id", "@default(cuid())"]
    default: str | None = None
    is_relation: bool = False

    def render(self) -> str:
        t = self.type + ("[]" if self.is_list else ("?" if self.is_optional else ""))
        attrs = " ".join(self.attributes)
        return f"  {self.name} {t}" + (f" {attrs}" if attrs else "")


@dataclass
class Model:
    name: str
    fields: list[Field] = dc_field(default_factory=list)
    block_attributes: list[str] = dc_field(default_factory=list)  # e.g. ["@@map(\"users\")"]
    comment: str = ""

    def field_names(self) -> set[str]:
        return {f.name for f in self.fields}

    def relation_fields(self) -> list[Field]:
        return [f for f in self.fields if f.is_relation]

    def scalar_fields(self) -> list[Field]:
        return [f for f in self.fields if not f.is_relation]

    def primary_key(self) -> Field | None:
        for f in self.fields:
            if "@id" in f.attributes:
                return f
        for ba in self.block_attributes:
            if ba.startswith("@@id"):
                return self.fields[0] if self.fields else None
        return None


@dataclass
class Enum:
    name: str
    values: list[str] = dc_field(default_factory=list)


@dataclass
class ParsedSchema:
    models: list[Model]
    enums: list[Enum]
    datasource: dict[str, str]  # provider / url
    generator: dict[str, str]
    raw: str

    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def get_model(self, name: str) -> Model | None:
        return next((m for m in self.models if m.name == name), None)

    def symbol_table(self) -> dict[str, Any]:
        """Grounded symbol table: models -> fields/enums. Powers the hallucination metric."""
        return {
            "models": {
                m.name: {
                    "fields": [f.name for f in m.fields],
                    "scalars": [f.name for f in m.scalar_fields()],
                    "relations": [f.name for f in m.relation_fields()],
                }
                for m in self.models
            },
            "enums": {e.name: list(e.values) for e in self.enums},
        }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _strip_comments(text: str) -> str:
    text = re.sub(r"^///.*$", "", text, flags=re.M)  # doc comments
    text = re.sub(r"(?m)^\s*//.*$", "", text)  # line comments
    return text


def _match_block(text: str, start: int) -> tuple[int, int]:
    """Return (body_start, body_end) indices for the {...} starting at brace in `start`."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return start + 1, i
    raise ValueError("unbalanced braces in PSL")


_FIELD_RE = re.compile(
    r"^\s{2,}(?P<name>\w+)\s+(?P<type>[\w\"\.]+)(?P<arr>\[\])?(?P<opt>\?)?" r"(?P<attrs>.*)$"
)


def _parse_field(line: str, model_names: set[str]) -> Field | None:
    line = line.rstrip()
    if not line.strip() or line.strip().startswith("@@"):
        return None
    m = _FIELD_RE.match(line)
    if not m:
        return None
    name, ftype = m.group("name"), m.group("type")
    attrs = [a.strip() for a in m.group("attrs").split() if a.strip()]
    is_relation = ftype in model_names and ftype not in SCALAR_TYPES
    default = None
    for a in attrs:
        dm = re.match(r"@default\((.+)\)$", a)
        if dm:
            default = dm.group(1)
    return Field(
        name=name,
        type=ftype,
        is_list=bool(m.group("arr")),
        is_optional=bool(m.group("opt")),
        attributes=attrs,
        default=default,
        is_relation=is_relation,
    )


def parse_models(schema_text: str) -> list[Model]:
    """Parse only the model blocks (used by hashing / dedup / TED)."""
    return parse_schema(schema_text).models


def parse_schema(schema_text: str) -> ParsedSchema:
    text = _strip_comments(schema_text)
    models: list[Model] = []
    enums: list[Enum] = []
    datasource: dict[str, str] = {}
    generator: dict[str, str] = {}

    # Pass 1: collect model names so relation fields can be detected.
    model_names: set[str] = set()
    for m in _BLOCK_RE.finditer(text):
        if m.group(1) == "model":
            model_names.add(m.group(2))

    for m in _BLOCK_RE.finditer(text):
        kind, name = m.group(1), m.group(2)
        body_start, body_end = _match_block(text, m.end() - 1)
        body = text[body_start:body_end]

        if kind in ("model", "view"):
            mdl = Model(name=name, comment=m.group(0)[:0])
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("@@"):
                    mdl.block_attributes.append(stripped.rstrip())
                    continue
                f = _parse_field(line, model_names)
                if f is not None:
                    mdl.fields.append(f)
            models.append(mdl)
        elif kind == "enum":
            values = [
                ln.strip().split()[0]
                for ln in body.splitlines()
                if ln.strip() and not ln.strip().startswith("//")
            ]
            enums.append(Enum(name=name, values=values))
        elif kind == "datasource":
            for ln in body.splitlines():
                kv = re.match(r"\s*(\w+)\s*=\s*(.+?)\s*$", ln)
                if kv:
                    datasource[kv.group(1)] = kv.group(2).strip('"')
        elif kind == "generator":
            for ln in body.splitlines():
                kv = re.match(r"\s*(\w+)\s*=\s*(.+?)\s*$", ln)
                if kv:
                    generator[kv.group(1)] = kv.group(2).strip('"')

    return ParsedSchema(
        models=models, enums=enums, datasource=datasource, generator=generator, raw=schema_text
    )


# ---------------------------------------------------------------------------
# Rendering (used by synthetic generators)
# ---------------------------------------------------------------------------


def render_field(f: Field) -> str:
    return f.render()


def render_model(m: Model) -> str:
    lines = [f"model {m.name} {{"]
    lines += [f.render() for f in m.fields]
    lines += [f"  {ba}" for ba in m.block_attributes]
    lines.append("}")
    return "\n".join(lines)


def render_enum(e: Enum) -> str:
    lines = [f"enum {e.name} {{"]
    lines += [f"  {v}" for v in e.values]
    lines.append("}")
    return "\n".join(lines)


def render_schema(
    models: list[Model], enums: list[Enum] | None = None, provider: str = "sqlite", output_engine: bool = True
) -> str:
    """Render a full schema.prisma (Prisma v7 dialect: `prisma-client` generator
    with output; the connection URL lives in prisma.config.ts, not the schema)."""
    parts: list[str] = [
        'generator client {\n  provider = "prisma-client"\n  output  = "../app/generated/prisma"\n}',
        f'datasource db {{\n  provider = "{provider}"\n}}',
    ]
    for e in enums or []:
        parts.append(render_enum(e))
    for m in models:
        parts.append(render_model(m))
    return "\n\n".join(parts) + "\n"
