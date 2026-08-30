"""Dataset augmentation (report Part 8 expansions + robustness suite).

Augmentations (all label-preserving or explicitly paired):
- ``paraphrase_persona``  - re-render the same task under a different persona.
- ``entity_rename``       - rename models/fields consistently in schema + code
                            (Dr.Spider-style robustness perturbation).
- ``schema_shuffle``      - reorder models/fields (order-invariance check).
- ``distractor_insertion``- inject an irrelevant model into the schema context.
- ``field_alias_swap``    - swap a field name for a synonymous one everywhere.

Every augmented record carries ``meta.tags += ['augmented', '<kind>']`` and is
re-verified by the same gates (augmentation never bypasses verification).
"""

from __future__ import annotations

import random
import re
from typing import Any

from prismacoder.psl import parse_schema

SYNONYMS = {
    "User": ["Account", "Member", "Profile"],
    "Order": ["Purchase", "Checkout"],
    "Product": ["Item", "Good"],
    "Post": ["Article", "Entry"],
    "Comment": ["Reply", "Remark"],
    "Category": ["Group", "Section"],
    "status": ["state", "stage"],
    "email": ["contactEmail", "mailAddress"],
    "name": ["label", "displayName"],
    "title": ["heading", "subject"],
    "createdAt": ["registeredAt", "createdOn"],
    "quantity": ["qty", "count"],
    "amount": ["total", "sum"],
    "price": ["unitPrice", "cost"],
}


class Augmenter:
    def __init__(self, rng: random.Random):
        self.rng = rng

    # -- 1. persona paraphrase --------------------------------------------------
    def paraphrase_persona(self, instruction: str, persona: str) -> tuple[str, str] | None:
        swaps = {
            "junior": (
                "pm",
                lambda s: re.sub(r"^Write a Prisma Client function that", "We need the data layer to", s),
            ),
            "pm": ("expert", lambda s: re.sub(r"^We need the data layer to", "Prisma Client:", s)),
            "expert": (
                "junior",
                lambda s: re.sub(r"^Prisma Client:", "Write a Prisma Client function that", s),
            ),
        }
        fn = swaps.get(persona)
        if not fn:
            return None
        new_instruction, new_persona = fn[1](instruction), fn[0]
        return (new_instruction, new_persona) if new_instruction != instruction else None

    # -- 2. consistent entity/field rename --------------------------------------
    def entity_rename(self, schema: str, code: str) -> tuple[str, str, dict[str, str]] | None:
        parsed = parse_schema(schema)
        if not parsed.models:
            return None
        candidates = []
        for m in parsed.models:
            if m.name in SYNONYMS:
                candidates.append((m.name, self.rng.choice(SYNONYMS[m.name])))
        for m in parsed.models:
            for f in m.scalar_fields():
                if f.name in SYNONYMS and (m.name, f.name) not in [(a, b) for a, b in candidates]:
                    candidates.append((f.name, self.rng.choice(SYNONYMS[f.name])))
        if not candidates:
            return None
        mapping = {old: new for old, new in candidates[: self.rng.randint(1, min(3, len(candidates)))]}
        new_schema, new_code = schema, code
        for old, new in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
            # word-boundary rename that respects camelCase attribute references
            new_schema = re.sub(rf"\b{re.escape(old)}\b", new, new_schema)
            new_code = re.sub(rf"\b{re.escape(old)}\b", new, new_code)
        return new_schema, new_code, mapping

    # -- 3. schema shuffle --------------------------------------------------------
    def schema_shuffle(self, schema: str) -> str:
        parsed = parse_schema(schema)
        if len(parsed.models) < 2:
            return schema
        models = parsed.models[:]
        self.rng.shuffle(models)
        for m in models:
            self.rng.shuffle(m.fields)
        from prismacoder.psl import render_schema

        return render_schema(
            models, parsed.enums, provider=parsed.datasource.get("provider", "sqlite").split(".")[0]
        )

    # -- 4. distractor insertion ---------------------------------------------------
    def distractor_insertion(self, schema: str, distractor_pool: list[str]) -> str | None:
        """Insert an off-task model block from a pool of small schemas."""
        if not distractor_pool:
            return None
        parsed = parse_schema(schema)
        # don't collide with existing names
        for _ in range(5):
            candidate = self.rng.choice(distractor_pool)
            names = re.findall(r"(?m)^model\s+(\w+)\s*\{", candidate)
            if names and not set(names) & set(parsed.model_names()):
                return schema.rstrip() + "\n\n" + candidate.strip() + "\n"
        return None

    # -- 5. full-record augmentation dispatch ---------------------------------------
    def augment_record(self, rec: dict[str, Any], distractor_pool: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        task = rec.get("task", {})
        schema = task.get("context", {}).get("schema", "")
        code = task.get("reference", "")

        persona_swap = self.paraphrase_persona(task.get("instruction", ""), task.get("persona", "junior"))
        if persona_swap:
            copy = self._clone(rec)
            copy["task"]["instruction"] = persona_swap[0]
            copy["task"]["persona"] = persona_swap[1]
            copy = self._retag(copy, "persona_paraphrase")
            out.append(copy)

        if schema and code:
            renamed = self.entity_rename(schema, code)
            if renamed:
                new_schema, new_code, mapping = renamed
                copy = self._clone(rec)
                copy["task"]["context"]["schema"] = new_schema
                copy["task"]["reference"] = new_code
                copy = self._retag(copy, "entity_rename")
                copy["meta"]["tags"].append("rename:" + ",".join(f"{k}->{v}" for k, v in mapping.items()))
                out.append(copy)

        if schema:
            shuffled = self.schema_shuffle(schema)
            if shuffled != schema:
                copy = self._clone(rec)
                copy["task"]["context"]["schema"] = shuffled
                copy = self._retag(copy, "schema_shuffle")
                out.append(copy)

            distr = self.distractor_insertion(schema, distractor_pool)
            if distr:
                copy = self._clone(rec)
                copy["task"]["context"]["schema"] = distr
                copy = self._retag(copy, "distractor_insertion")
                out.append(copy)

        return out

    # -- helpers ---------------------------------------------------------------------
    @staticmethod
    def _clone(rec: dict[str, Any]) -> dict[str, Any]:
        import copy

        return copy.deepcopy(rec)

    @staticmethod
    def _retag(rec: dict[str, Any], kind: str) -> dict[str, Any]:
        rec.setdefault("meta", {}).setdefault("tags", []).extend(["augmented", kind])
        rec["source"]["track"] = "augmented"
        return rec
