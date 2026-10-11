"""Grammar sampler + perturbation engine (report 9.4).

``SchemaSampler``: constrained random process over the 12 domain archetypes,
parameterized by entity count (3-12), relation depth (1-4), m-n style
(implicit/explicit), key style (autoincrement/uuid/cuid/composite), enum
density, soft-delete, multi-tenancy, and audit timestamps. Output is a
``SampledSchema``: the rendered PSL *plus* the internal representation the
reference generators use (so composed answers are mechanically correct).

``PerturbationEngine``: operations matrix (add/remove/retype field; introduce/
merge/split relations; toggle optionality; swap key strategy; add composite
unique; add index; change cascade rules) producing schema-edit pairs with
mechanically checkable before/after semantics.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from prismacoder.psl import Enum, Field, Model, render_schema
from prismacoder.synthetic.vocab import DomainSpec, FieldSpec, domain_by_key

KEY_STYLES = ("autoincrement", "uuid", "cuid", "composite")
DEFAULTS_FOR = {
    "String": ['""'],
    "Int": ["0", "1"],
    "Float": ["0.0"],
    "Boolean": ["true", "false"],
    "DateTime": ["now()"],
    "BigInt": ["0"],
}


@dataclass
class SampledSchema:
    domain: str
    provider: str
    models: list[Model]
    enums: list[Enum]
    key_style: str
    soft_delete: bool
    multi_tenant: bool
    tenant_field: str | None
    audit_fields: bool
    params: dict[str, Any] = dc_field(default_factory=dict)

    # -- introspection used by reference generators ---------------------------
    def model(self, name: str) -> Model | None:
        return next((m for m in self.models if m.name == name), None)

    def model_names(self) -> list[str]:
        return [m.name for m in self.models]

    def scalars(self, name: str) -> list[Field]:
        m = self.model(name)
        return [f for f in (m.scalar_fields() if m else []) if not f.name.startswith("_")]

    def unique_scalars(self, name: str) -> list[Field]:
        return [f for f in self.scalars(name) if "@unique" in f.attributes or "@id" in f.attributes]

    def required_scalars(self, name: str) -> list[Field]:
        """Fields needed for a create() payload (id/audit/defaults excluded)."""
        out = []
        for f in self.scalars(name):
            if "@id" in f.attributes or "@default" in " ".join(f.attributes):
                continue
            if f.name in ("createdAt", "updatedAt"):
                continue
            if f.is_optional:
                continue
            out.append(f)
        return out

    def key_info(self, name: str) -> dict:
        """Key descriptor for a model: {kind, fields: [(name, type)], scalar_type}.

        kind: autoincrement | uuid | cuid | composite. Used by the relation
        wiring so FK scalars always match the referenced key type (V1 gate).
        """
        m = self.model(name)
        if m is None:
            return {"kind": "autoincrement", "fields": [("id", "Int")], "scalar_type": "Int"}
        for f in m.fields:
            if "@id" in f.attributes:
                return {"kind": "autoincrement", "fields": [(f.name, f.type)], "scalar_type": f.type}
        for ba in m.block_attributes:
            if ba.startswith("@@id"):
                import re as _re

                cols = [c.strip() for c in (_re.findall(r"\[([^\]]+)\]", ba) or [""])[0].split(",")]
                cols = [c for c in cols if c]
                types = []
                for c in cols:
                    ft = next((f.type for f in m.fields if f.name == c), "Int")
                    types.append((c, ft))
                return {"kind": "composite", "fields": types, "scalar_type": types[0][1] if types else "Int"}
        return {"kind": "autoincrement", "fields": [("id", "Int")], "scalar_type": "Int"}

    def id_scalar_type(self, name: str) -> str:
        return self.key_info(name)["scalar_type"]

    def _build_relations(self, models: list[Model], entities: list[Any]) -> None:
        """Wire both sides of every relation with explicit @relation attributes.

        Forms emitted (Prisma v7-valid):
          1m : P {{ items C[] @relation(name) }}   C {{ pId Int,
                 p P @relation(name, fields:[pId], references:[id]) }}
          11 : same as 1m but fk Int? and single side optional
          mn : A {{ bs B[] @relation(name) }}      B {{ as A[] @relation(name) }}
          self: fk scalar + single (fields/refs) + list back-side, shared name
        """
        by_name = {m.name: m for m in models}
        seen_mn: set[tuple[str, str, str]] = set()

        def has_field(m: Model, name: str) -> bool:
            return any(f.name == name for f in m.fields)

        def ensure_fk(m: Model, name: str, ftype: str, *, is_optional: bool = False) -> None:
            """Create the FK scalar, or - if a same-named column already exists (tenant placeholder, domain field such as
            companyId) - align its type/optionality with the referenced key. Reusing it unchanged produced FK type mismatches."""
            f = next((x for x in m.fields if x.name == name), None)
            if f is None:
                m.fields.append(Field(name, ftype, is_optional=is_optional))
                return
            if f.is_relation:
                return
            f.type = ftype
            f.is_optional = is_optional
            f.is_list = False
            f.attributes = [a for a in f.attributes if not a.startswith(("@default", "@db."))]

        def get_or_add(
            m: Model, name: str, ftype: str, *, is_list: bool = False, is_optional: bool = False
        ) -> Field:
            f = next((x for x in m.fields if x.name == name and x.is_relation), None)
            if f is None:
                f = Field(name, ftype, is_list=is_list, is_optional=is_optional, is_relation=True)
                m.fields.append(f)
            f.is_relation = True
            f.is_list = is_list
            return f

        for e in entities:
            parent = by_name.get(e.name)
            if parent is None:
                continue
            for rel in e.rels:
                if rel.cardinality == "self":
                    sm = by_name.get(rel.target)
                    if sm is None:
                        continue
                    name = f"{sm.name}_self_{rel.name}"
                    key = self.key_info(sm.name)
                    fk_fields: list[str] = []
                    ref_fields: list[str] = []
                    if key["kind"] == "composite":
                        for kname, ktype in key["fields"]:
                            fk_name = f"parent{_singular(sm.name)}_{kname}"
                            ensure_fk(sm, fk_name, ktype, is_optional=True)
                            fk_fields.append(fk_name)
                            ref_fields.append(kname)
                    else:
                        fk = f"parent{_singular(sm.name)}Id"
                        ensure_fk(sm, fk, key["scalar_type"], is_optional=True)
                        fk_fields.append(fk)
                        ref_fields.append("id")
                    single = get_or_add(sm, rel.name, sm.name, is_optional=True)
                    fk_list = ", ".join(fk_fields)
                    ref_list = ", ".join(ref_fields)
                    single.attributes = [
                        f'@relation(name: "{name}", fields: [{fk_list}], references: [{ref_list}])'
                    ]
                    back = get_or_add(sm, rel.target_name or "children", sm.name, is_list=True)
                    back.attributes = [f'@relation(name: "{name}")']
                    continue

                child = by_name.get(rel.target)
                if child is None:
                    continue

                if rel.cardinality == "1m":
                    name = f"{e.name}_{rel.name}_{rel.target}"
                    lst = get_or_add(parent, rel.name, rel.target, is_list=True)
                    lst.attributes = [f'@relation(name: "{name}")']
                    key = self.key_info(e.name)
                    if key["kind"] == "composite":
                        fk_fields = []
                        for kname, ktype in key["fields"]:
                            fk_name = f"{_snake(e.name)}_{kname}"
                            ensure_fk(child, fk_name, ktype)
                            fk_fields.append(fk_name)
                        single_name = _camel_prop(e.name)
                        single = get_or_add(child, single_name, e.name)
                        single.is_optional = False
                        fk_list = ", ".join(fk_fields)
                        ref_list = ", ".join(k for k, _ in key["fields"])
                        single.attributes = [
                            f'@relation(name: "{name}", fields: [{fk_list}], references: [{ref_list}])'
                        ]
                    else:
                        fk = f"{_snake(e.name)}Id"
                        ensure_fk(child, fk, key["scalar_type"])
                        single_name = _camel_prop(e.name)
                        single = get_or_add(child, single_name, e.name)
                        single.is_optional = False
                        single.attributes = [f'@relation(name: "{name}", fields: [{fk}], references: [id])']

                elif rel.cardinality == "11":
                    name = f"{e.name}_{rel.name}_{rel.target}"
                    lst = get_or_add(parent, rel.name, rel.target, is_list=True)
                    lst.attributes = [f'@relation(name: "{name}")']
                    key = self.key_info(e.name)
                    if key["kind"] == "composite":
                        fk_fields = []
                        for kname, ktype in key["fields"]:
                            fk_name = f"{_snake(e.name)}_{kname}"
                            ensure_fk(child, fk_name, ktype, is_optional=True)
                            fk_fields.append(fk_name)
                        single_name = _camel_prop(e.name)
                        single = get_or_add(child, single_name, e.name, is_optional=True)
                        single.is_optional = True
                        fk_list = ", ".join(fk_fields)
                        ref_list = ", ".join(k for k, _ in key["fields"])
                        single.attributes = [
                            f'@relation(name: "{name}", fields: [{fk_list}], references: [{ref_list}])'
                        ]
                    else:
                        fk = f"{_snake(e.name)}Id"
                        ensure_fk(child, fk, key["scalar_type"], is_optional=True)
                        single_name = _camel_prop(e.name)
                        single = get_or_add(child, single_name, e.name, is_optional=True)
                        single.is_optional = True
                        single.attributes = [f'@relation(name: "{name}", fields: [{fk}], references: [id])']

                elif rel.cardinality == "mn":
                    if (e.name, rel.target, rel.name) in seen_mn:
                        continue
                    # implicit m-n needs single-field primary keys on BOTH sides;
                    # with composite keys the relation is not expressible -> drop
                    if (
                        self.key_info(e.name)["kind"] == "composite"
                        or self.key_info(rel.target)["kind"] == "composite"
                    ):
                        continue
                    seen_mn.add((e.name, rel.target, rel.name))
                    name = f"{e.name}_to_{rel.target}"
                    side_a = get_or_add(parent, rel.name, rel.target, is_list=True)
                    side_a.attributes = [f'@relation(name: "{name}")']
                    opp_label = _camel_prop(e.name)
                    if not opp_label.endswith("s"):
                        opp_label += "s"
                    opp = next((f for f in child.fields if f.type == e.name and f.is_relation), None)
                    if opp is None:
                        opp = Field(opp_label, e.name, is_list=True, is_relation=True)
                        child.fields.append(opp)
                    opp.is_relation = True
                    opp.is_list = True
                    opp.attributes = [f'@relation(name: "{name}")']

    def render(self) -> str:
        return render_schema(self.models, self.enums, provider=self.provider)

    def complexity(self) -> dict[str, int]:
        rel_depth = 0
        for m in self.models:
            for f in m.relation_fields():
                tgt = self.model(f.type)
                if tgt and any(tf.type == m.name for tf in tgt.relation_fields()):
                    rel_depth = max(rel_depth, 2)
                else:
                    rel_depth = max(rel_depth, 1)
        return {
            "models": len(self.models),
            "relations": sum(len(m.relation_fields()) for m in self.models) // 2,
            "nesting": max(rel_depth, 1),
            "ops": 0,
            "ambiguity": 0,
        }


class SchemaSampler:
    def __init__(self, rng: random.Random):
        self.rng = rng

    # -- main entry -----------------------------------------------------------
    def sample(
        self,
        domain: str | DomainSpec | None = None,
        *,
        entity_count: int | None = None,
        level: str | None = None,
        provider: str | None = None,
    ) -> SampledSchema:
        spec = (
            domain
            if isinstance(domain, DomainSpec)
            else domain_by_key(
                domain
                or self.rng.choice(
                    [
                        d.key
                        for d in __import__(
                            "prismacoder.synthetic.vocab", fromlist=["DOMAINS_SPECS"]
                        ).DOMAINS_SPECS
                    ]
                )
            )
        )
        n = entity_count or self.rng.randint(3, min(12, len(spec.entities)))
        if level == "L1":
            n = min(n, 6)
        elif level == "L2":
            n = max(4, min(n, 10))
        elif level == "L3":
            n = max(8, n)

        entities = self.rng.sample(list(spec.entities), min(n, len(spec.entities)))
        names = {e.name for e in entities}
        # keep only rels whose target survived sampling
        entities = [EntitySpec_trim(e, names) for e in entities]
        entities = [e for e in entities if e is not None]

        key_style = self.rng.choice(KEY_STYLES)
        soft_delete = self.rng.random() < 0.4
        multi_tenant = bool(spec.tenant_field) and self.rng.random() < 0.35
        audit_fields = self.rng.random() < 0.75
        enum_density = self.rng.random()
        use_enums = enum_density < 0.6 and spec.enums
        provider = provider or self.rng.choice(["sqlite", "postgresql", "mysql"])

        models = [
            self._model(
                e,
                key_style,
                soft_delete,
                audit_fields,
                multi_tenant,
                spec.tenant_field if multi_tenant else None,
            )
            for e in entities
        ]
        sampled = SampledSchema(
            domain=spec.key,
            provider=provider,
            models=models,
            enums=[] if not use_enums else list(spec.enums[: self.rng.randint(1, len(spec.enums))]),
            key_style=key_style,
            soft_delete=soft_delete,
            multi_tenant=multi_tenant,
            tenant_field=spec.tenant_field if multi_tenant else None,
            audit_fields=audit_fields,
            params={"entity_count": len(models), "level_hint": level or ""},
        )
        sampled._build_relations(models, entities)
        if use_enums:
            models = self._inline_enums(models, spec.enums if self.rng.random() < 0.5 else spec.enums[:1])
        return sampled

    # -- builders -------------------------------------------------------------
    def _key_field(self, key_style: str, composite_group: int) -> Field:
        if key_style == "autoincrement":
            return Field("id", "Int", attributes=["@id", "@default(autoincrement())"])
        if key_style == "uuid":
            return Field("id", "String", attributes=["@id", "@default(uuid())"])
        if key_style == "cuid":
            return Field("id", "String", attributes=["@id", "@default(cuid())"])
        return Field(f"group{composite_group}Id", "Int")  # composite: @@id finished by caller

    def _model(
        self,
        e: Any,
        key_style: str,
        soft_delete: bool,
        audit: bool,
        multi_tenant: bool,
        tenant_field: str | None,
    ) -> Model:
        m = Model(name=e.name)
        if key_style == "composite" and len(e.fields) >= 2:
            keys = [f.name for f in e.fields[:2]]
            m.fields.append(Field(f"{keys[0]}_id", "Int"))
            m.fields.append(Field(f"{keys[1]}_id", "Int"))
            m.block_attributes.append(f"@@id([{keys[0]}_id, {keys[1]}_id])")
        else:
            m.fields.append(self._key_field(key_style if key_style != "composite" else "autoincrement", 0))

        for fs in e.fields:
            m.fields.append(self._field_from_spec(fs))

        # tenant scoping: either a scalar FK placeholder or an optional relation
        if multi_tenant and tenant_field:
            m.fields.append(Field(tenant_field, "String"))

        # Relations are wired AFTER all models exist (see _build_relations) so
        # both sides of every relation are emitted with paired fields - the V1
        # gate rejects one-sided relations, so this pass is correctness-critical.
        if soft_delete and not any(f.name == "deletedAt" for f in m.fields):
            m.fields.append(Field("deletedAt", "DateTime", is_optional=True))
        return m

    def _field_from_spec(self, fs: FieldSpec) -> Field:
        attrs: list[str] = []
        if fs.unique:
            attrs.append("@unique")
        if fs.default is not None:
            attrs.append(f"@default({fs.default})")
        return Field(fs.name, fs.type, is_optional=fs.optional, attributes=attrs)

    def _inline_enums(self, models: list[Model], enums: tuple[Any, ...]) -> list[Model]:
        """Swap a 'status'-style String field for an enum type where names line up.

        PSL requires bare enum values in @default (the V1 gate rejects quoted
        strings on enum fields): defaults are unquoted when the value is a
        member of the enum and dropped otherwise.
        """
        enum_values = {e.name: set(e.values) for e in enums}
        for e in enums:
            field_names = {"status", "kind", "severity", "role", "visibility", "direction"}
            for m in models:
                for f in m.fields:
                    if f.type != "String" or not (f.name in field_names or e.name.lower() in f.name.lower()):
                        continue
                    f.type = e.name
                    keep: list[str] = []
                    for a in f.attributes:
                        if a.startswith("@default"):
                            raw = a[len("@default(") : -1]
                            unquoted = raw.strip('"').strip("'")
                            if unquoted in enum_values[e.name]:
                                keep.append(f"@default({unquoted})")
                            continue
                        keep.append(a)
                    f.attributes = keep
        return models


def EntitySpec_trim(e: Any, keep_names: set[str]) -> Any | None:  # noqa: N802 (kept name stable)
    """Copy an EntitySpec keeping only rels whose target is still sampled."""
    from dataclasses import replace

    rels = tuple(r for r in e.rels if r.target in keep_names or r.cardinality == "self")
    return replace(e, rels=rels)


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _singular(name: str) -> str:
    return name[:-1] if name.endswith("s") else name


def _camel_prop(name: str) -> str:
    """Model name -> camelCase client/field property (User -> user)."""
    return name[:1].lower() + name[1:]


class PerturbationEngine:
    """Operations matrix -> before/after schema pairs with change descriptors."""

    OPS = (
        "add_field",
        "remove_field",
        "retype_field",
        "toggle_optionality",
        "add_unique",
        "add_index",
        "add_composite_unique",
        "swap_key_style",
        "add_enum_field",
        "change_cascade",
    )

    def __init__(self, rng: random.Random):
        self.rng = rng

    def perturb(self, sampled: SampledSchema) -> tuple[str, str, dict[str, Any]]:
        """Returns (before_psl, after_psl, change_metadata). Applies 1-3 ops."""
        before = sampled.render()
        n_ops = self.rng.randint(1, 3)
        ops = self.rng.sample(self.OPS, n_ops)
        models = [self._clone(m) for m in sampled.models]
        applied: list[str] = []
        for op in ops:
            fn = getattr(self, f"_op_{op}")
            if fn(models, sampled):          # ops report whether they really changed the schema
                applied.append(op)
        after_schema = SampledSchema(
            domain=sampled.domain,
            provider=sampled.provider,
            models=models,
            enums=sampled.enums,
            key_style=sampled.key_style,
            soft_delete=sampled.soft_delete,
            multi_tenant=sampled.multi_tenant,
            tenant_field=sampled.tenant_field,
            audit_fields=sampled.audit_fields,
            params=dict(sampled.params, ops=applied),
        )
        after = after_schema.render()
        if after == before:
            # every chosen op was a no-op on this schema (e.g. swap_key_style on
            # a composite-key model) -> force an observable change
            self._op_add_field(models, sampled)
            applied.append("add_field")
            after_schema = SampledSchema(
                domain=sampled.domain,
                provider=sampled.provider,
                models=models,
                enums=sampled.enums,
                key_style=sampled.key_style,
                soft_delete=sampled.soft_delete,
                multi_tenant=sampled.multi_tenant,
                tenant_field=sampled.tenant_field,
                audit_fields=sampled.audit_fields,
                params=dict(sampled.params, ops=applied),
            )
            after = after_schema.render()
        return before, after, {"ops": applied}

    # -- ops ------------------------------------------------------------------
    @staticmethod
    def _clone(m: Model) -> Model:
        import copy

        return copy.deepcopy(m)

    def _rand_model(self, models: list[Model]) -> Model:
        return self.rng.choice(models)

    # -- helpers shared by the ops: which scalars must stay as they are for the schema to remain valid ----
    @staticmethod
    def _protected(models: list[Model], m: Model) -> set[str]:
        """Scalar fields of `m` that other constructs depend on (FK columns, referenced keys, index members)."""
        keep: set[str] = set()
        for f in m.relation_fields():                               # FK scalars used by this model's relations
            for a in f.attributes:
                mt = re.search(r"fields:\s*\[([^\]]*)\]", a)
                if mt:
                    keep |= {x.strip() for x in mt.group(1).split(",") if x.strip()}
        for ba in m.block_attributes:                               # @@id / @@index / @@unique members
            for mt in re.finditer(r"\[([^\]]*)\]", ba):
                keep |= {x.strip().split("(")[0] for x in mt.group(1).split(",") if x.strip()}
        for other in models:                                        # keys referenced by other models' relations
            for f in other.relation_fields():
                if f.type != m.name:
                    continue
                for a in f.attributes:
                    mt = re.search(r"references:\s*\[([^\]]*)\]", a)
                    if mt:
                        keep |= {x.strip() for x in mt.group(1).split(",") if x.strip()}
        return keep

    def _op_add_field(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        for _ in range(50):
            name = f"extraField{self.rng.randint(2, 99)}"
            if name not in m.field_names():
                break
        else:
            return False
        ftype = self.rng.choice(["String", "Int", "Boolean", "Float", "DateTime"])
        m.fields.append(Field(name, ftype, is_optional=self.rng.random() < 0.5))
        return True

    def _op_remove_field(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        keep = self._protected(models, m)
        removable = [
            f
            for f in m.fields
            if not f.is_relation
            and "@id" not in f.attributes
            and f.name not in keep
            and f.name not in ("createdAt", "updatedAt")
        ]
        if not removable:
            return False
        m.fields.remove(self.rng.choice(removable))
        return True

    def _op_retype_field(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        keep = self._protected(models, m)
        # only plain scalars: no attributes (so no @default/@db.* that would stop matching the new type), not a key/FK
        cands = [
            f
            for f in m.scalar_fields()
            if not f.attributes and not f.is_list and f.type in ("String", "Int", "Float", "Boolean") and f.name not in keep
        ]
        if not cands:
            return False
        f = self.rng.choice(cands)
        f.type = self.rng.choice([t for t in ("String", "Int", "Float", "Boolean") if t != f.type])
        return True

    def _op_toggle_optionality(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        keep = self._protected(models, m)
        cands = [
            f for f in m.scalar_fields() if "@id" not in f.attributes and not f.is_list and f.name not in keep
        ]
        if not cands:
            return False
        f = self.rng.choice(cands)
        f.is_optional = not f.is_optional
        return True

    def _op_add_unique(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        cands = [
            f
            for f in m.scalar_fields()
            if "@unique" not in f.attributes and "@id" not in f.attributes and not f.is_list
        ]
        if not cands:
            return False
        self.rng.choice(cands).attributes.append("@unique")
        return True

    def _op_add_index(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        cands = [f.name for f in m.scalar_fields() if f"@@index([{f.name}])" not in m.block_attributes and not f.is_list]
        if not cands:
            return False
        m.block_attributes.append(f"@@index([{self.rng.choice(cands)}])")
        return True

    def _op_add_composite_unique(self, models: list[Model], s: SampledSchema) -> bool:
        m = self._rand_model(models)
        cands = [f.name for f in m.scalar_fields() if "@id" not in f.attributes and not f.is_list][:2]
        if len(cands) < 2:
            return False
        attr = f"@@unique([{cands[0]}, {cands[1]}])"
        if attr in m.block_attributes:
            return False
        m.block_attributes.append(attr)
        return True

    def _op_swap_key_style(self, models: list[Model], s: SampledSchema) -> bool:
        if s.provider == "mongodb":                     # Mongo ids must stay @db.ObjectId / @map("_id")
            return False
        for m in models:
            if any(ba.startswith("@@id") for ba in m.block_attributes):
                continue                                # composite key
            if any(f.type == m.name for o in models for f in o.relation_fields()):
                continue                                # changing this id's type would break FK columns (other models or self-relations)
            ids = [f for f in m.fields if "@id" in f.attributes]
            if len(ids) != 1 or any(a.startswith("@db.") for a in ids[0].attributes):
                continue
            f = ids[0]
            old = (f.type, next((a for a in f.attributes if a.startswith("@default")), None))
            new_style = self.rng.choice(["uuid", "cuid", "autoincrement"])
            new = ("Int", "@default(autoincrement())") if new_style == "autoincrement" else ("String", f"@default({new_style}())")
            if new == old:
                continue
            f.type = new[0]
            f.attributes = ["@id", new[1]]
            return True
        return False

    def _op_add_enum_field(self, models: list[Model], s: SampledSchema) -> bool:
        if not s.enums:
            return False
        m = self._rand_model(models)
        e = self.rng.choice(s.enums)
        name = f"{_snake(e.name)}Ref"
        if name in m.field_names():
            return False
        m.fields.append(Field(name, e.name, is_optional=True))
        return True

    def _op_change_cascade(self, models: list[Model], s: SampledSchema) -> bool:
        # keep fields:/references: and only change onDelete (the old code replaced the whole @relation(...))
        owning = [
            (m, f, i)
            for m in models
            for f in m.relation_fields()
            for i, a in enumerate(f.attributes)
            if a.startswith("@relation(") and "fields:" in a
        ]
        if not owning:
            return False
        m, f, i = self.rng.choice(owning)
        attr = f.attributes[i]
        cur = re.search(r"onDelete:\s*(\w+)", attr)
        options = ["Cascade", "Restrict"] + (["SetNull"] if f.is_optional else [])
        options = [o for o in options if not cur or o != cur.group(1)]
        if not options:
            return False
        base = re.sub(r",?\s*onDelete:\s*\w+", "", attr)
        f.attributes[i] = base[:-1] + f", onDelete: {self.rng.choice(options)})"
        return True
