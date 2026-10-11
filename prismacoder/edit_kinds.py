"""Detect WHICH kinds of change separate two Prisma schemas (the T2 edit-task vocabulary).

Used to check that a model's edited schema (or the generator's own reference) really contains the changes the prompt names.
"""
import re

def parse_struct(schema):
    """-> (models, enums). models: {name: {'fields': {fname: dict(type, base, optional, list, attrs)}, 'block': [str]}}"""
    models, enums, cur, cur_kind = {}, {}, None, None
    for line in schema.splitlines():
        st = line.strip()
        m = re.match(r"(model|enum|type)\s+(\w+)\s*\{", st)
        if m:
            cur_kind, cur = m.group(1), m.group(2)
            if cur_kind == "model": models[cur] = {"fields": {}, "block": []}
            elif cur_kind == "enum": enums[cur] = []
            else: cur = None
            continue
        if cur and st.startswith("}"):
            cur = cur_kind = None
            continue
        if not cur or not st or st.startswith("//"):
            continue
        if cur_kind == "enum":
            enums[cur].append(st.split()[0]); continue
        if st.startswith("@@"):
            models[cur]["block"].append(re.sub(r"\s+", "", st)); continue
        parts = st.split(None, 2)
        if len(parts) < 2: continue
        t = parts[1]
        models[cur]["fields"][parts[0]] = {"type": t, "base": t.rstrip("?").replace("[]", ""), "optional": t.endswith("?"),
                                           "list": t.endswith("[]"), "attrs": parts[2] if len(parts) > 2 else ""}
    return models, enums

def _count_block(models, prefix):
    return sum(1 for m in models.values() for b in m["block"] if b.startswith(prefix))

def _count_field_attr(models, attr):
    return sum(1 for m in models.values() for f in m["fields"].values() if re.search(r"@unique\b", f["attrs"]))

def _ondelete(models):
    out = {}
    for mn, m in models.items():
        for fn, f in m["fields"].items():
            x = re.search(r"onDelete:\s*(\w+)", f["attrs"])
            if x: out[(mn, fn)] = x.group(1)
    return out

def _id_style(models):
    out = {}
    for mn, m in models.items():
        for fn, f in m["fields"].items():
            if "@id" in f["attrs"]:
                d = re.search(r"@default\((\w+)", f["attrs"])
                out[(mn, fn)] = (f["base"], d.group(1) if d else None)
    return out

def detect_kinds(before, after):
    """Which kinds of change happened between two schemas? -> set of op names"""
    a, ae = parse_struct(before); b, be = parse_struct(after)
    got = set()
    for mn in a:
        if mn not in b: continue
        fa, fb = a[mn]["fields"], b[mn]["fields"]
        if set(fb) - set(fa): got.add("add_field")
        if set(fa) - set(fb): got.add("remove_field")
        for fn in set(fa) & set(fb):
            if fa[fn]["base"] != fb[fn]["base"]: got.add("retype_field")
            if fa[fn]["optional"] != fb[fn]["optional"]: got.add("toggle_optionality")
    if _count_field_attr(b, "@unique") > _count_field_attr(a, "@unique"): got.add("add_unique")
    if _count_block(b, "@@unique") > _count_block(a, "@@unique"): got.add("add_composite_unique")
    if _count_block(b, "@@index") > _count_block(a, "@@index"): got.add("add_index")
    if _id_style(a) != _id_style(b) and any(k in _id_style(b) for k in _id_style(a)): got.add("swap_key_style")
    enum_names_b = set(be)
    en = lambda ms, names: sum(1 for m in ms.values() for f in m["fields"].values() if f["base"] in names)
    if en(b, enum_names_b) > en(a, set(ae)): got.add("add_enum_field")
    if _ondelete(a) != _ondelete(b): got.add("change_cascade")
    return got

def kinds_ok(before, after, required):
    return set(required) <= detect_kinds(before, after)
