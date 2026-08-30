"""Unit tests: PSL parser, dedup, license gate, secrets, PII, quality, version norm."""

from prismacoder.dedup import Deduplicator, estimated_jaccard, minhash_signature
from prismacoder.hashing import model_graph_signature
from prismacoder.license_filter import check_license, license_text_spdx
from prismacoder.pii import scrub_text
from prismacoder.psl import parse_schema, render_schema
from prismacoder.quality import check_schema, check_source_code
from prismacoder.secrets import scan_and_clean
from prismacoder.version_norm import detect_version, normalize_to_v7


class TestPSL:
    def test_parse_roundtrip(self, sample_schema):
        parsed = parse_schema(sample_schema)
        assert [m.name for m in parsed.models] == ["User", "Post"]
        assert parsed.enums == []
        assert parsed.datasource["provider"] == "sqlite"
        user = parsed.get_model("User")
        assert user.primary_key() is not None
        rel = [f for f in user.relation_fields()]
        assert len(rel) == 1 and rel[0].type == "Post" and rel[0].is_list

    def test_relation_pairing(self, sample_schema):
        parsed = parse_schema(sample_schema)
        post = parsed.get_model("Post")
        author = post.relation_fields()[0]
        assert "@relation(" in " ".join(author.attributes)

    def test_render_is_reparseable(self, sample_schema):
        parsed = parse_schema(sample_schema)
        rendered = render_schema(parsed.models, parsed.enums, provider="postgresql")
        reparsed = parse_schema(rendered)
        assert [m.name for m in reparsed.models] == [m.name for m in parsed.models]


class TestDedup:
    def test_exact_duplicate(self, sample_schema):
        d = Deduplicator()
        r1 = {"id": "a", "schema": sample_schema}
        r2 = {"id": "b", "schema": "  " + sample_schema.replace("\n", "  ") + "  "}
        assert d.consider(r1) is True
        assert d.consider(r2) is False
        assert d.stats.exact_dupes == 1

    def test_isomorphic_schemas_collide(self):
        d = Deduplicator()
        s1 = """datasource db { provider = "sqlite" url = env("DATABASE_URL") }
model Alpha { id Int @id value Int }
model Beta { id Int @id alphaId Int alpha Alpha }"""
        s2 = """datasource db { provider = "sqlite" url = env("DATABASE_URL") }
model Gamma { id Int @id value Int }
model Delta { id Int @id gammaId Int gamma Gamma }"""
        assert model_graph_signature(s1) == model_graph_signature(s2)
        assert d.consider({"id": "x1", "schema": s1}) is True
        assert d.consider({"id": "x2", "schema": s2}) is False
        assert d.stats.isomorphic_dupes == 1

    def test_near_duplicate_minhash(self):
        base = " ".join(f"word{i}" for i in range(200))
        other = " ".join(f"word{i}" for i in range(190)) + " trailing words here"
        sig1, sig2 = minhash_signature(base), minhash_signature(other)
        assert estimated_jaccard(sig1, sig2) > 0.8
        d = Deduplicator()
        assert d.consider({"id": "n1", "schema": base}) is True
        assert d.consider({"id": "n2", "schema": other}) is False


class TestLicense:
    def test_allowlist(self):
        for spdx in ("MIT", "Apache-2.0", "BSD-3-Clause", "ISC"):
            assert check_license(spdx).allowed

    def test_copyleft_denied(self):
        for spdx in ("GPL-3.0", "AGPL-3.0", "LGPL-2.1"):
            assert not check_license(spdx).allowed

    def test_unknown_denied(self):
        assert not check_license("UNKNOWN").allowed
        assert not check_license("").allowed

    def test_text_detection(self):
        assert license_text_spdx("Apache License Version 2.0 ...") == "Apache-2.0"
        assert license_text_spdx("GNU AFFERO GENERAL PUBLIC LICENSE ...") == "AGPL-3.0"


class TestSecrets:
    def test_aws_key_drops(self):
        text = 'const key = "AKIAIOSFODNN7EXAMPLE";'
        cleaned, findings, drop = scan_and_clean(text)
        assert drop is True
        assert any(f.kind == "aws_access_key_id" for f in findings)

    def test_generic_secret_redacted(self):
        text = 'const config = { password: "hunter2hunter2hunter2" };'
        cleaned, findings, drop = scan_and_clean(text)
        assert drop is False
        assert "hunter2hunter2hunter2" not in cleaned
        assert "REDACTED_SECRET_PLACEHOLDER" in cleaned


class TestPII:
    def test_email_scrubbed(self):
        out, edits = scrub_text("contact: john.doe@company.org")
        assert "john.doe@company.org" not in out
        assert "user@example.invalid" in out

    def test_safe_email_kept(self):
        out, _ = scrub_text("contact: test@example.com")
        assert "test@example.com" in out


class TestQuality:
    def test_valid_schema_passes(self, sample_schema):
        res = check_schema(sample_schema)
        assert res.ok, res.problems

    def test_broken_schemas_fail(self):
        assert not check_schema("model A { id Int @id").ok  # unbalanced
        assert not check_schema('datasource db { provider = "sqlite" }').ok  # no models
        unpaired = (
            'datasource db { provider = "sqlite" url = env("DATABASE_URL") }\n'
            "model A { id Int @id b B }\n"
            "model B { id Int @id }"
        )
        assert not check_schema(unpaired).ok  # missing @relation attribute

    def test_source_check(self):
        assert check_source_code("const x = 1;").ok
        assert not check_source_code("const x = {").ok


class TestVersionNorm:
    def test_client_js_normalized(self):
        legacy = 'generator client { provider = "prisma-client-js" }'
        out, report = normalize_to_v7(legacy)
        assert 'provider = "prisma-client"' in out
        assert "output" in out
        assert report.changed

    def test_stale_preview_features_dropped(self):
        s = 'generator client {\n  provider = "prisma-client"\n  previewFeatures = ["views"]\n}'
        out, _ = normalize_to_v7(s)
        assert "views" not in out

    def test_detect_version(self):
        assert detect_version('provider = "prisma-client-js"') == "v4/v5"
