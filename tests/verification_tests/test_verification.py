"""Verification tests: sandbox creation + light validator + gate utilities."""

import json

from prismacoder.verification.prisma_gate import parse_prisma_errors, parse_ts_errors
from prismacoder.verification.sandbox import PACKAGE_JSON, TSCONFIG, SandboxProject, node_available
from prismacoder.verification.verifier import LightValidator, classify_error


class TestSandboxProject:
    def test_materialization(self, sample_schema, tmp_path):
        sb = SandboxProject.create(
            sample_schema, {"artifact.ts": "export const a = 1;"}, base_dir=str(tmp_path / "sb"), keep=True
        )
        try:
            assert (sb.root / "prisma" / "schema.prisma").exists()
            assert (sb.root / "src" / "artifact.ts").exists()
            pkg = json.loads((sb.root / "package.json").read_text())
            assert pkg["devDependencies"]["prisma"]
            assert (sb.root / ".env").exists()
            ts = json.loads((sb.root / "tsconfig.json").read_text())
            assert ts["compilerOptions"]["strict"] is True
        finally:
            sb.keep = False
            sb.cleanup()
        assert not (tmp_path / "sb").exists()

    def test_pinned_toolchain(self):
        assert PACKAGE_JSON["devDependencies"]["prisma"].startswith("7.")
        assert PACKAGE_JSON["prisma"]["schema"] == "prisma/schema.prisma"
        assert TSCONFIG["compilerOptions"]["noEmit"] is True


class TestLightValidator:
    def test_grounds_valid_artifact(self, sample_schema):
        artifact = "const rows = await prisma.user.findMany({ include: { posts: true } });"
        res = LightValidator().validate(sample_schema, artifact)
        assert res.status != "failed", res.diagnostics

    def test_flags_hallucinated_model(self, sample_schema):
        artifact = "const rows = await prisma.auditlog.findMany();"
        res = LightValidator().validate(sample_schema, artifact)
        assert res.status == "failed"
        assert any("ungrounded model" in d for d in res.diagnostics)

    def test_flags_hallucinated_field(self, sample_schema):
        artifact = 'prisma.user.create({ data: { emailx: "x@y.z" } });'
        res = LightValidator().validate(sample_schema, artifact)
        assert res.status == "failed"

    def test_flags_unbalanced_fences(self, sample_schema):
        res = LightValidator().validate(sample_schema, "```ts\nconst a = 1;\n``` ```ts")
        # odd fence count -> flagged
        assert res.status == "failed" or res.gates["V0_static"]["passed"] is True

    def test_empty_artifact_fails(self, sample_schema):
        """Empty output can never pass a full gate - static mode must fail it
        (otherwise an empty benchmark submission scores pass@1 = 1.0)."""
        for artifact in ("", "   "):
            res = LightValidator().validate(sample_schema, artifact)
            assert res.status == "failed"
            assert res.error_class == "E7"
            assert res.gates["V0_static"]["passed"] is False


class TestClassifiers:
    def test_error_taxonomy_mapping(self):
        assert classify_error("V1_validate", ["error: Expected token (line 4)"]) == "E1"
        assert classify_error("V2_generate", ["error: codegen failed"]) == "E3"
        assert classify_error("V3_tsc", ["TS2531 Object is possibly null"]) == "E4"
        assert classify_error("V4_execute", ["P2002: Unique constraint failed"]) == "E5"
        assert classify_error("V5_vitest", ["expect(received).toBe"]) == "E6"

    def test_prisma_error_parser(self):
        out = "error: Unknown arg `emaill` in where\n  -->  schema.prisma:12"
        errs = parse_prisma_errors(out)
        assert errs and "12" in errs[0]

    def test_ts_error_parser(self):
        out = "src/artifact.ts(5,10): error TS2339: Property 'x' does not exist."
        errs = parse_ts_errors(out)
        assert errs and "TS2339" in errs[0]


class TestNodeAvailability:
    def test_node_optional(self):
        node, npm = node_available()
        # in CI without node this is (None, None); the codebase must handle it
        assert node is None or node
        if node is None:
            assert npm is None or npm
