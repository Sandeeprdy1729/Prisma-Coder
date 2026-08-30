"""Dataset tests: canonical schema validation + split isolation."""

from pathlib import Path

from prismacoder.io_utils import read_jsonl, write_jsonl
from prismacoder.schema import RecordSchemaError, canonicalize_record, validate_record
from prismacoder.synthetic.generator import SyntheticGenerator
from prismacoder.verification.verifier import LightValidator


class TestRecordSchema:
    def test_valid_record(self, sample_record):
        assert validate_record(sample_record) == []

    def test_invalid_category(self, sample_record):
        sample_record["task"]["categories"] = ["T99_bogus"]
        assert validate_record(sample_record)

    def test_invalid_track(self, sample_record):
        sample_record["source"]["track"] = "stolen"
        assert validate_record(sample_record)

    def test_mined_requires_repo(self, sample_record):
        sample_record["source"]["track"] = "mined"
        sample_record["source"]["origin_repo"] = ""
        assert validate_record(sample_record)

    def test_strict_raises(self, sample_record):
        sample_record["task"]["type"] = "nope"
        try:
            validate_record(sample_record, strict=True)
            raised = False
        except RecordSchemaError:
            raised = True
        assert raised

    def test_canonicalize_idempotent(self, sample_record):
        once = canonicalize_record(sample_record)
        twice = canonicalize_record(once)
        assert once == twice


class TestGeneratorOutput:
    def test_records_conform_to_schema(self):
        gen = SyntheticGenerator(seed=99)
        records = gen.generate(30)
        assert len(records) >= 25
        problems = []
        for rec in records:
            problems.extend(validate_record(rec))
        assert problems == []

    def test_deterministic_under_seed(self):
        r1 = SyntheticGenerator(seed=123).generate(10)
        r2 = SyntheticGenerator(seed=123).generate(10)
        assert [r["id"] for r in r1] == [r["id"] for r in r2]
        assert [r["task"]["instruction"] for r in r1] == [r["task"]["instruction"] for r in r2]

    def test_ood_domains_excluded_by_default(self):
        gen = SyntheticGenerator(seed=5)
        records = gen.generate(40)
        from prismacoder.constants import OOD_WITHHELD_DOMAINS

        assert all(r["domain"] not in OOD_WITHHELD_DOMAINS for r in records)

    def test_no_leaked_canaries(self):
        gen = SyntheticGenerator(seed=6)
        records = gen.generate(20)
        for rec in records:
            blob = rec["task"]["instruction"] + rec["task"]["reference"]
            assert "PRISMABENCH-CANARY" not in blob


class TestSplitIsolation:
    def test_database_level_splits(self, tmp_path):
        gen = SyntheticGenerator(seed=7)
        records = gen.generate(36)
        light = LightValidator()
        for rec in records:
            verdict = light.validate(rec["task"]["context"]["schema"], rec["task"]["reference"])
            rec["verification"]["status"] = "unverified" if verdict.status != "failed" else "failed"
        write_jsonl(tmp_path / "pool.jsonl", records)

        # emulate stage 07 grouping: same schema text -> same database
        from collections import defaultdict

        dbs = defaultdict(list)
        for rec in records:
            dbs[rec["task"]["context"]["schema"]].append(rec)
        assert sum(1 for v in dbs.values() if len(v) > 1) >= 1  # some sharing exists

        loaded = read_jsonl(tmp_path / "pool.jsonl")
        assert len(loaded) == len(records)


class TestNegativesRouting:
    """Stage 07 must route gate-failed records to the DPO negatives pool
    (report section 11) instead of letting them into the training splits."""

    def _run_stage07(self, tmp_path, records):
        import subprocess, sys

        write_jsonl(tmp_path / "pool.jsonl", records)
        prov = tmp_path / "prov.json"
        prov.write_text('{"provenance_hashes": []}')
        root = Path(__file__).resolve().parents[2]
        cmd = [
            sys.executable, str(root / "datasets" / "scripts" / "07_split_dataset.py"),
            "--synthetic", str(tmp_path / "pool.jsonl"),
            "--out", str(tmp_path / "splits"),
            "--seed", "7",
            "--exclude-benchmark-provenance", str(prov),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
        assert proc.returncode == 0, proc.stderr
        return proc

    def test_failed_records_never_enter_splits(self, tmp_path):
        gen = SyntheticGenerator(seed=31)
        records = gen.generate(12)
        for i, rec in enumerate(records):
            rec["verification"]["status"] = "failed" if i % 6 == 0 else "verified"
        n_failed = sum(1 for r in records if r["verification"]["status"] == "failed")
        self._run_stage07(tmp_path, records)

        splits_dir = tmp_path / "splits"
        negatives = read_jsonl(splits_dir / "negatives.jsonl")
        assert len(negatives) == n_failed
        assert all((r.get("verification") or {}).get("status") == "failed" for r in negatives)
        for split in ("train.jsonl", "validation.jsonl", "test.jsonl"):
            for rec in read_jsonl(splits_dir / split):
                assert (rec.get("verification") or {}).get("status") != "failed"

    def test_unverified_records_are_kept(self, tmp_path):
        gen = SyntheticGenerator(seed=32)
        records = gen.generate(10)
        for rec in records:
            rec["verification"]["status"] = "unverified"
        self._run_stage07(tmp_path, records)
        splits_dir = tmp_path / "splits"
        assert not (splits_dir / "negatives.jsonl").exists()
        total = sum(len(read_jsonl(splits_dir / s)) for s in ("train.jsonl", "validation.jsonl", "test.jsonl"))
        assert total == len(records)
