"""Benchmark tests: composition, public/private split, evaluation harness."""

from pathlib import Path

from benchmark.evaluator.evaluate import load_reference_index, score_task
from benchmark.evaluator.leaderboard import collect, render_markdown
from benchmark.generator.generate_benchmark import BenchmarkBuilder
from benchmark.generator.task_specs import TASK_SPECS, build_plan, public_task_shape
from prismacoder.eval_metrics.registry import default_registry
from prismacoder.io_utils import read_jsonl, write_json, write_jsonl
from prismacoder.verification.verifier import LightValidator


class TestPlan:
    def test_exact_table_101_composition(self):
        from prismacoder.constants import BENCH_TASK_COUNTS

        plan = build_plan()
        totals = {}
        for category, _level, n in plan:
            totals[category] = totals.get(category, 0) + n
        assert totals == BENCH_TASK_COUNTS

    def test_specs_gates_match_levels(self):
        for spec in TASK_SPECS:
            assert spec.levels
            assert spec.gates


class TestBuilder:
    def test_smoke_build_public_private(self):
        builder = BenchmarkBuilder(seed=42, scale=0.02)
        public, private, provenance = builder.build()
        assert len(public) + len(private) >= 20
        for t in public:
            assert t["reference"] is None and t["tests"] is None
            assert t["canary"].startswith("PRISMABENCH-CANARY-")
            assert t["instruction"] and t["context"]["schema"]
        for t in private:
            assert t["reference"]
        assert provenance["canaries"] == sorted(set(provenance["canaries"]))
        assert len(provenance["provenance_hashes"]) == len(public) + len(private)

    def test_public_shape_withholds_reference(self):
        builder = BenchmarkBuilder(seed=1, scale=0.01)
        public, _private, _prov = builder.build()
        task = next(iter(public))
        shape = public_task_shape({**task, "reference": "SECRET", "tests": {"x": 1}})
        assert shape["reference"] is None and shape["tests"] is None


class TestEvaluator:
    def _fixture(self, tmp_path: Path):
        builder = BenchmarkBuilder(seed=3, scale=0.01)
        public, private, _prov = builder.build()
        tasks_path = tmp_path / "public.jsonl"
        keys_path = tmp_path / "keys.jsonl"
        write_jsonl(tasks_path, public)
        write_jsonl(keys_path, private)
        return public, private, tasks_path, keys_path

    def test_reference_index(self, tmp_path):
        public, private, tasks_path, keys_path = self._fixture(tmp_path)
        idx = load_reference_index(keys_path, read_jsonl(tasks_path))
        assert set(idx) == {t["bench_id"] for t in private}

    def test_score_task_static_mode(self, tmp_path):
        public, private, tasks_path, keys_path = self._fixture(tmp_path)
        refs = load_reference_index(keys_path, public)
        task = public[0]
        registry = default_registry()
        light = LightValidator()

        good = score_task(
            task,
            {"bench_id": task["bench_id"], "output": "const ok = 1;"},
            refs.get(task["bench_id"]),
            None,
            light,
            registry,
        )
        bad = score_task(
            task,
            {"bench_id": task["bench_id"], "output": "%%%nope((("},
            refs.get(task["bench_id"]),
            None,
            light,
            registry,
        )
        assert isinstance(good["composite"], float)
        assert good["composite"] >= bad["composite"]

    def test_leaderboard_render(self, tmp_path):
        results = {
            "model": "unit-test",
            "mode": "static",
            "aggregate": {"n_tasks": 4, "pass@1": {"rate": 0.25, "wilson95": [0.05, 0.6]}, "code_bleu": 0.42},
        }
        p = tmp_path / "res.json"
        write_json(p, results)
        entries = collect(tmp_path)
        md = render_markdown(entries)
        assert "unit-test" in md and "static" in md
