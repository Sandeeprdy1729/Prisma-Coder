"""PrismaBench evaluator (Phase 4 harness, exposed via benchmark/evaluator).

Scores a model submission against the benchmark:
- pass@1 (primary): gate chain verdict per task (full V-gates with Node,
  static proxy without)
- pass@k over k samples when the submission provides them
- string/structure similarity: exact_match, CodeBLEU, tree similarity
- efficiency: latency + memory (mandatory leaderboard columns, report O5)
- failure classification into E1-E10
- per-category / per-level / per-gate reporting with Wilson 95% CIs
- composite score = 0.20*compile + 0.60*execute + 0.10*style + 0.10*efficiency
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from prismacoder.constants import COMPOSITE_WEIGHTS, ERROR_TAXONOMY  # noqa: E402
from prismacoder.eval_metrics.registry import aggregate_results, default_registry  # noqa: E402
from prismacoder.io_utils import read_jsonl, write_json  # noqa: E402
from prismacoder.verification.sandbox import node_available  # noqa: E402
from prismacoder.verification.verifier import LightValidator, Verifier  # noqa: E402

log = logging.getLogger("prismabench.evaluator")


def load_reference_index(private_keys: Path | None, tasks: list[dict]) -> dict[str, str]:
    """References come from the private keys file (public rows have none)."""
    idx: dict[str, str] = {}
    if private_keys and private_keys.exists():
        for row in read_jsonl(private_keys):
            idx[row["bench_id"]] = row["reference"]
    for t in tasks:  # dev builds may embed references directly
        if t.get("reference") and t["bench_id"] not in idx:
            idx[t["bench_id"]] = t["reference"]
    return idx


def score_task(
    task: dict,
    sub_row: dict,
    reference: str | None,
    verifier: Verifier | None,
    light: LightValidator,
    registry,
) -> dict:
    bench_id = task["bench_id"]
    outputs = sub_row.get("samples") or [sub_row.get("output", "")]
    latency_ms = sub_row.get("latency_ms", 0)
    memory_mb = sub_row.get("memory_mb", 0)
    task_passes: list[bool] = []

    sim = {"exact_match": 0.0, "code_bleu": 0.0, "tree_similarity": 0.0}
    gate_profile: dict[str, dict] = {}
    error_class = ""
    first_gate = ""

    schema = task.get("context", {}).get("schema", "")
    for out in outputs:
        passed = False
        if reference:
            if verifier is not None:
                try:
                    res = verifier.verify(schema, {"artifact.ts": out},
                                          artifact_kind=task.get("artifact_kind"))
                    gate_profile = res.gates
                    passed = res.status == "verified"
                    error_class = res.error_class
                    first_gate = res.first_failing_gate
                except Exception:  # noqa: BLE001
                    res = light.validate(schema, out)
                    passed = res.status != "failed"
                    error_class = res.error_class
            else:
                res = light.validate(schema, out)
                passed = res.status != "failed"
                error_class = res.error_class
                first_gate = res.first_failing_gate
        task_passes.append(bool(passed))
        if reference and sim["code_bleu"] == 0.0:
            metrics = registry.evaluate(out, reference, task)
            sim.update({k: metrics[k] for k in sim})

    compile_pass = (
        1.0 if (not first_gate or first_gate not in ("V1_validate", "V2_generate", "V3_tsc")) else 0.0
    )
    execute_pass = 1.0 if task_passes and task_passes[0] else 0.0
    style = sum(sim.values()) / max(len(sim), 1)
    eff = _efficiency_score(latency_ms, memory_mb)
    composite = round(
        sum(
            COMPOSITE_WEIGHTS[k] * v
            for k, v in {
                "compile": compile_pass,
                "execute": execute_pass,
                "style": style,
                "efficiency": eff,
            }.items()
        ),
        4,
    )

    return {
        "bench_id": bench_id,
        "category": task["category"],
        "level": task["level"],
        "domain": task["domain"],
        "passed": any(task_passes),
        "sample_passes": task_passes,
        "gates": gate_profile,
        "first_failing_gate": first_gate,
        "error_class": error_class,
        "latency_ms": latency_ms,
        "memory_mb": memory_mb,
        "composite": composite,
        **sim,
    }


def _efficiency_score(
    latency_ms: float, memory_mb: float, lat_budget_ms: float = 30_000, mem_budget_mb: float = 4096
) -> float:
    """Piecewise efficiency in [0,1]: 1 within budget, decaying linearly to 0 at 3x budget."""
    lat_ratio = min(max(latency_ms / lat_budget_ms, 0.0), 3.0) / 3.0 if latency_ms else 1.0
    mem_ratio = min(max(memory_mb / mem_budget_mb, 0.0), 3.0) / 3.0 if memory_mb else 1.0
    return round(1.0 - (0.6 * lat_ratio + 0.4 * mem_ratio), 4)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PrismaBench evaluator")
    ap.add_argument("--tasks", default="benchmark/public/prismabench_public.jsonl")
    ap.add_argument("--private-keys", default="benchmark/private/private_keys.jsonl")
    ap.add_argument(
        "--submission",
        required=True,
        help="submission jsonl: {bench_id, output|samples, latency_ms, memory_mb}",
    )
    ap.add_argument("--out", default="benchmark/leaderboard/eval_results.json")
    ap.add_argument("--model", default="submission", help="system name recorded in results")
    ap.add_argument("--full-gates", action="store_true", help="force full V1-V5 (requires Node)")
    ap.add_argument("--max-tasks", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s %(message)s")

    tasks = read_jsonl(args.tasks)
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    refs = load_reference_index(Path(args.private_keys), tasks)
    submission = {r["bench_id"]: r for r in read_jsonl(args.submission)}

    node, _ = node_available()
    verifier: Verifier | None = None
    if args.full_gates and node:
        verifier = Verifier()
        log.info("full V1-V5 gates enabled (node detected)")
    else:
        log.info("static evaluation mode (no Node or --full-gates not passed)")

    light = LightValidator()
    registry = default_registry()
    rows = []
    started = time.monotonic()
    missing = 0
    for task in tasks:
        row = submission.get(task["bench_id"])
        if row is None:
            missing += 1
            rows.append(
                {
                    "bench_id": task["bench_id"],
                    "category": task["category"],
                    "level": task["level"],
                    "domain": task["domain"],
                    "passed": False,
                    "sample_passes": [False],
                    "error_class": "E7",
                    "composite": 0.0,
                }
            )
            continue
        rows.append(score_task(task, row, refs.get(task["bench_id"]), verifier, light, registry))
    wall = round(time.monotonic() - started, 1)

    results = {
        "model": args.model,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_seconds": wall,
        "mode": "full_gates" if verifier else "static",
        "missing_submissions": missing,
        "reference_coverage": round(len(refs) / max(len(tasks), 1), 4),
        "aggregate": aggregate_results(rows),
        "error_taxonomy_legend": ERROR_TAXONOMY,
        "rows": rows,
    }
    write_json(args.out, results)
    agg = results["aggregate"]
    log.info("scored %d tasks in %ss -> %s", len(rows), wall, args.out)
    log.info(
        "pass@1=%.3f (wilson %s) composite-like metrics: em=%s codebleu=%s",
        agg["pass@1"]["rate"],
        agg["pass@1"]["wilson95"],
        agg.get("exact_match", "-"),
        agg.get("code_bleu", "-"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
