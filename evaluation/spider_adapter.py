#!/usr/bin/env python3
"""Cross-benchmark comparison: Spider / BIRD adapters (report Part 13).

Spider (Yu et al., EMNLP 2018 dev split) and BIRD (Li et al., NeurIPS 2023)
are Text-to-SQL benchmarks; they serve as the *context anchors* for the
report's related-work claims and for measuring how a Prisma-tuned model
transfers toward raw SQL. The adapters:

1. load the official public json/jsonl dev sets,
2. run a model through the same chat pipeline used for Prisma tasks,
3. emit predictions in each benchmark's native evaluation format
   (Spider: list-of-{db_id, query, predicted_sql} for the official
   evaluation_scripts/test-suite-sql-eval; BIRD: {db_id, question,
   evidence, SQL, predicted_SQL} rows for their official runner).

Execution-accuracy computation itself is delegated to the official
evaluation scripts of each benchmark (pinned in docs/REPRODUCIBILITY.md) -
we do not re-implement their SQL execution semantics, only the plumbing.

Usage:
    python evaluation/spider_adapter.py --data spider/dev.json --databases spider/databases \
        --model runs/.../merged --limit 100 --out reports/spider_predictions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def load_examples(path: Path, limit: int) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    return rows[:limit] if limit else rows


def build_prompt(question: str, schema_ddl: str, evidence: str = "") -> str:
    """Spider/BIRD prompt: DDL context + question (+ BIRD evidence hint)."""
    parts = [
        "You are an expert SQL generator. Given the database schema and the question, "
        "output one SQLite query in a ```sql code block and nothing else.",
        f"Schema:\n{schema_ddl}",
        f"Question: {question}",
    ]
    if evidence:
        parts.append(f"Hint: {evidence}")
    return "\n\n".join(parts)


def extract_sql(text: str) -> str:
    import re

    blocks = re.findall(r"```sql\n(.*?)```", text, re.S)
    if blocks:
        return blocks[0].strip().rstrip(";") + ";"
    return text.strip().rstrip(";") + ";"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spider/BIRD cross-benchmark adapter")
    ap.add_argument("--benchmark", required=True, choices=["spider", "bird"])
    ap.add_argument("--data", required=True, help="dev.json (spider) or dev.jsonl (bird)")
    ap.add_argument("--databases", default="", help="databases dir (for schema DDL files)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    examples = load_examples(Path(args.data), args.limit)
    predictions = []

    # lazy load the pipeline once
    from prismacoder.inference.pipeline import PrismaCoderPipeline

    pipe = PrismaCoderPipeline.from_local(args.model, max_new_tokens=256, temperature=0.0, repair=False)

    for ex in examples:
        db_id = ex.get("db_id", "")
        question = ex.get("question", "")
        evidence = ex.get("evidence", "") if args.benchmark == "bird" else ""
        ddl = ""
        if args.databases:
            ddl_path = Path(args.databases) / db_id / "schema.sql"
            if ddl_path.exists():
                ddl = ddl_path.read_text(encoding="utf-8", errors="replace")[:4000]
        text = pipe.raw_generate(pipe._build_messages(build_prompt(question, ddl, evidence), ""))
        row = {"db_id": db_id, "question": question, "predicted_SQL": extract_sql(text)}
        if args.benchmark == "spider":
            row["query"] = ex.get("query", "")
            row["predicted_sql"] = row["predicted_SQL"]
        else:
            row["evidence"] = evidence
            row["SQL"] = ex.get("SQL", "")
        predictions.append(row)

    from prismacoder.io_utils import write_json

    write_json(args.out, predictions)
    print(f"wrote {len(predictions)} predictions -> {args.out}")
    print(
        "next: run the benchmark's OFFICIAL evaluation script on this file "
        "(see docs/REPRODUCIBILITY.md for pinned versions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
