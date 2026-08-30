#!/usr/bin/env python3
"""Verifier worker: stdlib HTTP service exposing POST /verify.

Request:
    {"schema": "...", "source_files": {"a.ts": "..."}, "test_files": {...},
     "provider": "sqlite", "gates": ["V1_validate", ...]}

Response:
    {"status": "verified|failed|unverified", "gates": {...},
     "error_class": "E4", "first_failing_gate": "V3_tsc", "runtime_ms": 8123}

Each job runs in a fresh temp SandboxProject (node_modules copied from the
image cache when PRISMACODER_SANDBOX_CACHE is present).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prismacoder.verification.sandbox import SandboxProject, node_available  # noqa: E402
from prismacoder.verification.verifier import Verifier, classify_error  # noqa: E402

CACHE = Path(__file__).resolve().parents[2] / ".." / "node_modules"


class Handler(BaseHTTPRequestHandler):
    server_version = "PrismaVerifier/1.0"

    def log_message(self, fmt, *args):  # quiet-ish
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            node, _ = node_available()
            body = json.dumps({"status": "ok", "node": bool(node)}).encode()
            self._send(200, body)
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):  # noqa: N802
        if self.path != "/verify":
            self._send(404, b'{"error":"not found"}')
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            job = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"invalid json"}')
            return
        started = time.monotonic()
        result = self._run_job(job)
        result["runtime_ms"] = int((time.monotonic() - started) * 1000)
        self._send(200, json.dumps(result).encode())

    def _run_job(self, job: dict) -> dict:
        schema = job.get("schema", "")
        gates = tuple(job.get("gates") or ("V1_validate", "V2_generate", "V3_tsc",
                                           "V4_execute", "V5_vitest"))
        verifier = Verifier(run_install=False, gates=gates)
        sb = SandboxProject.create(
            schema, job.get("source_files") or {}, job.get("test_files") or {},
            env_url=job.get("database_url"))
        try:
            cache = Path(__import__("os").environ.get("PRISMACODER_SANDBOX_CACHE", ""))
            if cache and cache.exists():
                shutil.copytree(cache, sb.root / "node_modules", dirs_exist_ok=True)
            result = verifier.verify(
                schema,
                job.get("source_files") or {},
                job.get("test_files") or {},
                provider=job.get("provider", "sqlite"),
                migration_sql=job.get("migration_sql"),
                seed_body=job.get("seed_body"),
            )
            return {
                "status": result.status,
                "gates": result.gates,
                "error_class": result.error_class,
                "first_failing_gate": result.first_failing_gate,
                "diagnostics": result.diagnostics[:20],
            }
        except Exception as e:  # noqa: BLE001
            return {"status": "unverified", "gates": {},
                    "error": f"{type(e).__name__}: {e}"}
        finally:
            sb.cleanup()

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PrismaCoder verifier worker")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"verifier worker on {args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
