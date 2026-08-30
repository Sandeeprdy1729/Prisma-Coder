"""FastAPI inference server (REST + SSE streaming + validation endpoint).

    uvicorn inference.server.app:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health            liveness + model info
    POST /v1/generate       {instruction, schema?, options?} -> artifact + verdict
    POST /v1/generate/stream  same payload, SSE token stream (no repair loop)
    POST /v1/validate       {schema, artifact} -> gate verdict (static/full)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from functools import lru_cache

log = logging.getLogger("prismacoder.server")


def _get_settings() -> dict:
    return {
        "model": os.environ.get("PRISMACODER_MODEL", "prismacoder/PrismaCoder-3B"),
        "adapter": os.environ.get("PRISMACODER_ADAPTER", ""),
        "full_gates": os.environ.get("PRISMACODER_FULL_GATES", "0") == "1",
        "repair_attempts": int(os.environ.get("PRISMACODER_REPAIR_ATTEMPTS", "3")),
        "max_new_tokens": int(os.environ.get("PRISMACODER_MAX_NEW_TOKENS", "1024")),
    }


@lru_cache(maxsize=1)
def get_pipeline():
    from prismacoder.inference.pipeline import PrismaCoderPipeline

    cfg = _get_settings()
    if cfg["adapter"]:
        pipe = PrismaCoderPipeline.from_base_and_adapter(
            cfg["model"],
            cfg["adapter"],
            use_full_gates=cfg["full_gates"],
            repair_attempts=cfg["repair_attempts"],
            max_new_tokens=cfg["max_new_tokens"],
        )
    else:
        pipe = PrismaCoderPipeline.from_local(
            cfg["model"],
            use_full_gates=cfg["full_gates"],
            repair_attempts=cfg["repair_attempts"],
            max_new_tokens=cfg["max_new_tokens"],
        )
    log.info("pipeline ready: model=%s full_gates=%s", cfg["model"], cfg["full_gates"])
    return pipe


def create_app():
    try:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse
    except ImportError as e:
        raise ImportError("serve extras required: pip install -e '.[serve]'") from e

    from prismacoder.inference.server.schemas import (  # noqa: E402
        GenerateRequest,
        GenerateResponse,
        ValidateRequest,
        ValidateResponse,
    )

    from prismacoder.verification.verifier import LightValidator  # noqa: E402

    app = FastAPI(
        title="PrismaCoder API",
        version="1.0.0",
        description="Text-to-Prisma generation with gate-aligned validation and repair",
    )
    light = LightValidator()

    @app.get("/health")
    def health() -> dict:
        cfg = _get_settings()
        return {
            "status": "ok",
            "model": cfg["model"],
            "adapter": bool(cfg["adapter"]),
            "full_gates": cfg["full_gates"],
        }

    @app.post("/v1/generate", response_model=GenerateResponse)
    def generate(req: GenerateRequest) -> GenerateResponse:
        pipe = get_pipeline()
        result = pipe.run(req.instruction, req.schema or "", repair=req.repair and req.max_attempts > 1)
        if req.max_attempts and len(result.attempts) > req.max_attempts:
            result.attempts = result.attempts[: req.max_attempts]
        return GenerateResponse(
            artifact=result.artifact,
            status=result.status,
            attempts=[
                {
                    "attempt": a.attempt,
                    "status": a.status,
                    "gate": a.gate,
                    "error_class": a.error_class,
                    "diagnostics": a.diagnostics,
                }
                for a in result.attempts
            ],
            latency_ms=pipe.last_stats.latency_ms,
            prompt_tokens=pipe.last_stats.prompt_tokens,
            completion_tokens=pipe.last_stats.completion_tokens,
        )

    @app.post("/v1/generate/stream")
    async def generate_stream(req: GenerateRequest) -> StreamingResponse:
        pipe = get_pipeline()
        loop = asyncio.get_event_loop()

        async def event_stream():
            yield _sse({"type": "start", "model": _get_settings()["model"]})
            queue: asyncio.Queue = asyncio.Queue()
            streamer = pipe.stream(req.instruction, req.schema or "")

            async def consume():
                try:
                    for token in streamer:
                        await queue.put(("token", token))
                finally:
                    await queue.put(("done", ""))

            task = loop.create_task(consume())
            buf = []
            while True:
                kind, payload = await queue.get()
                if kind == "token":
                    buf.append(payload)
                    yield _sse({"type": "token", "text": payload})
                else:
                    break
            await task
            verdict = light.validate(req.schema or "", "".join(buf))
            yield _sse({"type": "end", "status": verdict.status, "error_class": verdict.error_class})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/v1/validate", response_model=ValidateResponse)
    def validate(req: ValidateRequest) -> ValidateResponse:
        cfg = _get_settings()
        if cfg["full_gates"]:
            try:
                pipe = get_pipeline()
                if pipe.full_verifier is not None:
                    res = pipe.full_verifier.verify(req.schema, {"artifact.ts": req.artifact})
                    return ValidateResponse(
                        status=res.status,
                        error_class=res.error_class,
                        diagnostics=res.diagnostics[:20],
                        gates=res.gates,
                    )
            except Exception:  # noqa: BLE001
                log.exception("full gates failed; falling back to static")
        res = light.validate(req.schema, req.artifact)
        return ValidateResponse(
            status=res.status, error_class=res.error_class, diagnostics=res.diagnostics[:20], gates=res.gates
        )

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


app = create_app()
