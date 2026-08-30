"""Pydantic schemas for the PrismaCoder API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    instruction: str = Field(..., min_length=3, description="Natural-language request")
    schema: str = Field("", description="Optional Prisma schema (PSL) context")
    repair: bool = Field(True, description="Run the validate-and-repair loop")
    max_attempts: int = Field(3, ge=1, le=5, description="Repair attempt budget")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_new_tokens: int | None = Field(None, ge=64, le=8192)


class Attempt(BaseModel):
    attempt: int
    status: str
    gate: str
    error_class: str
    diagnostics: list[str] = []


class GenerateResponse(BaseModel):
    artifact: str
    status: str  # verified | failed | unverified
    attempts: list[Attempt]
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int


class ValidateRequest(BaseModel):
    schema: str = Field(..., description="Prisma schema (PSL)")
    artifact: str = Field(..., description="Candidate artifact (PSL or TS)")


class ValidateResponse(BaseModel):
    status: str  # verified | unverified | failed
    error_class: str = ""
    diagnostics: list[str] = []
    gates: dict[str, Any] = {}
