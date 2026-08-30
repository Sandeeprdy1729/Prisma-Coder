"""Shared fixtures."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture()
def sample_schema() -> str:
    return """generator client {
  provider = "prisma-client"
  output  = "../app/generated/prisma"
}

datasource db {
  provider = "sqlite"
  url      = env("DATABASE_URL")
}

model User {
  id        Int      @id @default(autoincrement())
  email     String   @unique
  name      String?
  createdAt DateTime @default(now())
  posts     Post[]
}

model Post {
  id       Int     @id @default(autoincrement())
  title    String
  authorId Int
  author   User    @relation(fields: [authorId], references: [id])
}
"""


@pytest.fixture()
def sample_record(sample_schema: str) -> dict:
    return {
        "id": "pf-00001-0000001",
        "schema_version": "1.0",
        "source": {
            "track": "synthetic",
            "origin_repo": "",
            "url": "",
            "license": "MIT",
            "commit": "",
            "path": "",
            "provenance_hash": "sha256:abc",
        },
        "domain": "ecommerce",
        "task": {
            "type": "crud",
            "category": "T4_client_crud",
            "difficulty": "L1",
            "categories": ["T4_client_crud"],
            "instruction": "List users filtered by email, newest first, limited to 20 results.",
            "context": {"schema": sample_schema, "dialect": "sqlite", "files": []},
            "input": "",
            "reference": "const x = prisma.user.findMany({});",
            "persona": "junior",
            "tests": {},
            "destructive_risk": False,
        },
        "verification": {"status": "unverified", "gates": {}, "runtime_ms": 0, "verifier_version": "1.0.0"},
        "difficulty_scores": {"models": 2, "relations": 1, "nesting": 1, "ops": 1, "ambiguity": 0},
        "style_tags": ["plain_list"],
        "meta": {"created": "", "generator_version": "1.0.0", "seed": 1, "tags": [], "canary": ""},
    }
