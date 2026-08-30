"""PrismaCoder: open-source Text-to-Prisma models, data (PrismaForge), and benchmark (PrismaBench).

Package layout
--------------
- ``prismacoder.schema``        canonical record schema shared by every pipeline stage
- ``prismacoder.crawler``       GitHub mining (Phase 1, mined track)
- ``prismacoder.extractor``     Prisma artifact extraction from mined repos
- ``prismacoder.cleaning``      license/secret/PII/quality filters + dedup
- ``prismacoder.synthetic``     synthetic track generators (schemas, CRUD, variants)
- ``prismacoder.verification``  the V1-V5 gate chain (validate/generate/tsc/seed/vitest)
- ``prismacoder.eval_metrics``  exact-match, pass@k, BLEU, CodeBLEU, tree edit distance
"""

__version__ = "1.0.0"

from prismacoder.schema import (
    CATEGORY_IDS,
    DIFFICULTY_LEVELS,
    DOMAINS,
    RecordSchemaError,
    canonicalize_record,
    validate_record,
)

__all__ = [
    "__version__",
    "CATEGORY_IDS",
    "DIFFICULTY_LEVELS",
    "DOMAINS",
    "RecordSchemaError",
    "canonicalize_record",
    "validate_record",
]
