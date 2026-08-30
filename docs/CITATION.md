# Citation

If you use PrismaCoder, PrismaForge, or PrismaBench, please cite:

```bibtex
@techreport{prismacoder2026,
  title        = {Text-to-Prisma {ORM} Generation: A Research Study and
                  Technical Roadmap for PrismaCoder-3B, PrismaForge, and
                  PrismaBench},
  author       = {PrismaCoder Team},
  institution  = {PrismaCoder Project},
  year         = {2026},
  month        = {August},
  number       = {PC-TR-2026-01},
  url          = {https://github.com/prismacoder/PrismaCoder},
  abstract     = {We present the first open-source program for Text-to-Prisma:
                  PrismaForge, a 120k-record execution-verified corpus built by a
                  four-gate verifier chain (prisma validate, prisma generate, tsc,
                  database execution); PrismaBench, a 1,200-task benchmark across
                  12 task categories, 3 mechanical difficulty levels, and 12
                  domains; and PrismaCoder-3B, a Qwen2.5-Coder-3B fine-tune trained
                  with continued pretraining, supervised fine-tuning, and verifier-
                  labeled DPO.}
}
```

## Artifact-specific citations

```bibtex
@misc{prismaforge2026,
  title  = {PrismaForge: An Execution-Verified Corpus for Text-to-Prisma},
  author = {PrismaCoder Team},
  year   = {2026},
  howpublished = {\\url{https://huggingface.co/datasets/prismacoder/PrismaForge}}
}

@misc{prismabench2026,
  title  = {PrismaBench: An Execution-Based Benchmark for Text-to-Prisma Generation},
  author = {PrismaCoder Team},
  year   = {2026},
  howpublished = {\\url{https://huggingface.co/datasets/prismacoder/PrismaBench}}
}

@software{prismacoder3b2026,
  title  = {PrismaCoder-3B: A Fine-tuned 3B Model for Prisma Schema and Client Generation},
  author = {PrismaCoder Team},
  year   = {2026},
  url    = {https://huggingface.co/prismacoder/PrismaCoder-3B}
}
```

When building on the evaluation methodology (unbiased pass@k, gate profiles,
E1-E10 taxonomy), also cite the underlying methodology sources listed in the
research report's bibliography (Chen et al. 2021 for pass@k; Spider and BIRD
for cross-benchmark context).
