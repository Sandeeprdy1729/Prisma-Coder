"""Synthetic track package: grammar sampler, perturbation engine, task generators.

Division of labor follows report 9.7: *mechanics write answers* - every
reference solution is composed from templates over the sampler's internal
representation (never by an LLM), then passes the V1-V5 gates. The optional
teacher-LLM verbalizer only rewrites the *request* side (see
``instruction_gen.TeacherHook``).
"""

from prismacoder.synthetic.augment import Augmenter
from prismacoder.synthetic.generator import GenerationReport, SyntheticGenerator
from prismacoder.synthetic.schema_gen import PerturbationEngine, SchemaSampler

__all__ = [
    "SyntheticGenerator",
    "GenerationReport",
    "SchemaSampler",
    "PerturbationEngine",
    "Augmenter",
]
