# Base model selection: Qwen2.5-Coder-3B-Instruct

The research brief asked for the best base under ~3B for Text-to-Prisma. The
candidate field and the weighted verdict (weights: Prisma/TS code prior 0.30,
license 0.15, context length 0.15, tooling/PEFT maturity 0.15, instruction
formatting 0.10, multilingual-data hygiene 0.05, size/GPU cost 0.10):

| Candidate | Params | License | Code prior | Context | Notes | Weighted |
|---|---|---|---|---|---|---|
| **Qwen2.5-Coder-3B-Instruct** | 3.09B | Apache-2.0 | very high (dedicated code model line) | 32k | chat template + strong FIM; best PSL/TS exposure of the cohort | **9.0 / 10** |
| Qwen3-4B | 4.0B | Apache-2.0 | high | 32k (128k w/ YaRN) | strongest reasoning per param but above the 3B budget; kept as ablation family member | 8.6 |
| SmolLM3-3B | 3.08B | Apache-2.0 | medium | 64k | strong given size, dual-mode reasoning; thinner TypeScript distribution than Qwen-Coder | 8.2 |
| StarCoder2-3B | 3.0B | BigCode-OpenRAIL-M | high (code-only) | 16k | code-native but base-only (no chat), restrictive license for some org uses | 7.6 |
| Llama 3.2 3B | 3.2B | Llama 3.2 license | medium | 128k | general-purpose; license adds deployment friction | 7.3 |
| Gemma 2 2B | 2.6B | Gemma license | medium-low | 8k | smallest VRAM; weakest code prior; short context | 6.8 |

## Decision

**Qwen2.5-Coder-3B-Instruct** is the primary base for PrismaCoder-3B:

1. **Code-dominant pretraining** with explicit TypeScript/JS density - the
   highest prior on the exact surface (PSL + Prisma Client TS) we fine-tune.
2. **Apache-2.0** - clean redistribution for an Apache-2.0 project.
3. **32k context** - fits repository-scale T8/T10 contexts at 4k training
   window with headroom for inference.
4. **Instruct + chat template out of the box** - SFT starts from a
   well-formatted assistant prior rather than a raw base.
5. **PEFT/bnb maturity** - QLoRA nf4 + FlashAttention-2 paths are
   well-trodden for this family.

## Ablation family (report Part 12)

- Qwen2.5-Coder-3B-Instruct (primary)
- Qwen3-4B (capacity check, above-budget reference point)
- SmolLM3-3B (architecture check)
- StarCoder2-3B (base-only, code-native check)
- Llama 3.2 3B (family check)

The CPT stage (S0) uses **Qwen/Qwen2.5-Coder-3B-Base** to avoid double-SFT on
the instruct prior before domain adaptation.
