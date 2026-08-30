# `notebooks/`

| Notebook | Purpose |
|---|---|
| [`kaggle/prismacoder_qlora_kaggle.ipynb`](../kaggle/) | **Fine-tuning notebook** - end-to-end QLoRA on Kaggle: install -> data -> 4-bit model -> train -> static eval -> merge -> push -> GGUF |
| [`02_inference.ipynb`](02_inference.ipynb) | Inference: NL (+schema) -> generate -> validate -> auto-repair -> final code |
| [`03_evaluation.ipynb`](03_evaluation.ipynb) | PrismaBench scoring, per-category reports, error analysis (E1-E10), charts, leaderboard; optional Spider/BIRD adapters |

All notebooks are validated CI artifacts: JSON parses and every code cell
compiles (`tests/training_tests/test_notebooks.py`). They run top-to-bottom
without manual edits - GPU sessions are only required where noted.
