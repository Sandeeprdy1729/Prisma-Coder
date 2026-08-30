"""Training tests: config lint, data formatting, notebook integrity."""

import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = sorted((REPO_ROOT / "training" / "configs").glob("*.yaml"))

from training.scripts.train_sft import load_config, validate_config  # noqa: E402
from training.trainer.collators import pack_examples  # noqa: E402
from training.trainer.data import (  # noqa: E402
    SoftOrderingMixer,
    build_assistant_message,
    build_user_message,
)


class TestConfigs:
    def test_all_yaml_parse(self):
        assert len(CONFIGS) >= 4
        for p in CONFIGS:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
            assert isinstance(data, dict), p

    def test_table_121_hyperparameters(self):
        ql = load_config(str(REPO_ROOT / "training/configs/qlora_3b.yaml"))
        assert ql["training"]["warmup_ratio"] == 0.03
        assert ql["training"]["lr_scheduler_type"] == "cosine"
        assert ql["data"]["neftune_noise_alpha"] == 5
        full = load_config(str(REPO_ROOT / "training/configs/full_ft.yaml"))
        assert full["training"]["learning_rate"] == 1.5e-5
        dpo = load_config(str(REPO_ROOT / "training/configs/dpo_qlora.yaml"))
        assert dpo["training"]["beta"] == 0.05
        assert dpo["training"]["learning_rate"] == 5.0e-7
        cpt = load_config(str(REPO_ROOT / "training/configs/cpt.yaml"))
        assert cpt["training"]["learning_rate"] == 8.0e-5

    def test_validate_config_flags(self):
        ql = load_config(str(REPO_ROOT / "training/configs/qlora_3b.yaml"))
        assert validate_config(ql) == []
        bad = {
            "model": {"quantization": {"load_in_4bit": False}},
            "training": {"learning_rate": 0.5, "lr_scheduler_type": "linear", "warmup_ratio": 0.5},
            "data": {},
        }
        problems = validate_config(bad)
        assert problems


class TestDataFormatting:
    def test_chat_messages(self, sample_record):
        user = build_user_message(sample_record)
        assert sample_record["task"]["instruction"] in user
        assert "prisma" in user
        assistant = build_assistant_message(sample_record)
        assert assistant

    def test_soft_ordering(self, sample_record):
        mixer = SoftOrderingMixer(seed=1)
        w1 = mixer.weights_for_epoch(1)
        w3 = mixer.weights_for_epoch(3)
        assert w1["L1"] > w1["L3"]
        assert w3["L3"] > w1["L3"]


class TestPacking:
    def test_pack_examples_respects_len(self):
        rows = [{"input_ids": [1, 2, 3], "labels": [-100, -100, 3]} for _ in range(10)]
        packed = pack_examples(rows, max_seq_len=7, pad_token_id=0)
        assert all(len(r["input_ids"]) <= 7 for r in packed)
        total_tokens = sum(len(r["input_ids"]) for r in packed)
        assert total_tokens >= 10 * 3  # separators included, nothing lost


class TestNotebooks:
    """Every committed notebook parses as JSON and code cells compile."""

    def _notebooks(self):
        return sorted((REPO_ROOT / "notebooks").glob("*.ipynb")) + sorted(
            (REPO_ROOT / "kaggle").glob("*.ipynb")
        )

    def test_notebooks_valid(self):
        import ast

        notebooks = self._notebooks()
        assert len(notebooks) == 5
        for nb_path in notebooks:
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            assert nb["nbformat"] == 4, nb_path
            for cell in nb["cells"]:
                if cell["cell_type"] != "code":
                    continue
                py_lines, in_magic = [], False
                for ln in "".join(cell["source"]).splitlines():
                    if in_magic:
                        in_magic = ln.rstrip().endswith("\\")
                        continue
                    if ln.lstrip().startswith(("%", "!")):
                        in_magic = ln.rstrip().endswith("\\")
                        continue
                    py_lines.append(ln)
                ast.parse("\n".join(py_lines))


class TestDeepSpeedConfigs:
    def test_json_configs_valid(self):
        for name in ("zero2.json", "zero3.json"):
            p = REPO_ROOT / "training" / "configs" / "deepspeed" / name
            data = json.loads(p.read_text(encoding="utf-8"))
            assert data["zero_optimization"]["stage"] in (2, 3)
