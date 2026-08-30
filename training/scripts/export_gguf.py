#!/usr/bin/env python3
"""Export a merged model to GGUF (optionally quantized) via llama.cpp.

    python training/scripts/export_gguf.py \
        --model runs/prismacoder-3b-qlora-sft/merged \
        --out exports/PrismaCoder-3B-Q8_0.gguf \
        [--quant q8_0] [--llamacpp-dir third_party/llama.cpp]

Steps: clone llama.cpp (pinned commit) if absent -> convert_hf_to_gguf.py ->
optional `llama-quantize`.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

LLAMACPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMACPP_PIN = "b4456"  # pin for reproducible conversion


def ensure_llamacpp(dirpath: Path) -> Path:
    if dirpath.exists() and (dirpath / "convert_hf_to_gguf.py").exists():
        return dirpath
    if not shutil.which("git"):
        raise RuntimeError("git is required to fetch llama.cpp for GGUF conversion")
    subprocess.run(["git", "clone", LLAMACPP_REPO, str(dirpath)], check=True)
    subprocess.run(["git", "-C", str(dirpath), "checkout", LLAMACPP_PIN], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(dirpath / "requirements.txt")], check=True
    )
    return dirpath


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export merged model to GGUF")
    ap.add_argument("--model", required=True, help="merged HF model dir")
    ap.add_argument("--out", required=True, help="output .gguf path (f16/f32 bf)")
    ap.add_argument(
        "--quant",
        default="",
        choices=["", "q8_0", "q4_k_m", "q5_k_m"],
        help="post-quantize with llama-quantize",
    )
    ap.add_argument("--llamacpp-dir", default="third_party/llama.cpp")
    args = ap.parse_args(argv)

    model_dir = Path(args.model)
    if not (model_dir / "config.json").exists():
        print(f"not a HF model dir (missing config.json): {model_dir}")
        return 2
    lc = ensure_llamacpp(Path(args.llamacpp_dir))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print("converting to GGUF (f16)...")
    subprocess.run(
        [
            sys.executable,
            str(lc / "convert_hf_to_gguf.py"),
            str(model_dir),
            "--outfile",
            str(out),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    if args.quant:
        quantized = out.with_name(out.stem + f"-{args.quant.upper()}.gguf")
        # build or reuse llama-quantize binary
        binpath = lc / "build" / "bin" / "llama-quantize"
        if not binpath.exists():
            print("building llama.cpp binaries (cmake)...")
            (lc / "build").mkdir(exist_ok=True)
            subprocess.run(["cmake", "-B", str(lc / "build"), "-S", str(lc), "-DGGML_NATIVE=ON"], check=True)
            subprocess.run(["cmake", "--build", str(lc / "build"), "--config", "Release", "-j"], check=True)
        print(f"quantizing -> {quantized}")
        subprocess.run([str(binpath), str(out), str(quantized), args.quant], check=True)
        out.unlink(missing_ok=True)
        print(f"done: {quantized}")
    else:
        print(f"done: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
