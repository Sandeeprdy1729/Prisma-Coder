#!/usr/bin/env python3
"""Gradio UI for PrismaCoder (also the base for the HF Space app).

    python inference/gradio_app.py

Env:
    PRISMACODER_MODEL   model dir / hub id (default prismacoder/PrismaCoder-3B)
    PRISMACODER_ADAPTER optional adapter on top of the base
    GRADIO_SERVER_PORT  port (default 7860)
"""

from __future__ import annotations

import os

import gradio as gr

from prismacoder.inference.pipeline import PrismaCoderPipeline

EXAMPLE_SCHEMA = """generator client {
  provider = "prisma-client"
  output  = "../app/generated/prisma"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model Order {
  id        Int      @id @default(autoincrement())
  status    String   @default("PENDING")
  totalCents Int     @default(0)
  userId    Int
  createdAt DateTime @default(now())
}

model User {
  id     Int     @id @default(autoincrement())
  email  String  @unique
  orders Order[]
}"""

PIPE: PrismaCoderPipeline | None = None


def _pipeline() -> PrismaCoderPipeline:
    global PIPE
    if PIPE is None:
        model = os.environ.get("PRISMACODER_MODEL", "prismacoder/PrismaCoder-3B")
        adapter = os.environ.get("PRISMACODER_ADAPTER", "")
        if adapter:
            PIPE = PrismaCoderPipeline.from_base_and_adapter(model, adapter)
        else:
            PIPE = PrismaCoderPipeline.from_local(model)
    return PIPE


def generate(instruction: str, schema: str, repair: bool, temperature: float):
    if not instruction.strip():
        return "Please enter a natural-language request.", "", ""
    pipe = _pipeline()
    pipe.temperature = temperature
    result = pipe.run(instruction, schema or "", repair=repair)
    summary = result.error_summary
    verdict_md = (
        f"**status**: `{result.status}` | attempts: {summary['attempts']} | "
        f"final gate: `{summary['final_gate'] or '-'}` | "
        f"error class: `{summary['final_error_class'] or '-'}`\n\n"
        f"{summary['taxonomy_meaning']}"
    )
    return result.artifact, verdict_md, f"{pipe.last_stats.latency_ms} ms"


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PrismaCoder", theme=gr.themes.Soft()) as ui:
        gr.Markdown(
            "# PrismaCoder - Text to Prisma ORM\n"
            "Natural language -> validated `schema.prisma` / Prisma Client TypeScript. "
            "Outputs are checked against the provided schema (symbol grounding) and "
            "repaired automatically on failure."
        )
        with gr.Row():
            with gr.Column(scale=5):
                instruction = gr.Textbox(
                    lines=4,
                    label="Request (natural language)",
                    placeholder="List the 20 most recent orders with their items, "
                    "newest first, excluding cancelled ones.",
                )
                schema = gr.Code(
                    lines=14, label="schema.prisma context (optional)", value=EXAMPLE_SCHEMA, language="sql"
                )
                with gr.Row():
                    repair = gr.Checkbox(value=True, label="Auto-repair on failure")
                    temperature = gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature")
                btn = gr.Button("Generate", variant="primary")
            with gr.Column(scale=6):
                output = gr.Code(lines=20, label="Generated artifact", language="typescript")
                verdict = gr.Markdown(label="Verdict")
                stats = gr.Markdown()
        btn.click(generate, [instruction, schema, repair, temperature], [output, verdict, stats])
        gr.Examples(
            [
                [
                    "Create a new order for user with email 'a@b.com' with total 1999 cents.",
                    EXAMPLE_SCHEMA,
                    True,
                    0.2,
                ],
                [
                    "List users with more than 3 orders, include their order counts.",
                    EXAMPLE_SCHEMA,
                    True,
                    0.2,
                ],
            ],
            [instruction, schema, repair, temperature],
        )
    return ui


if __name__ == "__main__":
    build_ui().queue().launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    )
