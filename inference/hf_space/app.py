"""Self-contained HuggingFace Space app (app.py convention).

Copy this file + README into a Space (SDK: gradio4). Loads the published
PrismaCoder-3B checkpoint from the Hub - no repo code required.
"""

from __future__ import annotations

import os

import gradio as gr

MODEL_ID = os.environ.get("PRISMACODER_MODEL", "prismacoder/PrismaCoder-3B")
EXAMPLE_SCHEMA = """datasource db {
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

SYSTEM_PROMPT = (
    "You are PrismaCoder, an expert in the Prisma ORM. You write schema.prisma (PSL) and "
    "TypeScript Prisma Client code that compiles, validates, and executes correctly. "
    "Output only the requested artifact inside a fenced code block. Use Prisma v7 conventions."
)

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mdl = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        mdl.eval()
        _pipeline = (mdl, tok)
    return _pipeline


def generate(instruction: str, schema: str, temperature: float):
    if not instruction.strip():
        return "Enter a request first.", ""
    mdl, tok = _load()
    content = instruction + (f"\n\n```prisma\n{schema}\n```" if schema.strip() else "")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-4),
        top_p=0.9,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    text = tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text, "generated (static validation runs server-side in the full API)"


with gr.Blocks(title="PrismaCoder") as ui:
    gr.Markdown(
        "# PrismaCoder-3B\nFirst open-source Text-to-Prisma model "
        "(Qwen2.5-Coder-3B base, QLoRA SFT on PrismaForge)."
    )
    instruction = gr.Textbox(lines=4, label="Request")
    schema = gr.Code(lines=10, label="schema.prisma (optional)", value=EXAMPLE_SCHEMA, language="sql")
    temperature = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Temperature")
    btn = gr.Button("Generate", variant="primary")
    out_code = gr.Code(lines=18, label="Artifact", language="typescript")
    out_md = gr.Markdown()
    btn.click(generate, [instruction, schema, temperature], [out_code, out_md])

if __name__ == "__main__":
    ui.queue().launch(server_name="0.0.0.0", server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
