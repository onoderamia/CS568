"""
Flask API server for SmolLM2 + BPO LoRA prompt optimization.
Runs on http://localhost:5001

Endpoints:
  POST /api/optimize  { "prompt": "..." }  ->  { "optimized": "..." }
"""

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)
CORS(app)

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "models/smollm_bpo_lora"

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32

print(f"[server] Loading model on {device}...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
print("[server] Model ready.")


def run_optimize(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite user prompts to be clearer and more specific "
                "so an AI will give better answers. Output only the rewritten prompt, nothing else. "
                "Limit the response to 64 tokens or less."
            ),
        },
        {"role": "user", "content": f"Optimize this prompt: {prompt}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    reply = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    for stop in ["<|im_end|>", "<|endoftext|>", "\n\n"]:
        if stop in reply:
            reply = reply.split(stop)[0].strip()
    return reply


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        return jsonify({"optimized": run_optimize(prompt)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=False)
