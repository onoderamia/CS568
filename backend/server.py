"""
Flask API server for SmolLM2 + LoRA adapters.
Runs on http://localhost:5001

Endpoints:
  POST /api/optimize       { "prompt": "..." }  ->  { "optimized": "..." }
  POST /api/score_dims     { "prompt": "..." }  ->  {
                              "overall": 0-100,
                              "scores": { "clarity": ..., "specificity": ..., "ambiguity": ..., "tone": ... }
                            }
"""

import os
import re

# Prevent transformers from trying to import TensorFlow/JAX backends
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)
CORS(app)

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
BPO_ADAPTER_PATH = "models/smollm_bpo_lora"
RANKED_ADAPTER_PATH = "models/smollm_ranked_lora"

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32

print("[server] Loading tokenizer from base model...")
# Load tokenizer from base model — the adapter tokenizer_configs use a non-standard class
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
opt_tokenizer = tokenizer
score_tokenizer = tokenizer

print(f"[server] Loading BPO (optimize) model on {device}...")
_base_opt = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
opt_model = PeftModel.from_pretrained(_base_opt, BPO_ADAPTER_PATH)
opt_model.to(device)
opt_model.eval()

print(f"[server] Loading ranked (scoring) model on {device}...")
_base_score = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
score_model = PeftModel.from_pretrained(_base_score, RANKED_ADAPTER_PATH)
score_model.to(device)
score_model.eval()

print("[server] All models ready.")

OPTIMIZE_SYSTEM_PROMPT = (
    "You are a prompt optimization assistant. Your sole task is to rewrite the user's prompt "
    "to be clearer, more specific, and more effective for AI systems.\n\n"
    "STRICT RULES — follow all of them:\n"
    "1. Output ONLY the rewritten prompt. No explanations, no preamble, no labels, no commentary.\n"
    "2. Preserve the original user intent exactly — do not change what is being asked.\n"
    "3. Do NOT answer, fulfill, or respond to the content of the prompt — only rewrite it.\n"
    "4. If the original is a question, keep it as a question. If it is a task, keep it as a task.\n"
    "5. Keep the rewritten prompt under 80 words.\n"
    "6. Improve: specificity, clarity, context, role framing, and actionability.\n"
    "7. Add constraints or output format hints where helpful.\n"
    "8. Never start with 'Rewritten:', 'Optimized:', or any label."
)

DIMENSION_GUIDELINES = {
    "clarity": (
        "how clear, direct, and easy to understand the prompt is for an AI assistant. "
        "A score of 5 means the prompt is perfectly clear with no confusion possible. "
        "A score of 1 means the prompt is very hard to parse or understand."
    ),
    "specificity": (
        "how specific, detailed, and precise the prompt is. "
        "A score of 5 means the prompt includes all necessary context, constraints, desired format, and scope. "
        "A score of 1 means the prompt is extremely vague with no useful details."
    ),
    "ambiguity": (
        "how free the prompt is from ambiguity or multiple interpretations. "
        "A score of 5 means the prompt has exactly one clear interpretation. "
        "A score of 1 means the prompt is highly ambiguous and could mean many different things."
    ),
    "tone": (
        "how appropriate, professional, and effective the tone is for eliciting a high-quality AI response. "
        "A score of 5 means the tone is ideal — polite, direct, and sets appropriate expectations. "
        "A score of 1 means the tone is counterproductive, rude, or poorly framed."
    ),
}

SCORE_SYSTEM_PROMPT = (
    "You are an expert prompt-quality rater. "
    "You rate a specific dimension of a prompt on a scale from 1 to 5, "
    "where 1 is very poor and 5 is excellent. "
    "Respond ONLY with the numeric score (e.g., 3.8). No explanation, no text, just the number."
)


def run_optimize(prompt: str) -> str:
    messages = [
        {"role": "system", "content": OPTIMIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Optimize this prompt: {prompt}"},
    ]
    input_text = opt_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = opt_tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = opt_model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=opt_tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    reply = opt_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    # Strip special tokens
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in reply:
            reply = reply.split(stop)[0].strip()
    # Strip meta-commentary prefixes BEFORE cutting on newlines, so we don't lose the content
    prefixes_to_strip = [
        "Here's an optimized version of your prompt:",
        "Here's a rewritten version:",
        "Here's the optimized prompt:",
        "Rewritten:", "Optimized:", "Improved:", "Result:",
        "Optimized prompt:", "Rewritten prompt:",
    ]
    for prefix in prefixes_to_strip:
        if reply.lower().startswith(prefix.lower()):
            reply = reply[len(prefix):].strip()
            break
    return reply


def _parse_numeric_score(text: str) -> float:
    """Extract a numeric score from model output, defaulting to 1.0–5.0 range."""
    match = re.search(r"([-+]?\d+(\.\d+)?)", text)
    if not match:
        return 3.0
    value = float(match.group(1))
    return max(1.0, min(5.0, value))


def _scale_to_100(score_1_to_5: float) -> int:
    """Map a 1–5 score to an integer 0–100."""
    return int(round((score_1_to_5 - 1.0) / 4.0 * 100))


def run_score_dimension(prompt: str, dimension: str) -> int:
    """Return a single dimension score scaled to 0–100."""
    guideline = DIMENSION_GUIDELINES.get(dimension, f"the {dimension} of the prompt")
    messages = [
        {"role": "system", "content": SCORE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Rate the following prompt on a 1 to 5 scale specifically for {guideline}\n\n"
                f"Prompt: {prompt}"
            ),
        },
    ]
    input_text = score_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = score_tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = score_model.generate(
            **inputs,
            max_new_tokens=16,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=score_tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    reply = score_tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True
    ).strip()
    for stop in ["<|im_end|>", "<|endoftext|>", "\n\n"]:
        if stop in reply:
            reply = reply.split(stop)[0].strip()
    score_1_to_5 = _parse_numeric_score(reply)
    return _scale_to_100(score_1_to_5)


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


@app.route("/api/score_dims", methods=["POST"])
def score_dims():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        dims = ["clarity", "specificity", "ambiguity", "tone"]
        scores = {dim: run_score_dimension(prompt, dim) for dim in dims}
        # Overall is the average of all dimension scores
        overall = round(sum(scores.values()) / len(scores))
        return jsonify({"overall": overall, "scores": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=False)
