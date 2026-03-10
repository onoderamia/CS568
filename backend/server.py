"""
Flask API server for SmolLM2 + LoRA adapters.
Runs on http://localhost:5001

Endpoints:
  POST /api/optimize       { "prompt": "..." }  ->  { "optimized": "..." }
  POST /api/score_overall  { "prompt": "..." }  ->  { "overall": 0-100 }
  POST /api/score_dims     { "prompt": "..." }  ->  {
                              "overall": 0-100,
                              "scores": { "clarity": ..., "specificity": ..., "ambiguity": ..., "tone": ... }
                            }
"""

import re

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

print(f"[server] Loading base model on {device}...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=dtype)

print("[server] Loading BPO (optimize) LoRA adapter...")
opt_model = PeftModel.from_pretrained(base_model, BPO_ADAPTER_PATH)
opt_model.to(device)
opt_model.eval()
opt_tokenizer = AutoTokenizer.from_pretrained(BPO_ADAPTER_PATH)

print("[server] Loading ranked (scoring) LoRA adapter...")
score_model = PeftModel.from_pretrained(base_model, RANKED_ADAPTER_PATH)
score_model.to(device)
score_model.eval()
score_tokenizer = AutoTokenizer.from_pretrained(RANKED_ADAPTER_PATH)

print("[server] All models ready.")


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
    input_text = opt_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = opt_tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = opt_model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=opt_tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    reply = opt_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    for stop in ["<|im_end|>", "<|endoftext|>", "\n\n"]:
        if stop in reply:
            reply = reply.split(stop)[0].strip()
    return reply


def _parse_numeric_score(text: str) -> float:
    """Extract a numeric score from model output, defaulting to 1.0–5.0 range."""
    match = re.search(r"([-+]?\d+(\.\d+)?)", text)
    if not match:
        # Fallback if model outputs something unexpected
        return 3.0
    value = float(match.group(1))
    # Clamp to [1, 5] since we trained on that range
    return max(1.0, min(5.0, value))


def _scale_to_100(score_1_to_5: float) -> int:
    """Map a 1–5 score to an integer 0–100."""
    # 1 -> 0, 5 -> 100, linear
    return int(round((score_1_to_5 - 1.0) / 4.0 * 100))


def run_score_overall(prompt: str) -> int:
    """Return overall quality score scaled to 0–100."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt-quality rater. "
                "Given a user prompt, you output a single quality score between 1 and 5, "
                "where 1 is very poor and 5 is excellent. "
                "Respond ONLY with the numeric score (e.g., 3.8), no explanation."
            ),
        },
        {
            "role": "user",
            "content": f"Rate the quality of this prompt on a 1 to 5 scale: {prompt}",
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


def run_score_dimension(prompt: str, dimension: str) -> int:
    """Return a single dimension score scaled to 0–100."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert prompt-quality rater. "
                "You rate a specific dimension of a prompt between 1 and 5, "
                "where 1 is very poor and 5 is excellent. "
                "Respond ONLY with the numeric score (e.g., 3.8), no explanation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Rate the {dimension} of the following prompt on a 1 to 5 scale: {prompt}"
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


@app.route("/api/score_overall", methods=["POST"])
def score_overall():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        overall = run_score_overall(prompt)
        return jsonify({"overall": overall})
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
        # Also compute an overall score from the same model for convenience
        overall = run_score_overall(prompt)
        return jsonify({"overall": overall, "scores": scores})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=False)
