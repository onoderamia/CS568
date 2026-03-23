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

import logging
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

logger = logging.getLogger(__name__)

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
    "5. Keep the rewritten prompt under 110 words and aim for 2-3 sentences.\n"
    "6. Improve: specificity, clarity, context, role framing, and actionability.\n"
    "7. Include at least two concrete constraints (for example scope, audience, length, criteria) and an explicit output format hint.\n"
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
    def _generate(messages, max_new_tokens=140):
        input_text = opt_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = opt_tokenizer(input_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = opt_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=opt_tokenizer.eos_token_id,
            )
        input_len = inputs.input_ids.shape[1]
        return opt_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    def _cleanup(text):
        for stop in ["<|im_end|>", "<|endoftext|>"]:
            if stop in text:
                text = text.split(stop)[0].strip()
        prefixes_to_strip = [
            "Here's an optimized version of your prompt:",
            "Here's a rewritten version:",
            "Here's the optimized prompt:",
            "Rewritten:", "Optimized:", "Improved:", "Result:",
            "Optimized prompt:", "Rewritten prompt:",
        ]
        for prefix in prefixes_to_strip:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
                break
        text = text.replace("\n", " ").strip()
        text = re.sub(r"\b(and|or|with|to|for|that|which|including)\s*[.:;!?]?$", "", text, flags=re.IGNORECASE).strip()
        if text and text[-1] not in ".!?":
            last_full_stop = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
            if last_full_stop >= 20:
                text = text[: last_full_stop + 1].strip()
            else:
                text = f"{text}."
        if text.count('"') % 2 == 1:
            text += '"'
        if text.count("(") > text.count(")"):
            text += ")" * (text.count("(") - text.count(")"))
        return text

    def _rewrite_failure_reason(text, original):
        """Return None if rewrite passes validation; otherwise a short reason code for logging."""
        t = text.lower().strip()
        o = original.lower().strip()
        if not t:
            return "empty_output"
        if t == o:
            return "unchanged_from_original"

        # Meta wrappers / commentary means model didn't follow "output only rewrite".
        meta_prefixes = (
            "here's", "here is", "rewritten:", "optimized:", "improved:",
            "this prompt", "the rewritten prompt", "the optimized prompt",
        )
        if t.startswith(meta_prefixes):
            return "meta_wrapper_or_commentary_prefix"

        # Preserve original interaction form (question vs task).
        question_starts = (
            "what", "how", "why", "when", "where", "which", "who",
            "can", "could", "would", "should", "is", "are", "do", "does",
        )
        orig_is_question = o.endswith("?") or o.startswith(question_starts)
        rewrite_is_question = t.endswith("?") or t.startswith(question_starts)
        if orig_is_question != rewrite_is_question:
            return (
                "question_task_form_mismatch: "
                f"original_is_question={orig_is_question}, rewrite_is_question={rewrite_is_question}"
            )

        # Detect answer-like narrative openings (we want rewritten instructions).
        answer_like_starts = (
            "to ", "first", "second", "third", "there are", "it is", "it's",
            "this is", "in this", "the best way",
        )
        if t.startswith(answer_like_starts):
            for prefix in answer_like_starts:
                if t.startswith(prefix):
                    return f"answer_like_opening:{prefix.strip()!r}"
            return "answer_like_opening"

        # Common giveaway that model is explaining rather than rewriting.
        commentary_phrases = (
            "this version", "this rewrite", "i rewrote", "maintains the same intent",
        )
        for phrase in commentary_phrases:
            if phrase in t:
                return f"commentary_phrase:{phrase!r}"

        word_count = len(t.split())
        if word_count < 10:
            return f"too_short:word_count={word_count}"
        return None

    primary_messages = [
        {"role": "system", "content": OPTIMIZE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Optimize this prompt. Keep the same intent, but make the rewritten prompt more specific "
                "and structured in 2-3 sentences with clear constraints and output format guidance: "
                f"{prompt}"
            ),
        },
    ]
    reply = _cleanup(_generate(primary_messages))
    primary_reason = _rewrite_failure_reason(reply, prompt)
    if primary_reason is None:
        return reply

    logger.info(
        "[optimize] retry triggered: primary failed validation (%s); preview=%r",
        primary_reason,
        (reply[:240] + "…") if len(reply) > 240 else reply,
    )

    # Strict retry for malformed / answer-like / too-weak output.
    strict_messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's prompt only. Do not answer it. "
                "Return only the rewritten prompt text in 2-3 sentences."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original prompt: {prompt}\n"
                "Rewrite it with clearer scope, explicit constraints, and output format expectations."
            ),
        },
    ]
    retry = _cleanup(_generate(strict_messages, max_new_tokens=170))
    retry_reason = _rewrite_failure_reason(retry, prompt)
    if retry_reason is not None:
        logger.warning(
            "[optimize] second pass still failed validation (%s); returning it anyway; preview=%r",
            retry_reason,
            (retry[:240] + "…") if len(retry) > 240 else retry,
        )
    else:
        logger.info("[optimize] second pass passed validation")
    return retry

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(port=5001, debug=False)
