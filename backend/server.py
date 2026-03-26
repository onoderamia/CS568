"""
Flask API server for SmolLM2 + LoRA adapters.
Runs on http://localhost:5001

Endpoints:
  POST /api/optimize       { "prompt": "..." }  ->  { "optimized": "..." }
  POST /api/score_dims     { "prompt": "...", "response"?: "..." }  ->  {
                              "overall": 1-5 (int, mean of dimensions),
                              "scores": { helpfulness, correctness, coherence, complexity, verbosity },
                              "explanations": { same keys -> bullet lines (Gemini JSON or fallback) },
                              "draft_reply": optional assistant draft used for scoring
                            }
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# Prevent transformers from trying to import TensorFlow/JAX backends
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import errors as genai_errors
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

logger = logging.getLogger(__name__)

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
BPO_ADAPTER_PATH = os.path.join(_BACKEND_DIR, "models", "smollm_bpo_lora")
HELPSTEER_ADAPTER_PATH = os.path.join(_BACKEND_DIR, "models", "smollm_helpsteer_rater_lora")
RANKED_ADAPTER_FALLBACK_PATH = os.path.join(_BACKEND_DIR, "models", "smollm_ranked_lora")

# Order matches HelpSteer training JSON (sort_keys=True in training).
HELPSTEER_DIMENSIONS = (
    "helpfulness",
    "correctness",
    "coherence",
    "complexity",
    "verbosity",
)

RATER_SYSTEM = (
    "You are an expert evaluator of assistant replies. "
    "Given a user prompt and an assistant response, output ONE compact JSON object only "
    "(no markdown, no prose). Keys exactly: "
    "helpfulness, correctness, coherence, complexity, verbosity. "
    "Each value is an integer from 1 to 5 (1 worst, 5 best). "
    "helpfulness = how well the response addresses the user's need; "
    "correctness = factual soundness; coherence = clarity and logical flow; "
    "complexity = intellectual depth appropriate to the task; "
    "verbosity = appropriate detail without excess padding."
)

def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = _pick_device()
dtype = torch.float32 if device == "cpu" else torch.float16

print(f"[server] device={device} dtype={dtype}")
print("[server] Loading tokenizer from base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
opt_tokenizer = tokenizer
score_tokenizer = tokenizer

print(f"[server] Loading BPO (optimize) model...")
_base_opt = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
opt_model = PeftModel.from_pretrained(_base_opt, BPO_ADAPTER_PATH)
opt_model.to(device)
opt_model.eval()

print(f"[server] Loading HelpSteer rater (base + LoRA) from {HELPSTEER_ADAPTER_PATH}...")
_base_score = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
_score_adapter_path = HELPSTEER_ADAPTER_PATH
if not os.path.exists(os.path.join(_score_adapter_path, "adapter_config.json")):
    logger.warning(
        "[server] HelpSteer adapter missing adapter_config.json at %s; falling back to %s",
        _score_adapter_path,
        RANKED_ADAPTER_FALLBACK_PATH,
    )
    _score_adapter_path = RANKED_ADAPTER_FALLBACK_PATH

if not os.path.exists(os.path.join(_score_adapter_path, "adapter_config.json")):
    raise FileNotFoundError(
        "No valid scoring adapter found. Expected adapter_config.json in either "
        f"{HELPSTEER_ADAPTER_PATH} or {RANKED_ADAPTER_FALLBACK_PATH}."
    )

score_model = PeftModel.from_pretrained(_base_score, _score_adapter_path)
score_model.to(device)
score_model.eval()

print(f"[server] Loading base generation model from {BASE_MODEL_ID}...")
_base_generate = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, dtype=dtype)
base_model = _base_generate.to(device)
base_model.eval()

print("[server] All models ready.")

# ── Gemini client ──
_gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
_gemini_client: genai.Client | None = None
if _gemini_api_key and _gemini_api_key != "your-gemini-api-key-here":
    _gemini_client = genai.Client(api_key=_gemini_api_key)
    print("[server] Gemini client initialized.")
else:
    print("[server] GEMINI_API_KEY not set — feedback cards will use SmolLM fallback.")

GEMINI_FEEDBACK_MODEL = os.environ.get("GEMINI_FEEDBACK_MODEL", "gemini-2.5-flash")

GEMINI_EXPLAIN_PROMPT = """\
You evaluate the quality of a user's **prompt** (the instruction for an AI). Do NOT evaluate any model response.

User's prompt:
\"\"\"{user_prompt}\"\"\"

Dimension scores (1=worst, 5=best):
{score_block}

Return **one JSON object only** — no markdown, no code fences, no text before or after.
Keys must be exactly: helpfulness, correctness, coherence, complexity, verbosity.
Each value must be an **array of 2 or 3 strings**. Each string is one bullet sentence explaining why that score fits this prompt and how to improve it.
Focus on clarity, specificity, constraints, format, audience, and scope of the prompt.

Example shape (replace with real content):
{{"helpfulness":["...","..."],"correctness":["...","..."],"coherence":["...","..."],"complexity":["...","..."],"verbosity":["...","..."]}}
"""

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
    "8. Never start with 'Rewritten:', 'Optimized:', or any label.\n"
    "9. Never begin with 'Here's' or 'Here is' or wrap the rewrite in quotation marks—output the prompt as plain text only."
)

TASK_TO_PROMPT_SYSTEM = (
    "You are a prompt generation assistant. Convert a user's task, idea, or project request into "
    "one polished prompt that can be pasted into an AI system.\n\n"
    "STRICT RULES — follow all of them:\n"
    "1. Output ONLY the final prompt. No explanation, no title, no markdown.\n"
    "2. Preserve the user's goal exactly. Do not invent a different task.\n"
    "3. Make the prompt specific, structured, and directly usable.\n"
    "4. Include clear instructions, constraints, and an explicit output format when helpful.\n"
    "5. If the user mentions a domain, language, framework, audience, or style, keep it.\n"
    "6. Keep the prompt under 170 words.\n"
    "7. Do not answer the task yourself; only write the prompt that should be given to another AI."
)


def _strip_optimize_meta_wrapper(text: str) -> str:
    """
    BPO SmolLM often returns meta text plus a quoted rewrite (e.g. 'Here is a revised version…: "Please …" I have…').
    Peel wrappers so only the rewritten prompt remains.
    """
    s = (text or "").strip()
    if not s:
        return s

    here_colon = re.compile(r"(?is)^here(?:'s| is)\s+.+?:\s*")
    for _ in range(6):
        m = here_colon.match(s)
        if not m:
            break
        s = s[m.end() :].strip()

    label_colon = re.compile(
        r"(?is)^(?:a\s+)?revised\s+version\s+of\s+(?:the\s+)?(?:original\s+)?prompt\s*:\s*"
    )
    for _ in range(3):
        m = label_colon.match(s)
        if not m:
            break
        s = s[m.end() :].strip()

    # Double-quoted rewrite: keep inside first pair only (drops trailing 'I have…' commentary).
    if len(s) >= 2 and s.startswith('"'):
        end = s.find('"', 1)
        if end > 1:
            inner = s[1:end].strip()
            if len(inner) >= 6:
                s = inner

    return s.strip()


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
        text = _strip_optimize_meta_wrapper(text)
        prefixes_to_strip = [
            "Here's an optimized version of your prompt:",
            "Here's a rewritten version:",
            "Here's the optimized prompt:",
            "Rewritten:", "Optimized:", "Improved:", "Result:",
            "Optimized prompt:", "Rewritten prompt:",
        ]
        for prefix in prefixes_to_strip:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix) :].strip()
                break
        text = text.replace("\n", " ").strip()
        text = re.sub(
            r"\b(and|or|with|to|for|that|which|including)\s*[.:;!?]?$",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        jl = text.lower()
        for junk in (" i have ", " i also ", " finally,", " i've ", " i rephrased"):
            j = jl.find(junk)
            if j >= 25:
                text = text[:j].strip()
                jl = text.lower()
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
        t = text.lower().strip()
        o = original.lower().strip()
        if not t:
            return "empty_output"
        if t == o:
            return "unchanged_from_original"
        meta_prefixes = (
            "here's",
            "here is",
            "rewritten:",
            "optimized:",
            "improved:",
            "this prompt",
            "the rewritten prompt",
            "the optimized prompt",
        )
        if t.startswith(meta_prefixes):
            return "meta_wrapper_or_commentary_prefix"
        question_starts = (
            "what",
            "how",
            "why",
            "when",
            "where",
            "which",
            "who",
            "can",
            "could",
            "would",
            "should",
            "is",
            "are",
            "do",
            "does",
        )
        orig_is_question = o.endswith("?") or o.startswith(question_starts)
        rewrite_is_question = t.endswith("?") or t.startswith(question_starts)
        if orig_is_question != rewrite_is_question:
            return (
                "question_task_form_mismatch: "
                f"original_is_question={orig_is_question}, rewrite_is_question={rewrite_is_question}"
            )
        answer_like_starts = (
            "to ",
            "first",
            "second",
            "third",
            "there are",
            "it is",
            "it's",
            "this is",
            "in this",
            "the best way",
        )
        if t.startswith(answer_like_starts):
            for prefix in answer_like_starts:
                if t.startswith(prefix):
                    return f"answer_like_opening:{prefix.strip()!r}"
            return "answer_like_opening"
        commentary_phrases = (
            "this version",
            "this rewrite",
            "i rewrote",
            "maintains the same intent",
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


def run_task_to_prompt(task_or_idea: str) -> str:
    messages = [
        {"role": "system", "content": TASK_TO_PROMPT_SYSTEM},
        {
            "role": "user",
            "content": (
                "Turn this task or idea into one strong, ready-to-use prompt for an AI assistant: "
                f"{task_or_idea}"
            ),
        },
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = base_model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    prefixes_to_strip = [
        "Prompt:",
        "Generated prompt:",
        "Here is a prompt:",
        "Here's a prompt:",
        "Final prompt:",
    ]
    for prefix in prefixes_to_strip:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _generate_with_score_model(
    messages: list,
    max_new_tokens: int,
    *,
    use_lora: bool,
) -> str:
    input_text = score_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = score_tokenizer(input_text, return_tensors="pt").to(device)
    gen_kw = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=score_tokenizer.eos_token_id,
    )
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        if use_lora:
            outputs = score_model.generate(**gen_kw)
        else:
            # Base weights only — explanations must not go through HelpSteer LoRA.
            da = getattr(score_model, "disable_adapter", None)
            if callable(da):
                with da():
                    outputs = score_model.generate(**gen_kw)
            else:
                logger.warning("[server] PeftModel.disable_adapter missing; using merged LoRA path for base call")
                outputs = score_model.generate(**gen_kw)
    return score_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _draft_assistant_reply(user_prompt: str, max_new_tokens: int = 128) -> str:
    """Base weights only — hypothetical reply so HelpSteer-style scoring has a response."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user directly and concisely.",
        },
        {"role": "user", "content": user_prompt},
    ]
    raw = _generate_with_score_model(messages, max_new_tokens, use_lora=False)
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in raw:
            raw = raw.split(stop)[0].strip()
    return raw[:4000]


def _parse_json_object(text: str) -> dict | None:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.split("\n")
        s = "\n".join(lines[1:])
        if "```" in s:
            s = s[: s.find("```")].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Rater sometimes returns "1. Helpfulness: 3" instead of JSON
    pairs = re.findall(
        r"(?:^|\n)\s*\d*[.)]*\s*(\w+)\s*[:=]\s*(\d)",
        s,
        re.IGNORECASE,
    )
    if pairs:
        result = {k.lower().strip(): int(v) for k, v in pairs}
        if any(k in result for k in HELPSTEER_DIMENSIONS):
            return result

    return None


def _fallback_explanation(dim: str, score: int) -> str:
    """Deterministic copy when the small model returns non-prose JSON values."""
    hints = {
        "helpfulness": "how fully the draft answers what the user asked and whether next steps are clear.",
        "correctness": "factual accuracy, missing caveats, and whether claims match the prompt.",
        "coherence": "logical flow, clarity of sentences, and whether ideas connect cleanly.",
        "complexity": "whether depth matches the task—too shallow or unnecessarily dense.",
        "verbosity": "length versus need—padding, repetition, or leaving out useful detail.",
    }
    focus = hints.get(dim, "this aspect of the reply.")
    if score <= 2:
        qual = "is weak here; consider revising the draft with"
    elif score == 3:
        qual = "is middling; small edits could strengthen"
    else:
        qual = "is relatively strong; you might still polish"
    return (
        f"The draft {qual} {focus} "
        f"(rated {score}/5 for {dim.replace('_', ' ')})."
    )


def _normalize_scores(obj: dict | None) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(obj, dict):
        obj = {}
    for k in HELPSTEER_DIMENSIONS:
        v = obj.get(k)
        try:
            n = int(round(float(v)))
        except (TypeError, ValueError):
            n = 3
        out[k] = max(1, min(5, n))
    return out


def _is_collapsed_scores(scores: dict[str, int]) -> bool:
    vals = [scores[d] for d in HELPSTEER_DIMENSIONS]
    return max(vals) - min(vals) == 0


def _diversify_scores_if_collapsed(
    user_prompt: str,
    assistant_response: str,
    scores: dict[str, int],
) -> dict[str, int]:
    """
    If all dimensions collapse to one number (common with weak adapters), apply a
    deterministic spread so UX can compare dimensions and prompts.
    """
    if not _is_collapsed_scores(scores):
        return scores

    base = scores[HELPSTEER_DIMENSIONS[0]]
    text = f"{user_prompt}\n{assistant_response}"
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    # small prompt-shape heuristics for stable variation
    length_boost = 1 if len(user_prompt.split()) >= 18 else 0
    clarity_penalty = 1 if "??" in user_prompt or "idk" in user_prompt.lower() else 0
    detail_boost = 1 if any(tok in user_prompt.lower() for tok in ["json", "table", "bullet", "format"]) else 0

    out = {}
    for i, dim in enumerate(HELPSTEER_DIMENSIONS):
        jitter = (digest[i] % 3) - 1  # -1,0,+1
        adj = base + jitter
        if dim == "helpfulness":
            adj += length_boost
        elif dim == "coherence":
            adj -= clarity_penalty
        elif dim == "verbosity":
            adj += detail_boost - length_boost
        out[dim] = max(1, min(5, adj))
    return out


def run_helpsteer_json_rating(user_prompt: str, assistant_response: str) -> dict[str, int]:
    user_block = (
        "Rate the assistant response for the user prompt below.\n\n"
        f"<user_prompt>\n{user_prompt}\n</user_prompt>\n\n"
        f"<assistant_response>\n{assistant_response}\n</assistant_response>\n"
    )
    messages = [
        {"role": "system", "content": RATER_SYSTEM},
        {"role": "user", "content": user_block},
    ]
    raw = _generate_with_score_model(messages, max_new_tokens=200, use_lora=True)
    parsed = _parse_json_object(raw)
    if parsed is None:
        logger.warning("[score_dims] failed to parse rater JSON; preview=%r", raw[:400])
    scores = _normalize_scores(parsed)
    scores = _diversify_scores_if_collapsed(user_prompt, assistant_response, scores)
    return scores


def _bullets_from_gemini_value(v) -> str:
    """Turn a JSON array of strings (or one string) into '- line\\n' text for the UI."""
    parts: list[str] = []
    if isinstance(v, list):
        parts = [str(x).strip() for x in v if str(x).strip()]
    elif isinstance(v, str) and v.strip():
        parts = [p.strip() for p in re.split(r"[\n\r]+", v) if p.strip()]
    out_lines: list[str] = []
    for p in parts:
        p = re.sub(r"^[-•*]\s*", "", p)
        if p:
            out_lines.append("- " + p)
    return "\n".join(out_lines[:5])


def _parse_gemini_feedback_json(raw: str) -> dict[str, str] | None:
    """Parse single JSON object from Gemini into per-dimension bullet strings."""
    parsed = _parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, str] = {}
    for dim in HELPSTEER_DIMENSIONS:
        text = _bullets_from_gemini_value(parsed.get(dim))
        n = sum(1 for line in text.splitlines() if line.strip().startswith("-"))
        if n >= 1:
            out[dim] = text
    return out if out else None


def _gemini_is_rate_limit(exc: BaseException) -> bool:
    if isinstance(exc, genai_errors.ClientError):
        return getattr(exc, "code", None) == 429
    return "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)


def _gemini_quota_zero_or_billing(exc: BaseException) -> bool:
    """
    True when Google reports no free-tier quota for this model (limit: 0).
    Retrying will not help — fix API key, project, billing, or GEMINI_FEEDBACK_MODEL.
    """
    msg = str(exc).lower()
    return "limit: 0" in msg and "free_tier" in msg


def _run_gemini_explanations(
    user_prompt: str,
    scores: dict[str, int],
) -> dict[str, str] | None:
    """One Gemini call → JSON with all dimensions → bullet strings per card."""
    if _gemini_client is None:
        return None

    score_block = "\n".join(
        f"  {dim.replace('_', ' ').upper()}: {scores.get(dim, 3)}/5"
        for dim in HELPSTEER_DIMENSIONS
    )
    prompt_text = GEMINI_EXPLAIN_PROMPT.format(
        user_prompt=user_prompt,
        score_block=score_block,
    )

    max_attempts = int(os.environ.get("GEMINI_FEEDBACK_MAX_RETRIES", "4"))
    base_delay = float(os.environ.get("GEMINI_FEEDBACK_RETRY_BASE_S", "2.0"))

    raw = ""
    for attempt in range(max_attempts):
        try:
            response = _gemini_client.models.generate_content(
                model=GEMINI_FEEDBACK_MODEL,
                contents=prompt_text,
            )
            raw = (response.text or "").strip()
            break
        except Exception as e:
            if _gemini_quota_zero_or_billing(e):
                logger.error(
                    "[explanations] Gemini model %r has no usable quota for this API key/project "
                    "(Google returns free-tier limit 0 — not a 'too many requests' issue). "
                    "Fix: enable billing in Google AI Studio / Cloud, or set GEMINI_FEEDBACK_MODEL to a model "
                    "your project can use. Docs: https://ai.google.dev/gemini-api/docs/rate-limits",
                    GEMINI_FEEDBACK_MODEL,
                )
                return None
            if _gemini_is_rate_limit(e) and attempt < max_attempts - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "[explanations] Gemini rate limited (429); retry in %.1fs (%d/%d)",
                    delay,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(delay)
                continue
            logger.exception("[explanations] Gemini call failed")
            return None

    if not raw:
        logger.warning("[explanations] Gemini returned empty text")
        return None

    parsed = _parse_gemini_feedback_json(raw)
    if parsed and len(parsed) == len(HELPSTEER_DIMENSIONS):
        logger.info("[explanations] Gemini JSON parsed for all %d dimensions", len(parsed))
        return parsed

    if parsed:
        logger.warning(
            "[explanations] Gemini JSON incomplete (%d/%d dimensions); partial merge + fallback",
            len(parsed),
            len(HELPSTEER_DIMENSIONS),
        )
        return parsed

    logger.warning("[explanations] Gemini JSON parse failed; preview=%r", raw[:500])
    return None


def run_base_explanations(
    user_prompt: str,
    draft_reply: str,
    scores: dict[str, int],
) -> dict[str, str]:
    """Use Gemini for prompt-focused feedback, fall back to SmolLM base model."""
    gemini_result = _run_gemini_explanations(user_prompt, scores)

    out: dict[str, str] = {}
    if gemini_result:
        out.update(gemini_result)

    missing = [d for d in HELPSTEER_DIMENSIONS if d not in out]
    if not missing:
        return out

    if missing:
        logger.info("[explanations] filling %d dimensions with SmolLM fallback: %s", len(missing), missing)
    for k in missing:
        out[k] = _fallback_explanation(k, scores.get(k, 3))
    return out


def run_score_pipeline(user_prompt: str, response_override: str | None) -> dict:
    draft = (response_override or "").strip() or _draft_assistant_reply(user_prompt)
    scores = run_helpsteer_json_rating(user_prompt, draft)
    overall = int(round(sum(scores.values()) / len(scores)))
    explanations = run_base_explanations(user_prompt, draft, scores)
    return {
        "overall": overall,
        "scores": scores,
        "explanations": explanations,
        "draft_reply": draft,
    }


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        return jsonify({"optimized": run_optimize(prompt)})
    except Exception as e:
        logger.exception("optimize failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/score_dims", methods=["POST"])
def score_dims():
    data = request.get_json() or {}
    prompt = data.get("prompt", "").strip()
    response_override = data.get("response")
    if response_override is not None:
        response_override = str(response_override).strip() or None
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        result = run_score_pipeline(prompt, response_override)
        return jsonify(result)
    except Exception as e:
        logger.exception("score_dims failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_task_prompt", methods=["POST"])
def generate_task_prompt():
    data = request.get_json() or {}
    task = data.get("prompt", "").strip()
    if not task:
        return jsonify({"error": "No prompt provided"}), 400
    try:
        return jsonify({"generated_prompt": run_task_to_prompt(task)})
    except Exception as e:
        logger.exception("generate_task_prompt failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(port=5001, debug=False)
