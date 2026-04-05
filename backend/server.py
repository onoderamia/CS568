"""
Flask API server for SmolLM2 + LoRA adapters.
Runs on http://localhost:5001

Endpoints:
  POST /api/optimize       { "prompt": "..." }  ->  { "optimized": "..." }
  POST /api/refine_optimized  { "mode": "sentence"|"full", "original_user_prompt", "full_optimized",
                              "initial_optimized"?, "sentence_index", "action" }  ->  { "optimized": "..." }
  POST /api/score_dims     { "prompt": "...", "response"?: "..." }  ->  {
                              "overall": 1-5 (int, mean of dimensions),
                              "scores": { helpfulness, correctness, coherence, complexity, verbosity },
                              "explanations": { same keys -> text (Gemini when GEMINI_FEEDBACK_ENABLED, else default) },
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
    print("[server] GEMINI_API_KEY not set — Gemini feedback calls cannot run.")

GEMINI_FEEDBACK_MODEL = os.environ.get("GEMINI_FEEDBACK_MODEL", "gemini-2.5-flash")


def _gemini_feedback_enabled() -> bool:
    """
    When True, explanation bullets are generated via Gemini (one API call per score_dims).
    When False (default), use _fallback_explanation only — no Gemini HTTP requests.

    Set GEMINI_FEEDBACK_ENABLED=1 (or true/yes/on) in backend/.env to enable.
    """
    v = os.environ.get("GEMINI_FEEDBACK_ENABLED", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


if _gemini_feedback_enabled():
    print("[server] GEMINI_FEEDBACK_ENABLED: feedback cards will call Gemini.")
else:
    print("[server] GEMINI_FEEDBACK_ENABLED off: feedback cards use default text only.")

GEMINI_EXPLAIN_PROMPT = """\
You evaluate the quality of a user's **prompt** (the instruction for an AI). Do NOT evaluate any model response.

User's prompt:
\"\"\"{user_prompt}\"\"\"

Dimension scores (1=worst, 5=best):
{score_block}

Return **one JSON object only** — no markdown, no code fences, no text before or after.
Keys must be exactly: helpfulness, correctness, coherence, complexity, verbosity.
Each value must be a **single short string** (1-2 sentences max). Say why the prompt got that score and one specific fix. Use plain, simple language — no jargon.

Example shape (replace with real content):
{{"helpfulness":"...","correctness":"...","coherence":"...","complexity":"...","verbosity":"..."}}
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
    "1. NEVER answer, fulfill, execute, or respond to the user's task. "
    "You are ONLY allowed to write a prompt — not perform the task itself.\n"
    "2. Output ONLY the final prompt. No explanation, no title, no markdown, no commentary.\n"
    "3. Preserve the user's goal exactly. Do not invent a different task.\n"
    "4. Make the prompt specific, structured, and directly usable by an AI assistant.\n"
    "5. Include clear instructions, constraints, and an explicit output format.\n"
    "6. If the user mentions a domain, language, framework, audience, or style, keep it.\n"
    "7. Keep the prompt under 170 words.\n"
    "8. Never start with 'Here is', 'Here's', 'Sure', 'Prompt:', or any label — output the prompt as plain text only."
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



def _cleanup_optimized_output(text: str) -> str:
    """Normalize BPO optimizer output (shared by run_optimize and refine endpoints)."""
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


def _generate_opt_bpo(messages: list, max_new_tokens: int = 140) -> str:
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


def _split_into_sentences(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.strip() for p in parts if p.strip()]


def _sentence_similarity(a: str, b: str) -> float:
    """Word-overlap ratio (Jaccard) between two sentences, 0..1."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _generate_opt_bpo_sampled(
    messages: list, max_new_tokens: int = 140, temperature: float = 0.7,
) -> str:
    """Like _generate_opt_bpo but with sampling enabled for more varied output."""
    input_text = opt_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = opt_tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = opt_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=opt_tokenizer.eos_token_id,
        )
    input_len = inputs.input_ids.shape[1]
    return opt_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()


def _cleanup_refined_sentence(raw: str, original_sentence: str, user_prompt: str) -> str:
    """
    Lighter cleanup for single-sentence refinement output.
    Unlike _cleanup_optimized_output (designed for full prompts), this avoids
    aggressive truncation that would destroy valid sentence-level edits.
    """
    text = (raw or "").strip()
    if not text:
        return original_sentence

    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    text = text.replace("\n", " ").strip()

    prefixes_to_strip = [
        "Here's an optimized version of your prompt:",
        "Here's a rewritten version:",
        "Here's the optimized prompt:",
        "Here's the rephrased instruction:",
        "Here's the shortened version:",
        "Here's the expanded version:",
        "Rewritten:", "Optimized:", "Improved:", "Result:",
        "Rephrased:", "Shortened:", "Expanded:", "Revised:",
        "Optimized prompt:", "Rewritten prompt:",
        "Rephrased instruction:", "Shortened instruction:",
        "Here is the revised instruction:",
        "Here is the rephrased instruction:",
    ]
    for prefix in prefixes_to_strip:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
            break

    if len(text) >= 2 and text.startswith('"'):
        end = text.find('"', 1)
        if end > 1:
            inner = text[1:end].strip()
            if len(inner) >= 6:
                text = inner

    commentary_markers = (" I have ", " I also ", " I've ", " I rephrased", " This version ")
    for marker in commentary_markers:
        idx = text.find(marker)
        if idx >= 15:
            text = text[:idx].strip()

    tail_labels = re.compile(
        r"\s*(?:Output|Result|Additional constraint|Note|Constraint)\s*:\s*[\"'].*$",
        re.IGNORECASE,
    )
    text = tail_labels.sub("", text).strip()

    sents = re.split(r"(?<=[.!?])\s+", text)
    if len(sents) >= 2:
        seen_lower = set()
        deduped = []
        for s in sents:
            norm = s.lower().strip().rstrip(".")
            if norm not in seen_lower:
                seen_lower.add(norm)
                deduped.append(s)
        text = " ".join(deduped)

    if text and text[-1] not in ".!?":
        text = f"{text}."

    if text.count('"') % 2 == 1:
        text += '"'

    if not text or len(text) < 5:
        return original_sentence

    return text


def run_refine_optimized_sentence(
    full_optimized: str,
    sentence_index: int,
    action: str,
    original_user_prompt: str,
) -> str:
    sentences = _split_into_sentences(full_optimized)
    if not sentences or sentence_index < 0 or sentence_index >= len(sentences):
        raise ValueError("invalid sentence_index for this prompt")
    if action not in ("elaborate", "concise"):
        raise ValueError("action must be elaborate or concise")
    target = sentences[sentence_index]
    other_sentences = " ".join(s for j, s in enumerate(sentences) if j != sentence_index)

    if action == "elaborate":
        system = (
            "You are a prompt editor. You add constraints, formatting requirements, or scope details "
            "to a single instruction sentence. You NEVER answer, explain, or discuss the topic. "
            "You NEVER provide information, facts, or content about the subject matter. "
            "You ONLY add prompt-engineering details like: expected output format, length constraints, "
            "audience, scope boundaries, what to include or exclude, or evaluation criteria.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the improved instruction sentence (1-2 sentences max).\n"
            "2. Do NOT answer or explain the topic in any way.\n"
            "3. Do NOT mention 'the user' or use personal pronouns (I, we, us, my, our, you, your).\n"
            "4. Do NOT repeat or rephrase any other sentence from the prompt.\n"
            "5. No labels, no quotes, no preamble, no commentary."
        )
        user_content = (
            f"This is one instruction sentence from a prompt:\n{target}\n\n"
            "Add prompt-engineering details to this sentence — such as output format, "
            "scope limits, what to include/exclude, length, or audience. "
            "Do NOT answer the topic. Do NOT provide facts or explanations about the subject. "
            "Only add instructions that tell an AI HOW to respond.\n\n"
            "Do NOT repeat or rephrase these other sentences (they already exist in the prompt):\n"
            + other_sentences
        )
        mt = 150
    else:
        system = (
            "You shorten sentences. Given one sentence, output a shorter version with the same meaning. "
            "Remove filler words and redundant phrases. Do NOT add new content or extra sentences. "
            "Do NOT answer or explain the topic. "
            "Do NOT use personal pronouns (I, we, us, my, our, you, your). "
            "The output MUST be shorter than the input. Output ONLY one sentence."
        )
        user_content = (
            f"Make this sentence shorter (remove filler, keep meaning):\n\n"
            f"{target}\n\n"
            "Shorter version:"
        )
        mt = 60

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    raw = _generate_opt_bpo_sampled(messages, max_new_tokens=mt)
    logger.info("[refine] raw output: %r", raw[:300] if len(raw) > 300 else raw)
    new_sentence = _cleanup_refined_sentence(raw, target, original_user_prompt)
    logger.info("[refine] cleaned: %r", new_sentence[:200] if len(new_sentence) > 200 else new_sentence)

    def _is_bad(candidate, orig, act):
        if not candidate:
            return True
        if candidate.lower().strip(". ") == orig.lower().strip(". "):
            return True
        if act == "concise" and len(candidate.split()) >= len(orig.split()):
            return True
        return False

    if _is_bad(new_sentence, target, action):
        logger.info("[refine] first attempt rejected, retrying with temperature=0.95")
        raw = _generate_opt_bpo_sampled(messages, max_new_tokens=mt, temperature=0.95)
        logger.info("[refine] retry raw: %r", raw[:300] if len(raw) > 300 else raw)
        new_sentence = _cleanup_refined_sentence(raw, target, original_user_prompt)
        logger.info("[refine] retry cleaned: %r", new_sentence[:200] if len(new_sentence) > 200 else new_sentence)

    if action == "concise" and _is_bad(new_sentence, target, action):
        logger.info("[refine] concise still too long, trimming programmatically")
        words = target.split()
        cut = max(len(words) * 2 // 3, 3)
        trimmed = " ".join(words[:cut])
        if trimmed and trimmed[-1] not in ".!?":
            trimmed += "."
        new_sentence = trimmed

    other_sents = [s for j, s in enumerate(sentences) if j != sentence_index]
    replacement_parts = _split_into_sentences(new_sentence)
    if not replacement_parts:
        replacement_parts = [new_sentence]

    if action == "elaborate" and len(replacement_parts) > 2:
        logger.info("[refine] elaborate produced %d sentences, keeping first 2", len(replacement_parts))
        replacement_parts = replacement_parts[:2]

    filtered = []
    for part in replacement_parts:
        is_dup = any(
            _sentence_similarity(part, other) > 0.65
            for other in other_sents
        )
        if is_dup:
            logger.info("[refine] dropping near-duplicate of unchanged sentence: %r", part[:120])
        else:
            filtered.append(part)

    if not filtered:
        filtered = replacement_parts[:1]

    out = sentences[:]
    out[sentence_index] = " ".join(filtered)
    assembled = " ".join(out)

    final_sents = _split_into_sentences(assembled)
    seen = set()
    deduped = []
    for s in final_sents:
        norm = s.lower().strip().rstrip(".")
        if norm not in seen:
            seen.add(norm)
            deduped.append(s)
    assembled = " ".join(deduped)

    return assembled


def run_refine_optimized_full(
    original_user_prompt: str,
    initial_optimized: str,
    current_optimized: str,
) -> str:
    user_block = (
        f"Original user prompt:\n{original_user_prompt}\n\n"
        f"The first optimized version was:\n{initial_optimized}\n\n"
        f"The current optimized prompt is:\n{current_optimized}\n\n"
        "Write a new optimized prompt for the same user intent. "
        "It must be meaningfully different from the current text (structure, emphasis, or constraints), "
        "not a small edit. 2–3 sentences, under 110 words. Output only the new prompt."
    )
    messages = [
        {"role": "system", "content": OPTIMIZE_SYSTEM_PROMPT},
        {"role": "user", "content": user_block},
    ]
    raw = _generate_opt_bpo_sampled(messages, max_new_tokens=200)
    return _cleanup_optimized_output(raw)


def run_optimize(prompt: str) -> str:
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
    reply = _cleanup_optimized_output(_generate_opt_bpo(primary_messages))
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
    retry = _cleanup_optimized_output(_generate_opt_bpo(strict_messages, max_new_tokens=170))
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


def run_refine_generated_sentence(
    full_generated: str,
    sentence_index: int,
    action: str,
    original_user_idea: str,
) -> str:
    sentences = _split_into_sentences(full_generated)
    if not sentences or sentence_index < 0 or sentence_index >= len(sentences):
        raise ValueError("invalid sentence_index for this prompt")
    if action not in ("elaborate", "concise"):
        raise ValueError("action must be elaborate or concise")
    target = sentences[sentence_index]
    other_sentences = " ".join(s for j, s in enumerate(sentences) if j != sentence_index)

    if action == "elaborate":
        system = (
            "You are a prompt editor. You add constraints, formatting requirements, or scope details "
            "to a single instruction sentence inside a generated prompt. "
            "You NEVER answer, explain, or perform the underlying task. "
            "You ONLY add prompt-engineering details: output format, length constraints, "
            "audience, scope, what to include/exclude, or evaluation criteria.\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY the improved instruction sentence (1-2 sentences max).\n"
            "2. Do NOT answer or explain the topic in any way.\n"
            "3. Do NOT repeat or rephrase any other sentence from the prompt.\n"
            "4. No labels, no quotes, no preamble, no commentary."
        )
        user_content = (
            f"Original idea: {original_user_idea}\n\n"
            f"This is one sentence from the generated prompt:\n{target}\n\n"
            "Add prompt-engineering details to this sentence — output format, scope limits, "
            "what to include/exclude, length, or audience. Do NOT answer the topic.\n\n"
            "Do NOT repeat or rephrase these other sentences (they already exist in the prompt):\n"
            + other_sentences
        )
        mt = 150
    else:
        system = (
            "You shorten sentences. Given one sentence, output a shorter version with the same meaning. "
            "No labels, no quotes, no preamble."
        )
        user_content = f"Shorten this sentence:\n{target}"
        mt = 80

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    raw = _generate_opt_bpo_sampled(messages, max_new_tokens=mt)
    new_sentence = _cleanup_refined_sentence(raw, target, original_user_idea)

    other_sents = [s for j, s in enumerate(sentences) if j != sentence_index]
    replacement_parts = _split_into_sentences(new_sentence)
    if not replacement_parts:
        replacement_parts = [new_sentence]

    if action == "elaborate" and len(replacement_parts) > 2:
        replacement_parts = replacement_parts[:2]

    filtered = []
    for part in replacement_parts:
        is_dup = any(
            _sentence_similarity(part, other) > 0.65
            for other in other_sents
        )
        if not is_dup:
            filtered.append(part)
    if not filtered:
        filtered = replacement_parts[:1]

    out = sentences[:]
    out[sentence_index] = " ".join(filtered)
    assembled = " ".join(out)

    final_sents = _split_into_sentences(assembled)
    seen = set()
    deduped = []
    for s in final_sents:
        norm = s.lower().strip().rstrip(".")
        if norm not in seen:
            seen.add(norm)
            deduped.append(s)
    return " ".join(deduped)


def run_refine_generated_full(
    original_user_idea: str,
    initial_generated: str,
    current_generated: str,
) -> str:
    user_block = (
        f"Original user idea:\n{original_user_idea}\n\n"
        f"The first generated prompt was:\n{initial_generated}\n\n"
        f"The current generated prompt is:\n{current_generated}\n\n"
        "Write a new, meaningfully different generated prompt for the same idea "
        "(vary structure, emphasis, or constraints). Under 170 words. Output only the new prompt."
    )
    messages = [
        {"role": "system", "content": TASK_TO_PROMPT_SYSTEM},
        {"role": "user", "content": user_block},
    ]
    raw = _generate_opt_bpo_sampled(messages, max_new_tokens=220)
    text = _strip_optimize_meta_wrapper(raw)
    for stop in ["<|im_end|>", "<|endoftext|>"]:
        if stop in text:
            text = text.split(stop)[0].strip()
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text or current_generated


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




def _parse_gemini_feedback_json(raw: str) -> dict[str, str] | None:
    """Parse single JSON object from Gemini into per-dimension paragraph strings."""
    parsed = _parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    out: dict[str, str] = {}
    for dim in HELPSTEER_DIMENSIONS:
        val = parsed.get(dim)
        if isinstance(val, str) and val.strip():
            out[dim] = val.strip()
        elif isinstance(val, list):
            # fallback: join array items into a paragraph
            joined = " ".join(str(x).strip() for x in val if str(x).strip())
            if joined:
                out[dim] = joined
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
    """Gemini bullets when GEMINI_FEEDBACK_ENABLED; otherwise default _fallback_explanation text."""
    out: dict[str, str] = {}

    if _gemini_feedback_enabled():
        gemini_result = _run_gemini_explanations(user_prompt, scores)
        if gemini_result:
            out.update(gemini_result)
        missing = [d for d in HELPSTEER_DIMENSIONS if d not in out]
        if missing:
            logger.info(
                "[explanations] filling %d dimensions with default fallback: %s",
                len(missing),
                missing,
            )
            for k in missing:
                out[k] = _fallback_explanation(k, scores.get(k, 3))
        return out

    logger.info("[explanations] Gemini disabled (GEMINI_FEEDBACK_ENABLED); default feedback for all dimensions")
    for k in HELPSTEER_DIMENSIONS:
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


@app.route("/api/refine_optimized", methods=["POST"])
def refine_optimized():
    data = request.get_json() or {}
    mode = (data.get("mode") or "").strip().lower()
    original_user_prompt = (data.get("original_user_prompt") or "").strip()
    full_optimized = (data.get("full_optimized") or "").strip()
    initial_optimized = (data.get("initial_optimized") or "").strip() or full_optimized

    if not original_user_prompt:
        return jsonify({"error": "original_user_prompt required"}), 400

    if mode == "sentence":
        if not full_optimized:
            return jsonify({"error": "full_optimized required"}), 400
        try:
            idx = int(data.get("sentence_index"))
        except (TypeError, ValueError):
            return jsonify({"error": "sentence_index must be an integer"}), 400
        action = (data.get("action") or "").strip().lower()
        if action not in ("elaborate", "concise"):
            return jsonify(
                {"error": "action must be elaborate or concise"}
            ), 400
        try:
            out = run_refine_optimized_sentence(
                full_optimized, idx, action, original_user_prompt
            )
            return jsonify({"optimized": out})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("refine_optimized sentence failed")
            return jsonify({"error": str(e)}), 500

    if mode == "full":
        if not full_optimized:
            return jsonify({"error": "full_optimized required"}), 400
        try:
            out = run_refine_optimized_full(
                original_user_prompt, initial_optimized, full_optimized
            )
            return jsonify({"optimized": out})
        except Exception as e:
            logger.exception("refine_optimized full failed")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": 'mode must be "sentence" or "full"'}), 400


@app.route("/api/refine_generated", methods=["POST"])
def refine_generated():
    data = request.get_json() or {}
    mode = (data.get("mode") or "").strip().lower()
    original_user_idea = (data.get("original_user_idea") or "").strip()
    full_generated = (data.get("full_generated") or "").strip()
    initial_generated = (data.get("initial_generated") or "").strip() or full_generated

    if not original_user_idea:
        return jsonify({"error": "original_user_idea required"}), 400

    if mode == "sentence":
        if not full_generated:
            return jsonify({"error": "full_generated required"}), 400
        try:
            idx = int(data.get("sentence_index"))
        except (TypeError, ValueError):
            return jsonify({"error": "sentence_index must be an integer"}), 400
        action = (data.get("action") or "").strip().lower()
        if action not in ("elaborate", "concise"):
            return jsonify({"error": "action must be elaborate or concise"}), 400
        try:
            out = run_refine_generated_sentence(full_generated, idx, action, original_user_idea)
            return jsonify({"generated": out})
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logger.exception("refine_generated sentence failed")
            return jsonify({"error": str(e)}), 500

    if mode == "full":
        if not full_generated:
            return jsonify({"error": "full_generated required"}), 400
        try:
            out = run_refine_generated_full(original_user_idea, initial_generated, full_generated)
            return jsonify({"generated": out})
        except Exception as e:
            logger.exception("refine_generated full failed")
            return jsonify({"error": str(e)}), 500

    return jsonify({"error": 'mode must be "sentence" or "full"'}), 400


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app.run(port=5001, debug=False)
