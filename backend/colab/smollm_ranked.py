# -*- coding: utf-8 -*-
"""Colab / local trainer: multi-dimension response quality (HelpSteer → JSON scores).

Uses **nvidia/HelpSteer** (CC-BY-4.0, ~35k train rows): human ratings per *assistant
response* on five attributes (0–4), which we map to **1–5** for a consistent UI scale.

Dimensions (JSON keys):
  helpfulness, correctness, coherence, complexity, verbosity

Rough analogies if you previously used “prompt rubrics”:
  coherence ≈ clarity of the answer; verbosity relates to concision; helpfulness ≈
  overall utility of the reply. Labels are **about the response**, not the prompt alone.

**Why this fixes “all 4s”:** HelpSteer’s joint distribution of five attributes is diverse;
we also optionally **balance** by helpfulness bin so training isn’t dominated by one score.

**Next step (your plan):** keep a **separate** SmolLM (base or instruct) call that takes the
JSON scores and writes **natural-language explanations** per dimension for the frontend —
no explanation targets in HelpSteer, so that stage is instruction-following, not SFT here.

**Model:** default `unsloth/SmolLM2-360M-Instruct`. Set env `SMOLLM_MODEL=1.7b` for
`unsloth/SmolLM2-1.7B-Instruct` on a larger GPU (may be tight on free T4 in 4-bit).

Env:
  SMOLLM_MAX_STEPS=200
  SMOLLM_MODEL=360m|1.7b
  SMOLLM_BALANCE=1|0
  SMOLLM_MAX_PER_HELPFULNESS_BIN=2500
"""

!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install peft accelerate bitsandbytes

import json
import os
import random
import shutil
import warnings
from collections import defaultdict

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"transformers\.modeling_attn_mask_utils",
)

import torch
from datasets import Dataset, load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = int(os.environ.get("SMOLLM_MAX_SEQ_LEN", "2048"))
MAX_STEPS = int(os.environ.get("SMOLLM_MAX_STEPS", "200"))
MAX_RESPONSE_CHARS = int(os.environ.get("SMOLLM_MAX_RESPONSE_CHARS", "1800"))
SEED = int(os.environ.get("SMOLLM_SEED", "3407"))
OUTPUT_DIR = os.environ.get("SMOLLM_OUTPUT_DIR", "smollm_helpsteer_rater_lora")
BALANCE_BY_HELPFULNESS = os.environ.get("SMOLLM_BALANCE", "1").strip() == "1"
MAX_PER_HELPFULNESS_BIN = int(os.environ.get("SMOLLM_MAX_PER_HELPFULNESS_BIN", "2500"))

MODEL_PRESETS = {
    "360m": "unsloth/SmolLM2-360M-Instruct",
    "1.7b": "unsloth/SmolLM2-1.7B-Instruct",
}
_model_key = os.environ.get("SMOLLM_MODEL", "360m").strip().lower()
if _model_key not in MODEL_PRESETS:
    raise ValueError(f"SMOLLM_MODEL must be one of {list(MODEL_PRESETS)}; got {_model_key!r}")
MODEL_NAME = MODEL_PRESETS[_model_key]

# HelpSteer native names (order fixed for stable JSON).
DIMENSION_KEYS = (
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
    "helpfulness = how well the response addresses the user’s need; "
    "correctness = factual soundness; coherence = clarity and logical flow; "
    "complexity = intellectual depth appropriate to the task; "
    "verbosity = appropriate detail without excess padding."
)


def _to_1_5(x) -> int:
    v = int(x)
    return max(1, min(5, v + 1))  # HelpSteer is 0..4


def balance_by_helpfulness(ds, max_per_bin: int, seed: int):
    bins = defaultdict(list)
    for i in range(len(ds)):
        h = ds[i]["helpfulness"]
        if h is None:
            continue
        bins[int(h)].append(i)
    rng = random.Random(seed)
    picked = []
    for h in range(0, 5):
        idxs = bins.get(h, [])
        rng.shuffle(idxs)
        picked.extend(idxs[:max_per_bin])
    rng.shuffle(picked)
    return ds.select(picked)


def build_training_example(prompt: str, response: str, scores_15: dict) -> str:
    """Single ChatML example: model learns to emit the JSON target."""
    r = (response or "")[:MAX_RESPONSE_CHARS]
    target = json.dumps(scores_15, separators=(",", ":"), sort_keys=True)
    return (
        "<|im_start|>system\n"
        f"{RATER_SYSTEM}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Rate the assistant response for the user prompt below.\n\n"
        f"<user_prompt>\n{prompt}\n</user_prompt>\n\n"
        f"<assistant_response>\n{r}\n</assistant_response>\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{target}<|im_end|>"
    )


def formatting_batch(examples):
    """Batched map must return one text per row (use '' for skips, then filter)."""
    texts = []
    n = len(examples["prompt"])
    for i in range(n):
        p = examples["prompt"][i]
        r = examples["response"][i]
        if p is None or r is None:
            texts.append("")
            continue
        try:
            scores = {k: _to_1_5(examples[k][i]) for k in DIMENSION_KEYS}
        except (TypeError, KeyError, ValueError):
            texts.append("")
            continue
        texts.append(build_training_example(str(p), str(r), scores))
    return {"text": texts}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
raw = load_dataset("nvidia/HelpSteer", split="train")
if BALANCE_BY_HELPFULNESS:
    raw = balance_by_helpfulness(raw, MAX_PER_HELPFULNESS_BIN, SEED)
    print(
        f"Balanced HelpSteer train: up to {MAX_PER_HELPFULNESS_BIN} rows per helpfulness 0–4 "
        f"→ {len(raw)} rows"
    )
else:
    print(f"HelpSteer train (full): {len(raw)} rows")

mapped = raw.map(
    formatting_batch,
    batched=True,
    batch_size=256,
    remove_columns=raw.column_names,
    desc="format HelpSteer",
)
mapped = mapped.filter(lambda x: len(x["text"]) > 50)
full = mapped.shuffle(seed=SEED)
split = full.train_test_split(test_size=0.02, seed=SEED)
train_ds, eval_ds = split["train"], split["test"]
print(f"Supervision sequences: train={len(train_ds)} eval={len(eval_ds)}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=SEED,
)

eval_every = max(25, min(MAX_STEPS, MAX_STEPS // 4))
warmup = min(25, max(5, MAX_STEPS // 10))

ta_common = dict(
    output_dir="smollm_helpsteer_outputs",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=warmup,
    max_steps=MAX_STEPS,
    learning_rate=1e-4,
    logging_steps=max(1, MAX_STEPS // 20),
    eval_steps=eval_every,
    save_steps=eval_every,
    save_total_limit=2,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_strategy="steps",
)

try:
    training_args = TrainingArguments(**ta_common, evaluation_strategy="steps")
except TypeError:
    training_args = TrainingArguments(**ta_common, eval_strategy="steps")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
    args=training_args,
)

trainer.train()
FastLanguageModel.for_inference(model)


def safe_generate_from_inputs(inputs, max_new_tokens=128):
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.max_length = None
    gen_kw = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    try:
        return model.generate(**gen_kw)
    except RuntimeError as err:
        print(f"[safe_generate] fast path failed ({err}); trying _old_generate…")
        if hasattr(model, "_old_generate"):
            return model._old_generate(**gen_kw)
        raise


def run_json_rating(user_prompt: str, assistant_reply: str) -> str:
    r = (assistant_reply or "")[:MAX_RESPONSE_CHARS]
    user_block = (
        "Rate the assistant response for the user prompt below.\n\n"
        f"<user_prompt>\n{user_prompt}\n</user_prompt>\n\n"
        f"<assistant_response>\n{r}\n</assistant_response>\n"
    )
    messages = [
        {"role": "system", "content": RATER_SYSTEM},
        {"role": "user", "content": user_block},
    ]
    prompt_txt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt_txt, return_tensors="pt").to(device)
    out = safe_generate_from_inputs(inputs, max_new_tokens=160)
    ln = inputs["input_ids"].shape[1]
    return tokenizer.decode(out[0][ln:], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Smoke tests (diverse synthetic / held-out style pairs — not training labels)
# ---------------------------------------------------------------------------
print("\n=== JSON rater smoke (trained task) ===\n")
_SMOKE = [
    (
        "hi",
        "Hello! How can I help you today?",
    ),
    (
        "Explain quantum entanglement in two sentences for a teenager.",
        "Sure! Entanglement is when two particles stay linked so measuring one instantly affects the other, "
        "even far apart. It's spooky in plain language but tested in labs.",
    ),
    (
        "List three causes of the French Revolution with dates.",
        "Revolutions happen. Stuff was bad. People got mad.",
    ),
    (
        "Write a Python function that returns the factorial of n with type hints and a one-line docstring.",
        "def factorial(n: int) -> int:\n    '''Return n! for n >= 0.'''\n    if n < 0:\n        raise ValueError\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r",
    ),
]

for up, ar in _SMOKE:
    raw_out = run_json_rating(up, ar)
    print(f"user: {up[:80]!r}")
    print(f"raw:  {raw_out[:300]!r}{'...' if len(raw_out) > 300 else ''}\n")

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

_archive = shutil.make_archive(OUTPUT_DIR, "zip", ".", OUTPUT_DIR)
print(f"Saved LoRA → ./{OUTPUT_DIR}/ and {_archive}")
try:
    from google.colab import files as colab_files

    colab_files.download(os.path.basename(_archive))
except ImportError:
    print("(Not on Colab: skip files.download.)")
