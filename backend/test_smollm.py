"""
Test SmolLM2-360M-Instruct + BPO LoRA adapter for prompt optimization.
Runs on M1 Mac (MPS) or CPU via transformers + peft (no Unsloth).
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "models/smollm_bpo_lora"

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32
print(f"Using device: {device}")

print(f"Loading base model {BASE_MODEL_ID}...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=dtype,
)

print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

def optimize_prompt(prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You rewrite user prompts to be clearer and more specific "
                "so an AI will give better answers. Output only the rewritten prompt, nothing else."
                "Limit the response to 64 tokens or less."
            ),
        },
        {"role": "user", "content": f"Optimize this prompt: {prompt}"},
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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
    generated_ids = outputs[0][input_len:]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    for stop in ["<|im_end|>", "<|endoftext|>", "\n\n"]:
        if stop in reply:
            reply = reply.split(stop)[0].strip()

    return reply


test_prompts = [
    "Explain gravity.",
    "Write a story about a robot.",
    "Help me write an email to my boss about a raise.",
    "How do I make a website?",
]

for prompt in test_prompts:
    result = optimize_prompt(prompt)
    print(f"  Original : {prompt}")
    print(f"  Optimized: {result}")
    print()
