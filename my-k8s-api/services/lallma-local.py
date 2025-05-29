# services/llm_local.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict

MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_4bit=True
)

def generate_response(messages: List[Dict[str, str]], max_tokens=512) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

