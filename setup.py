import requests

API_URL = "https://scroll-library-squiggle.ngrok-free.dev/generate"

def load_model():
    print("Using Kaggle GPU — fine-tuned model!")

def generate_response(prompt, max_tokens=300):
    try:
        response = requests.post(
            API_URL,
            json={"prompt": prompt},
            timeout=120,
        )
        return response.json()["generated"]
    except Exception as e:
        return f"Error: {e}"
'''from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    base = "unsloth/Qwen2.5-Math-1.5B-Instruct"
    adapter = "SHIHICOOKIE02/qwen-math-rl-grpo"
    
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.pad_token = tokenizer.eos_token
    
    m = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(m, adapter)
    model = model.merge_and_unload()
    print("Fine-tuned model loaded!")

def generate_response(prompt, max_tokens=300):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)'''






'''from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def load_model_and_tokenizer(model_name="gpt2"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device} | Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.to(device)
    print(f"Model loaded! Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=100):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True)'''

'''from groq import Groq
import re

import os
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def generate_response(prompt, max_tokens=300):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content

def load_model_and_tokenizer(model_name=None):
    print("Using Groq API — no model loading needed!")
    return None, None'''