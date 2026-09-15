#!/usr/bin/env python
"""Merge a LoRA adapter into BF16 safetensors:  python scripts/merge_lora.py --base <dir> --lora <dir> --out <dir>"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--lora", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
m = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16, device_map="cpu")
m = PeftModel.from_pretrained(m, a.lora).merge_and_unload()
m.save_pretrained(a.out, safe_serialization=True); AutoTokenizer.from_pretrained(a.base).save_pretrained(a.out)
print("merged ->", a.out)
