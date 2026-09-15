"""HuggingFace transformers backend: generation, residual-stream capture, logits.

Used for direction extraction, KL measurement and (via swiftlab.edit) writing edited
safetensors.  Everything here imports torch lazily so the rest of the harness runs
without it.
"""
from __future__ import annotations
import time
from typing import Any

import numpy as np

from .base import Backend, GenRequest, Generation


class HFBackend(Backend):
    name = "hf"

    def __init__(self, cfg, model=None, tokenizer=None, **kw):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.cfg = cfg
        self.torch = torch
        self.tok = tokenizer or AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
        dtype = getattr(torch, cfg.dtype)
        load_kw: dict[str, Any] = {"torch_dtype": dtype, "device_map": cfg.device_map, "trust_remote_code": True}
        # 32 GB card + a 27B won't fit in BF16. Two ways to still capture activations for the edit:
        #   load_in_4bit: bitsandbytes 4-bit base (~16 GB VRAM); activations stay fp16, capture is exact enough.
        #   otherwise device_map="auto" spills layers to system RAM (needs ~54 GB free RAM), slower but no bnb.
        # The edit itself (swiftlab edit) never loads the model — it streams safetensors on CPU.
        if cfg.extra.get("load_in_4bit"):
            from transformers import BitsAndBytesConfig
            load_kw.pop("torch_dtype")
            load_kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                                                                bnb_4bit_quant_type=cfg.extra.get("bnb_quant_type", "nf4"),
                                                                bnb_4bit_use_double_quant=True)
        max_mem = cfg.extra.get("max_memory")            # e.g. {"0": "30GiB", "cpu": "120GiB"} for CPU offload
        if max_mem:
            load_kw["max_memory"] = max_mem
        self.model = model or AutoModelForCausalLM.from_pretrained(cfg.model, **load_kw)
        self.model.eval()
        self._layers = _find_layers(self.model)

    def n_layers(self) -> int:
        return len(self._layers)

    def count_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def render(self, req: GenRequest, assistant_prefix: str | None = None, enable_thinking: bool = True) -> str:
        msgs = ([{"role": "system", "content": req.context}] if req.context else []) + [{"role": "user", "content": req.prompt}]
        kw: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": enable_thinking,
                              "reasoning_effort": req.reasoning_effort or self.cfg.reasoning_effort}
        try:
            text = self.tok.apply_chat_template(msgs, **kw)
        except TypeError:
            kw.pop("reasoning_effort"); text = self.tok.apply_chat_template(msgs, **kw)
        if assistant_prefix is not None:
            text += assistant_prefix
        return text

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _generate_text(self, text: str, seed: int, max_tokens: int, temperature: float, logit_bias=None) -> tuple[str, int, str]:
        torch = self.torch
        ids = self.tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        processors = None
        if logit_bias:
            from transformers import LogitsProcessorList
            processors = LogitsProcessorList([_BiasProcessor(logit_bias, self.device)])
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                      top_p=self.cfg.top_p, logits_processor=processors, pad_token_id=self.tok.eos_token_id)
        new = out[0, ids["input_ids"].shape[1]:]
        finish = "stop" if (new[-1].item() in set(getattr(self.model.generation_config, "eos_token_id", []) or [self.tok.eos_token_id])) else "length"
        return self.tok.decode(new, skip_special_tokens=False), int(new.shape[0]), finish

    def generate(self, req: GenRequest) -> Generation:
        from ..trace import split_thinking
        t0 = time.perf_counter()
        text, n, finish = self._generate_text(self.render(req), req.seed, req.max_tokens or self.cfg.max_tokens,
                                              self.cfg.temperature if req.temperature is None else req.temperature, req.logit_bias)
        lat = time.perf_counter() - t0
        thinking, answer = split_thinking(text, self.cfg.think_open, self.cfg.think_close)
        tt = self.count_tokens(thinking)
        return Generation(req.task_id, req.seed, thinking, _strip_special(answer, self.tok), tt, max(0, n - tt), lat, finish_reason=finish, model=self.cfg.model)

    def complete_prefix(self, req: GenRequest, think_prefix: str, max_tokens: int = 256) -> str:
        prefix = f"{self.cfg.think_open}\n{think_prefix.rstrip()}\n{self.cfg.think_close}\n\n"
        rendered = self.render(req)
        # chat templates usually already emit the think opener; avoid doubling it
        if rendered.rstrip().endswith(self.cfg.think_open):
            rendered = rendered.rstrip()[: -len(self.cfg.think_open)]
        text, _, _ = self._generate_text(rendered + prefix, req.seed, max_tokens, 0.0)
        return _strip_special(text, self.tok)

    def capture(self, texts: list[str], spans: list[tuple[int, int]], layers: list[int]) -> dict[int, np.ndarray]:
        torch = self.torch
        store: dict[int, list] = {l: [] for l in layers}
        hooks = []
        cur: dict[int, Any] = {}

        def mk(l):
            def hook(_m, _i, out):
                cur[l] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook

        for l in layers:
            hooks.append(self._layers[l].register_forward_hook(mk(l)))
        try:
            for text, (a, b) in zip(texts, spans):
                enc = self.tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
                offs = enc.pop("offset_mapping")[0].tolist()
                tok_idx = [i for i, (s, e) in enumerate(offs) if e > a and s < b]
                if not tok_idx:
                    tok_idx = [len(offs) - 1]
                with torch.no_grad():
                    self.model(**enc.to(self.device))
                for l in layers:
                    h = cur[l][0, tok_idx, :].float().mean(0).cpu().numpy()
                    store[l].append(h)
        finally:
            for h in hooks:
                h.remove()
        return {l: np.stack(v) for l, v in store.items()}

    def next_token_dist(self, prompts: list[str]) -> np.ndarray:
        torch = self.torch
        rows = []
        for p in prompts:
            enc = self.tok(p, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[0, -1].float()
            rows.append(torch.softmax(logits, -1).cpu().numpy())
        return np.stack(rows)


class _BiasProcessor:
    def __init__(self, bias: dict[int, float], device):
        import torch
        self.ids = torch.tensor(list(bias.keys()), device=device)
        self.vals = torch.tensor(list(bias.values()), device=device)

    def __call__(self, input_ids, scores):
        scores[:, self.ids] += self.vals.to(scores.dtype)
        return scores


def _find_layers(model):
    for attr in ("model.layers", "model.language_model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for p in attr.split("."):
                obj = getattr(obj, p)
            return list(obj)
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layers; set them manually")


def _strip_special(text: str, tok) -> str:
    for t in (tok.eos_token, getattr(tok, "pad_token", None), "<|im_end|>", "<|endoftext|>"):
        if t:
            text = text.replace(t, "")
    return text.strip()
