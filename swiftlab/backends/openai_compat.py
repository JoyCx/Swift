"""OpenAI-compatible backend (vLLM, llama-server, SGLang, ukisai API).

Seeds, temperature and reasoning effort are sent per request so that both arms of a
paired evaluation see identical decoding settings.
"""
from __future__ import annotations
import json, time, urllib.request, urllib.error
from typing import Any

import numpy as np

from .base import Backend, GenRequest, Generation
from ..util import approx_tokens


class OpenAICompatBackend(Backend):
    name = "openai"

    def __init__(self, cfg, **kw):
        self.cfg = cfg
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg.api_key}"}
        self.timeout = float(cfg.extra.get("timeout", 3600))

    # ------------------------------------------------------------------ http
    def _post(self, path: str, body: dict, retries: int = 4) -> dict:
        data = json.dumps(body).encode("utf-8")
        last = None
        for i in range(retries):
            try:
                r = urllib.request.Request(self.base_url + path, data=data, headers=self.headers, method="POST")
                with urllib.request.urlopen(r, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:  # noqa: PERF203
                last = e
                time.sleep(2 ** i)
        raise RuntimeError(f"request failed after {retries} tries: {last}")

    def _messages(self, req: GenRequest) -> list[dict]:
        msgs = []
        if req.context:
            msgs.append({"role": "system", "content": req.context})
        msgs.append({"role": "user", "content": req.prompt})
        return msgs

    def _body(self, req: GenRequest) -> dict:
        effort = req.reasoning_effort or self.cfg.reasoning_effort
        body: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(req),
            "seed": req.seed,
            "temperature": self.cfg.temperature if req.temperature is None else req.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": req.max_tokens or self.cfg.max_tokens,
            "chat_template_kwargs": {"reasoning_effort": effort, "enable_thinking": True},
        }
        if self.cfg.extra.get("send_reasoning_effort_top_level", True):
            body["reasoning_effort"] = effort
        if req.logit_bias:
            body["logit_bias"] = {str(k): float(v) for k, v in req.logit_bias.items()}
        body.update(self.cfg.extra.get("request_extra", {}))
        return body

    # ------------------------------------------------------------------ api
    def generate(self, req: GenRequest) -> Generation:
        t0 = time.perf_counter()
        resp = self._post("/chat/completions", self._body(req))
        lat = time.perf_counter() - t0
        ch = resp["choices"][0]
        msg = ch["message"]
        content = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        if not thinking and self.cfg.think_open in content:
            from ..trace import split_thinking
            thinking, content = split_thinking(content, self.cfg.think_open, self.cfg.think_close)
        usage = resp.get("usage", {}) or {}
        det = usage.get("completion_tokens_details") or {}
        think_tok = det.get("reasoning_tokens")
        comp = usage.get("completion_tokens")
        if think_tok is None:
            think_tok = approx_tokens(thinking) if comp is None else int(round(comp * len(thinking) / max(1, len(thinking) + len(content))))
        ans_tok = (comp - think_tok) if comp is not None else approx_tokens(content)
        return Generation(req.task_id, req.seed, thinking, content, int(think_tok), int(max(0, ans_tok)), lat,
                          finish_reason=ch.get("finish_reason", "stop"), model=self.model,
                          extra={"usage": usage, "effort": req.reasoning_effort or self.cfg.reasoning_effort})

    def complete_prefix(self, req: GenRequest, think_prefix: str, max_tokens: int = 256) -> str:
        """Uses vLLM's continue_final_message; falls back to a raw /completions prompt template."""
        assistant = f"{self.cfg.think_open}\n{think_prefix.rstrip()}\n{self.cfg.think_close}\n\n"
        body = self._body(req)
        body["messages"] = self._messages(req) + [{"role": "assistant", "content": assistant}]
        body.update({"max_tokens": max_tokens, "add_generation_prompt": False, "continue_final_message": True})
        body["chat_template_kwargs"]["enable_thinking"] = False
        try:
            resp = self._post("/chat/completions", body, retries=2)
            return resp["choices"][0]["message"].get("content") or ""
        except RuntimeError:
            tpl = self.cfg.extra.get("prompt_template") or (
                "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}")
            prompt = tpl.format(system=req.context or "You are a helpful assistant.", user=req.prompt, assistant=assistant)
            resp = self._post("/completions", {"model": self.model, "prompt": prompt, "max_tokens": max_tokens,
                                                "temperature": 0.0, "seed": req.seed})
            return resp["choices"][0].get("text") or ""

    def next_token_dist(self, prompts: list[str]) -> np.ndarray:
        """Top-k logprob rows via /completions (vLLM). Missing mass is lumped into one bucket."""
        rows = []
        k = int(self.cfg.extra.get("kl_topk", 20))
        for p in prompts:
            resp = self._post("/completions", {"model": self.model, "prompt": p, "max_tokens": 1, "temperature": 0.0, "logprobs": k})
            lp = resp["choices"][0]["logprobs"]["top_logprobs"][0]
            rows.append(dict(sorted(lp.items())))
        return _align_rows(rows)

    def count_tokens(self, text: str) -> int:
        try:
            resp = self._post("/tokenize", {"model": self.model, "prompt": text}, retries=1)
            return int(resp.get("count") or len(resp.get("tokens", [])))
        except Exception:
            return approx_tokens(text)


def _align_rows(rows: list[dict[str, float]]) -> np.ndarray:
    vocab = sorted({t for r in rows for t in r})
    idx = {t: i for i, t in enumerate(vocab)}
    out = np.zeros((len(rows), len(vocab) + 1))
    for i, r in enumerate(rows):
        for t, lp in r.items():
            out[i, idx[t]] = np.exp(lp)
        out[i, -1] = max(0.0, 1.0 - out[i, :-1].sum())
    return out
