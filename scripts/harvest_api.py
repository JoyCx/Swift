"""Rate-limited, resumable reasoning-trace harvester for OpenAI-compatible endpoints.

Built for the free UkisAI Swift research API (5 requests/minute, Cloudflare in front), but works
against any OpenAI-compatible server (llama-server, vLLM, OpenRouter).

    python scripts/harvest_api.py --prompts data/harvest/prompts.jsonl --out runs/harvest/swift_api.jsonl

Input JSONL rows: {"id": str, "domain": str, "messages": [...]} or {"id", "domain", "prompt": str};
any extra keys (e.g. "verify", "meta") are copied to the output row untouched.

Behaviour that matters for a shared free endpoint:
* request *starts* are spaced 60/rpm seconds apart, so the limit holds regardless of how long
  generations take; in-flight requests are capped separately;
* every attempt, including failed ones, consumes quota upstream, so an outage (5xx / HTML error
  page / connection error) puts the harvester into exponential backoff with a single probe at a
  time instead of burning the budget;
* 429 honours Retry-After;
* responses are streamed (Cloudflare drops idle non-streaming connections after ~100 s);
* the output file is append-only; completed (id, sample) pairs are skipped on restart;
* create the file `<out>.stop` to finish in-flight requests and exit cleanly.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx


@dataclass
class Limiter:
    rpm: float
    max_inflight: int
    next_start: float = 0.0
    backoff_until: float = 0.0
    consecutive_failures: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        self.sem = asyncio.Semaphore(self.max_inflight)

    async def acquire_start(self) -> None:
        while True:
            async with self.lock:
                now = time.monotonic()
                wait = max(self.next_start, self.backoff_until) - now
                if wait <= 0:
                    self.next_start = now + 60.0 / self.rpm
                    return
            await asyncio.sleep(min(wait, 5.0))

    def outage(self) -> bool:
        return self.consecutive_failures > 0

    def fail(self, retry_after: float | None = None) -> float:
        self.consecutive_failures += 1
        delay = retry_after if retry_after is not None else min(900.0, 30.0 * 2 ** (self.consecutive_failures - 1))
        delay *= random.uniform(1.0, 1.2)
        self.backoff_until = max(self.backoff_until, time.monotonic() + delay)
        return delay

    def ok(self) -> None:
        self.consecutive_failures = 0


class TransientError(Exception):
    """outage=True: the server never started answering (5xx, error page, refused connection).
    Those retry forever under backoff; only failures after a response started count as attempts."""

    def __init__(self, msg: str, retry_after: float | None = None, outage: bool = False):
        super().__init__(msg)
        self.retry_after = retry_after
        self.outage = outage


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, file=sys.stderr, flush=True)


def load_done(out: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    done.add((r["id"], r["sample"]))
    return done


def build_body(args: argparse.Namespace, row: dict, sample: int) -> dict:
    messages = row.get("messages") or [{"role": "user", "content": row["prompt"]}]
    body: dict = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": row.get("max_tokens", args.max_tokens),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed_base + sample,
    }
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
        if not args.no_template_kwargs:
            body["chat_template_kwargs"] = {"reasoning_effort": args.reasoning_effort}
    if args.extra:
        body.update(json.loads(args.extra))
    return body


async def run_one(client: httpx.AsyncClient, args: argparse.Namespace, row: dict, sample: int) -> dict:
    body = build_body(args, row, sample)
    t0 = time.time()
    ttft = None
    reasoning: list[str] = []
    content: list[str] = []
    finish = None
    usage = None
    served_model = None
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    try:
        async with client.stream("POST", args.base_url.rstrip("/") + "/chat/completions", json=body, headers=headers) as resp:
            if resp.status_code == 429:
                ra = resp.headers.get("retry-after")
                raise TransientError("429", float(ra) if ra and ra.replace(".", "").isdigit() else 60.0, outage=True)
            if resp.status_code >= 500:
                raise TransientError(f"http {resp.status_code}", outage=True)
            if resp.status_code >= 400:
                text = (await resp.aread()).decode("utf-8", "replace")[:500]
                return {"error": f"http {resp.status_code}: {text}", "fatal": True}
            if "text/html" in resp.headers.get("content-type", ""):
                raise TransientError("html error page", outage=True)
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise TransientError(f"stream error: {str(chunk['error'])[:300]}")
                served_model = chunk.get("model", served_model)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    delta = ch.get("delta") or {}
                    r = delta.get("reasoning_content") or delta.get("reasoning")
                    c = delta.get("content")
                    if (r or c) and ttft is None:
                        ttft = time.time() - t0
                    if r:
                        reasoning.append(r)
                    if c:
                        content.append(c)
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
    except (httpx.ConnectError, httpx.ConnectTimeout) as e:
        raise TransientError(f"{type(e).__name__}: {e}", outage=True) from e
    except (httpx.TransportError, json.JSONDecodeError) as e:
        raise TransientError(f"{type(e).__name__}: {e}") from e
    if finish is None and not reasoning and not content:
        raise TransientError("empty stream")
    out = {k: v for k, v in row.items() if k not in ("messages", "prompt")}
    out.update(
        sample=sample,
        request={k: v for k, v in body.items() if k not in ("messages", "stream", "stream_options")},
        messages=body["messages"],
        reasoning="".join(reasoning),
        content="".join(content),
        finish_reason=finish,
        truncated=finish == "length",
        usage=usage,
        served_model=served_model,
        endpoint=args.base_url,
        t_start=t0,
        ttft_s=ttft,
        duration_s=time.time() - t0,
    )
    return out


async def worker(queue: asyncio.Queue, client, args, lim: Limiter, out_f, stats: dict, stop: Path) -> None:
    while True:
        try:
            row, sample, attempt = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if stop.exists():
            return
        async with lim.sem:
            # During an outage only one request probes at a time.
            while lim.outage() and stats["inflight"] > 0:
                await asyncio.sleep(2.0)
            await lim.acquire_start()
            if stop.exists():
                return
            stats["inflight"] += 1
            try:
                res = await run_one(client, args, row, sample)
            except TransientError as e:
                delay = lim.fail(e.retry_after)
                stats["transient"] += 1
                log(f"transient {row['id']}#{sample} attempt {attempt}: {e} -> backoff {delay:.0f}s")
                if e.outage:
                    queue.put_nowait((row, sample, attempt))
                elif attempt + 1 < args.max_attempts:
                    queue.put_nowait((row, sample, attempt + 1))
                else:
                    out_f.write(json.dumps({"id": row["id"], "sample": sample, "error": str(e)}) + "\n")
                    out_f.flush()
                continue
            finally:
                stats["inflight"] -= 1
        if res.get("fatal"):
            stats["fatal"] += 1
            log(f"FATAL {row['id']}#{sample}: {res['error']}")
            out_f.write(json.dumps({"id": row["id"], "sample": sample, "error": res["error"]}) + "\n")
            out_f.flush()
            if stats["fatal"] >= 3 and stats["ok"] == 0:
                log("3 fatal request errors before any success: request shape is wrong, stopping")
                stop.touch()
            continue
        lim.ok()
        stats["ok"] += 1
        stats["tokens"] += (res.get("usage") or {}).get("completion_tokens") or 0
        out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
        out_f.flush()
        u = res.get("usage") or {}
        log(
            f"ok {stats['ok']}/{stats['total']} {row['id']}#{sample} finish={res['finish_reason']} "
            f"think_chars={len(res['reasoning'])} completion_tokens={u.get('completion_tokens')} "
            f"{res['duration_s']:.0f}s"
        )


async def main_async(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stop = Path(str(out) + ".stop")
    if stop.exists():
        stop.unlink()
    done = load_done(out)
    rows = [json.loads(l) for l in Path(args.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    queue: asyncio.Queue = asyncio.Queue()
    for row in rows:
        for s in range(args.samples):
            if (row["id"], s) not in done:
                queue.put_nowait((row, s, 0))
    stats = {"ok": 0, "transient": 0, "fatal": 0, "inflight": 0, "tokens": 0, "total": queue.qsize()}
    log(f"{len(done)} already done, {queue.qsize()} queued, rpm={args.rpm} inflight<={args.max_inflight}")
    lim = Limiter(args.rpm, args.max_inflight)
    timeout = httpx.Timeout(connect=30.0, read=args.read_timeout, write=30.0, pool=None)
    async with httpx.AsyncClient(timeout=timeout, http2=False) as client:
        with out.open("a", encoding="utf-8") as out_f:
            await asyncio.gather(
                *(worker(queue, client, args, lim, out_f, stats, stop) for _ in range(args.max_inflight))
            )
    log(f"finished: {stats}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompts", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-url", default="https://ukisai.com/api/swift/v1")
    p.add_argument("--model", default="swift")
    p.add_argument("--api-key", default=os.environ.get("HARVEST_API_KEY", ""))
    p.add_argument("--rpm", type=float, default=4.5, help="request starts per minute (stay under the 5 RPM cap)")
    p.add_argument("--max-inflight", type=int, default=4)
    p.add_argument("--samples", type=int, default=1, help="samples per prompt (seed = seed_base + sample)")
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--reasoning-effort", default="xhigh")
    p.add_argument("--no-template-kwargs", action="store_true",
                   help="send reasoning_effort only as a top-level field (NInfer rejects chat_template_kwargs)")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--extra", default="", help="JSON merged into the request body, e.g. '{\"top_k\": 20}'")
    p.add_argument("--read-timeout", type=float, default=600.0, help="max silence between stream chunks")
    p.add_argument("--max-attempts", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
