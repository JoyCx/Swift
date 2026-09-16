#!/usr/bin/env python3
"""Send an image to the ninfer vision server and print the model's reading.

The Swift-Qwen3.8-27B NVFP4 (abliterated) artifact already carries the full
vision tower + projector (333 `model.visual.*` tensors, sourced from the base
model by the converter). No separate mmproj file exists in ninfer's format —
vision is baked into the .ninfer. You only need to launch the server with
--vision (see VISION_README.md) and POST an OpenAI-style image_url message.

Usage:
    python vision_demo.py path/to/image.png ["your question"] [--think]

Defaults: thinking OFF (crisp answer). Pass --think to see the reasoning trace.
Targets http://127.0.0.1:8080 ; override with NINFER_URL env var.
"""
import base64, json, os, sys, time, mimetypes, urllib.request

URL = os.environ.get("NINFER_URL", "http://127.0.0.1:8080") + "/v1/chat/completions"
MODEL = os.environ.get("NINFER_MODEL", "qwen3.8-27b")

def main():
    args = [a for a in sys.argv[1:]]
    think = "--think" in args
    args = [a for a in args if a != "--think"]
    if not args:
        print("usage: python vision_demo.py <image> [\"prompt\"] [--think]"); return 1
    img = args[0]
    prompt = args[1] if len(args) > 1 else "Describe this image in detail. Read any text exactly."
    mime = mimetypes.guess_type(img)[0] or "image/png"
    b64 = base64.b64encode(open(img, "rb").read()).decode()

    payload = {
        "model": MODEL, "temperature": 0, "max_tokens": 1536,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
    }
    if not think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    wall = time.time() - t0

    if "error" in resp:
        print("ERROR:", resp["error"]); return 1
    ch = resp["choices"][0]; m = ch["message"]; t = resp.get("timings", {}); u = resp.get("usage", {})
    if think and m.get("reasoning_content"):
        print("=== reasoning ===\n" + m["reasoning_content"] + "\n")
    print("=== answer ===\n" + (m.get("content") or "(empty — raise max_tokens or drop --think)"))
    print(f"\n[finish={ch.get('finish_reason')} | decode={t.get('predicted_per_second',0):.1f} tok/s | "
          f"prompt_tokens={u.get('prompt_tokens')} (incl image) | wall={wall:.2f}s]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
