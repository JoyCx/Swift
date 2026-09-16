"""Fetch canonical generation_config.json and build a no-mutation staging dir.

Staging lives on A: (same volume as the shards) so the shards can be hardlinked
rather than copied. The user's release dir is never written to.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
from pathlib import Path

RELEASE = Path("A:/models/Qwen3.8-27B")
STAGING = Path("A:/models/qwen38-staging-official")
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
GEN_URL = (
    f"https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{REVISION}/generation_config.json"
)
GEN_EXPECTED_SHA = "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"

# Everything the converter reads from the official --model dir, except
# generation_config.json which we replace with the canonical upstream bytes.
HARDLINK_NAMES = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_gen_config() -> bytes:
    req = urllib.request.Request(GEN_URL, headers={"User-Agent": "ninfer-convert/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    got = sha256(data)
    if got != GEN_EXPECTED_SHA:
        raise SystemExit(
            f"REFUSED: fetched generation_config.json hash {got} "
            f"!= expected {GEN_EXPECTED_SHA}; not trusting it"
        )
    print(f"fetched generation_config.json OK ({len(data)} bytes, sha256 verified)")
    return data


def build_staging(gen_bytes: bytes) -> None:
    if STAGING.exists():
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    # Hardlink the untouched-and-verified files (incl. the 18 shards).
    shards = sorted(RELEASE.glob("model-*-of-*.safetensors"))
    if len(shards) != 18:
        raise SystemExit(f"expected 18 shards, found {len(shards)}")
    for name in HARDLINK_NAMES:
        src = RELEASE / name
        if not src.is_file():
            raise SystemExit(f"missing expected official file: {src}")
        os.link(src, STAGING / name)
    for shard in shards:
        os.link(shard, STAGING / shard.name)

    # Write the canonical generation_config.json as a real file (not a link).
    (STAGING / "generation_config.json").write_bytes(gen_bytes)

    linked = len(HARDLINK_NAMES) + len(shards)
    print(f"staging built at {STAGING}: {linked} hardlinks + 1 canonical file")


def verify_all_resources() -> None:
    expected = {
        "tokenizer.json": "0997f410c57a",
        "tokenizer_config.json": "b11349aafa7c",
        "chat_template.jinja": "c3cf9e34abf4",
        "generation_config.json": "e70c136c1b78",
        "preprocessor_config.json": "27225450ac9c",
        "video_preprocessor_config.json": "7768af27c1fa",
    }
    ok = True
    for name, pfx in expected.items():
        got = sha256((STAGING / name).read_bytes())
        mark = "OK " if got.startswith(pfx) else "DIFF"
        if not got.startswith(pfx):
            ok = False
        print(f"  {mark} {name:32s} {got[:12]}")
    if not ok:
        raise SystemExit("resource verification failed")
    print("all six frontend resources verified against pinned hashes")


if __name__ == "__main__":
    gen = fetch_gen_config()
    build_staging(gen)
    verify_all_resources()
