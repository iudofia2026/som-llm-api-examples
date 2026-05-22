#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

"""Efficient OpenAI-compatible client example for api.som.chat.

Shows bounded concurrency, short retries for transient failures, and one reused
HTTP client.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

DEFAULT_BASE_URL = "https://api.som.chat/v1"
DEFAULT_MODEL = "Qwen3.5-122B-A10B-FP8"
DEFAULT_CONCURRENCY = 3
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

PROMPTS = [
    "Reply with exactly one word: pong",
    "What is the capital of France? Reply in one short sentence.",
    "Classify this text as business, science, tech, or sports: New chip design cuts training cost by 30%.",
]


def chat_payload(prompt: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }


async def post_with_retries(
    client: Any,
    payload: dict[str, Any],
    *,
    max_attempts: int = 4,
) -> dict[str, Any]:
    import httpx

    backoff = 1.0

    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post("/chat/completions", json=payload, timeout=120)
            if response.status_code in TRANSIENT_STATUS and attempt < max_attempts:
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after else backoff
                await asyncio.sleep(delay)
                backoff *= 2
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == max_attempts:
                raise RuntimeError(f"request failed after {max_attempts} attempts: {exc}") from exc
            await asyncio.sleep(backoff)
            backoff *= 2

    raise RuntimeError("unreachable")


async def run_one(client: Any, semaphore: asyncio.Semaphore, prompt: str, model: str) -> tuple[str, float, str]:
    async with semaphore:
        start = time.monotonic()
        data = await post_with_retries(client, chat_payload(prompt, model=model))
        elapsed = time.monotonic() - start

    message = data["choices"][0]["message"].get("content", "").strip()
    return prompt, elapsed, message


def self_test() -> None:
    payload = chat_payload(PROMPTS[0])
    json.dumps(payload)
    assert payload["model"] == DEFAULT_MODEL
    assert payload["stream"] is False
    assert payload["chat_template_kwargs"]["enable_thinking"] is False
    assert DEFAULT_CONCURRENCY > 0
    assert 429 in TRANSIENT_STATUS and 503 in TRANSIENT_STATUS
    print("ok: efficient-client example is well-formed")


async def run_live_async() -> None:
    api_key = os.environ.get("SOM_LLM_KEY")
    if not api_key:
        raise SystemExit("Set SOM_LLM_KEY for --live")

    import httpx

    base_url = os.environ.get("SOM_LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("SOM_LLM_MODEL", DEFAULT_MODEL)
    concurrency = int(os.environ.get("SOM_LLM_CONCURRENCY", str(DEFAULT_CONCURRENCY)))
    semaphore = asyncio.Semaphore(concurrency)
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(base_url=base_url, headers=headers) as client:
        results = await asyncio.gather(*(run_one(client, semaphore, prompt, model) for prompt in PROMPTS))

    print(f"Endpoint: {base_url}")
    print(f"Model:    {model}")
    for prompt, elapsed, message in results:
        print(f"- {elapsed:.2f}s | {prompt[:40]} -> {message[:80]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="validate without API access")
    parser.add_argument("--live", action="store_true", help="call the live API; requires SOM_LLM_KEY")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
    elif args.live:
        asyncio.run(run_live_async())
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
