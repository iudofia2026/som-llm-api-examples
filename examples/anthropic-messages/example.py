#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///

"""Minimal Anthropic-compatible Messages example for api.som.chat."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_BASE_URL = "https://api.som.chat"
DEFAULT_MODEL = "Qwen3.5-122B-A10B-FP8"


def message_payload(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
    }


def request_parts(api_key: str = "sk-som-example", model: str = DEFAULT_MODEL) -> tuple[str, dict[str, str], dict[str, Any]]:
    base_url = os.environ.get("SOM_LLM_ANTHROPIC_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base_url}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    return url, headers, message_payload(model=model)


def self_test() -> None:
    url, headers, payload = request_parts()
    json.dumps(payload)
    assert url == "https://api.som.chat/v1/messages"
    assert headers["x-api-key"] == "sk-som-example"
    assert headers["anthropic-version"] == "2023-06-01"
    assert payload["model"] == DEFAULT_MODEL
    assert payload["max_tokens"] == 32
    print("ok: anthropic-messages example is well-formed")


def run_live() -> None:
    api_key = os.environ.get("SOM_LLM_KEY")
    if not api_key:
        raise SystemExit("Set SOM_LLM_KEY for --live")

    import httpx

    model = os.environ.get("SOM_LLM_MODEL", DEFAULT_MODEL)
    url, headers, payload = request_parts(api_key=api_key, model=model)
    response = httpx.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
    print(text.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="validate without API access")
    parser.add_argument("--live", action="store_true", help="call the live API; requires SOM_LLM_KEY")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
    elif args.live:
        run_live()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
