#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Minimal OpenAI-compatible chat example for api.som.chat."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_BASE_URL = "https://api.som.chat/v1"
DEFAULT_MODEL = "Qwen3.5-122B-A10B-FP8"


def chat_request(model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly one word: pong"}],
        "temperature": 0,
        "max_tokens": 16,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def self_test() -> None:
    payload = chat_request()
    json.dumps(payload)
    assert payload["model"] == DEFAULT_MODEL
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 16
    assert payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    print("ok: openai-chat example is well-formed")


def run_live() -> None:
    api_key = os.environ.get("SOM_LLM_KEY")
    if not api_key:
        raise SystemExit("Set SOM_LLM_KEY for --live")

    from openai import OpenAI

    base_url = os.environ.get("SOM_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("SOM_LLM_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(**chat_request(model=model))
    print((response.choices[0].message.content or "").strip())


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
