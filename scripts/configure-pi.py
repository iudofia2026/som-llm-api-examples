#!/usr/bin/env python3
"""Install/update a Pi provider for the SOM LLM API.

This stores the environment variable name `SOM_LLM_KEY` in models.json, not the
plaintext key. Export SOM_LLM_KEY before running Pi.

Usage:
    export SOM_LLM_KEY=sk-som-...
    scripts/configure-pi.py
    pi --model som-chat/$(scripts/som-current-model.py):high
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.som.chat/v1"
DEFAULT_PROVIDER = "som-chat"


def api_key() -> str:
    key = os.environ.get("SOM_LLM_KEY") or os.environ.get("SOM_HPC_LLM_API_KEY")
    if not key:
        raise SystemExit("Set SOM_LLM_KEY")
    return key


def fetch_models(base_url: str) -> list[str]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + api_key()},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        payload: dict[str, Any] = json.load(response)
    return [item["id"] for item in payload.get("data", []) if isinstance(item, dict) and item.get("id")]


def model_entry(model_id: str) -> dict:
    is_coder = "coder" in model_id.lower() or "code" in model_id.lower()
    return {
        "id": model_id,
        "name": f"{model_id} (SOM LLM API)",
        # Important for Pi: with `reasoning: true` and the qwen-chat-template
        # compatibility setting, `:off` sends enable_thinking=false and
        # `:high` sends enable_thinking=true. With `reasoning: false`, Pi does
        # not send the Qwen thinking controls at all.
        "reasoning": True,
        "input": ["text"],
        "contextWindow": 65536,
        "maxTokens": 8192 if is_coder else 4096,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--base-url", default=os.environ.get("SOM_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument(
        "--models-json",
        default=os.path.expanduser("~/.pi/agent/models.json"),
        help="Pi models.json path",
    )
    args = parser.parse_args(argv)

    models = fetch_models(args.base_url)
    path = pathlib.Path(args.models_json)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"providers": {}}

    data.setdefault("providers", {})[args.provider] = {
        "baseUrl": args.base_url.rstrip("/"),
        "api": "openai-completions",
        "apiKey": "SOM_LLM_KEY",
        "compat": {
            "supportsDeveloperRole": False,
            "supportsReasoningEffort": False,
            "supportsUsageInStreaming": True,
            "maxTokensField": "max_tokens",
            "thinkingFormat": "qwen-chat-template",
        },
        "models": [model_entry(model_id) for model_id in models],
    }

    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"updated {path} provider {args.provider} with {len(models)} model(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
