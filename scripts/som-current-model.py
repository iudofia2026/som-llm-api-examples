#!/usr/bin/env python3
"""Print a currently available SOM LLM API model id.

Defaults to a coding-agent-friendly choice: prefer a model id containing
"coder" when one is advertised by /v1/models, otherwise use the first model.

Usage:
    export SOM_LLM_KEY=sk-som-...
    scripts/som-current-model.py
    scripts/som-current-model.py --purpose general
    scripts/som-current-model.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.som.chat/v1"


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


def choose_model(models: list[str], purpose: str) -> str:
    if not models:
        raise SystemExit("/v1/models returned no models")

    if purpose == "coding":
        for model in models:
            if "coder" in model.lower() or "code" in model.lower():
                return model

    return models[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=os.environ.get("SOM_LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--purpose", choices=["coding", "general"], default="coding")
    parser.add_argument("--all", action="store_true", help="print all advertised model ids")
    args = parser.parse_args(argv)

    models = fetch_models(args.base_url)
    if args.all:
        print("\n".join(models))
    else:
        print(choose_model(models, args.purpose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
