#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""JSON extraction examples for api.som.chat.

Patterns:
- `json_object` for long/open-ended extraction;
- two-pass notes -> JSON conversion for rich documents;
- strict `json_schema` only with bounded arrays/strings.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

DEFAULT_BASE_URL = "https://api.som.chat/v1"
DEFAULT_MODEL = "Qwen3.5-122B-A10B-FP8"

SAMPLE_TEXT = """
Board members reviewed a facilities repair proposal and approved a $12,000 roof
maintenance contract. The vote was unanimous. The superintendent noted that the
work should be completed before winter.
""".strip()

BOUNDED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "topic": {"type": "string", "maxLength": 80},
        "facts": {
            "type": "array",
            "maxItems": 3,
            "items": {"type": "string", "maxLength": 160},
        },
        "category": {
            "type": "string",
            "enum": ["budget spending", "school facilities", "procedural", "other"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["topic", "facts", "category", "confidence"],
}


def json_object_request(text: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Return only one valid JSON object. No markdown fencing.",
            },
            {
                "role": "user",
                "content": (
                    "Extract a short topic, up to three factual bullets, "
                    "a category, and a confidence score from this text:\n\n"
                    f"{text}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def notes_pass_request(text: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write concise extraction notes in plain text. Do not emit JSON. "
                    "Do not include hidden reasoning or markdown fencing."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Identify the key topic, facts, actors, and uncertainty in this text. "
                    "Be precise and do not invent details.\n\n"
                    f"{text}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
    }


def json_conversion_request(notes: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Convert notes to one valid JSON object. No markdown fencing.",
            },
            {
                "role": "user",
                "content": (
                    "Schema: {topic: string, facts: string[], category: string, "
                    "confidence: number}. Use only information in the notes.\n\n"
                    f"Notes:\n{notes}"
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def bounded_schema_request(text: str, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "Classify and summarize the text. Return JSON only."},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "bounded_extraction",
                "strict": True,
                "schema": BOUNDED_SCHEMA,
            },
        },
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def _assert_response_format_top_level(payload: dict[str, Any]) -> None:
    assert "response_format" in payload
    assert "response_format" not in payload.get("extra_body", {})


def _assert_schema_bounded(schema: dict[str, Any]) -> None:
    schema_type = schema.get("type")

    if schema_type == "array":
        assert "maxItems" in schema, f"array missing maxItems: {schema}"
        _assert_schema_bounded(schema.get("items", {}))
        return

    if schema_type == "string":
        assert "maxLength" in schema or "enum" in schema, f"string missing maxLength/enum: {schema}"
        return

    if schema_type == "object":
        assert schema.get("additionalProperties") is False
        for child in schema.get("properties", {}).values():
            _assert_schema_bounded(child)


def self_test() -> None:
    object_payload = json_object_request(SAMPLE_TEXT)
    notes_payload = notes_pass_request(SAMPLE_TEXT)
    conversion_payload = json_conversion_request("Topic: roof maintenance contract.")
    schema_payload = bounded_schema_request(SAMPLE_TEXT)

    for payload in [object_payload, notes_payload, conversion_payload, schema_payload]:
        json.dumps(payload)
        assert payload["model"] == DEFAULT_MODEL
        assert payload["temperature"] == 0
        assert "messages" in payload

    _assert_response_format_top_level(object_payload)
    _assert_response_format_top_level(conversion_payload)
    _assert_response_format_top_level(schema_payload)

    assert object_payload["response_format"] == {"type": "json_object"}
    assert conversion_payload["response_format"] == {"type": "json_object"}
    assert schema_payload["response_format"]["type"] == "json_schema"
    assert notes_payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert conversion_payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False

    _assert_schema_bounded(schema_payload["response_format"]["json_schema"]["schema"])
    print("ok: json-object-extraction example is well-formed")


def run_live() -> None:
    api_key = os.environ.get("SOM_LLM_KEY")
    if not api_key:
        raise SystemExit("Set SOM_LLM_KEY for --live")

    from openai import OpenAI

    base_url = os.environ.get("SOM_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.environ.get("SOM_LLM_MODEL", DEFAULT_MODEL)
    client = OpenAI(base_url=base_url, api_key=api_key)

    payload = json_object_request(SAMPLE_TEXT, model=model)
    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)
    print(json.dumps(parsed, indent=2))


def print_payloads() -> None:
    payloads = {
        "json_object": json_object_request(SAMPLE_TEXT),
        "notes_pass": notes_pass_request(SAMPLE_TEXT),
        "json_conversion": json_conversion_request("Topic: roof maintenance contract."),
        "bounded_schema": bounded_schema_request(SAMPLE_TEXT),
    }
    print(json.dumps(payloads, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="validate without API access")
    parser.add_argument("--print-payloads", action="store_true", help="print example request payloads")
    parser.add_argument("--live", action="store_true", help="call the live API; requires SOM_LLM_KEY")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
    elif args.print_payloads:
        print_payloads()
    elif args.live:
        run_live()
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
