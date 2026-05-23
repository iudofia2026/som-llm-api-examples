#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Classify text into one of a few labels."""

import os

from openai import OpenAI

BASE_URL = os.environ.get("SOM_LLM_BASE_URL", "https://api.som.chat/v1")


def current_model(client: OpenAI) -> str:
    """Return SOM_LLM_MODEL, or choose the first model from /v1/models."""
    if model := os.environ.get("SOM_LLM_MODEL"):
        return model

    models = [model.id for model in client.models.list().data]
    if not models:
        raise SystemExit("No models returned from /v1/models")
    return models[0]


def no_thinking() -> dict:
    """Qwen chat-template setting for short direct answers."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


api_key = os.environ.get("SOM_LLM_KEY")
if not api_key:
    raise SystemExit("Set SOM_LLM_KEY first")

client = OpenAI(api_key=api_key, base_url=BASE_URL)
model = current_model(client)

text = "New chip design cuts training cost by 30%."
labels = "business|science|technology|sports|other"

response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "Classify the text. Output exactly one label."},
        {"role": "user", "content": text},
    ],
    temperature=0,
    max_tokens=8,
    extra_body={
        "regex": f"({labels})",
        **no_thinking(),
    },
)

print(response.choices[0].message.content.strip())
