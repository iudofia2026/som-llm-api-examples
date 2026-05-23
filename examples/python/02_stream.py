#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Stream tokens as they arrive."""

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

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a short haiku about machine learning."}],
    temperature=0.7,
    max_tokens=128,
    stream=True,
    extra_body=no_thinking(),
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
