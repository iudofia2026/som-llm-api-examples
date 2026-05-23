#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Use thinking mode for a multi-step reasoning question."""

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


api_key = os.environ.get("SOM_LLM_KEY")
if not api_key:
    raise SystemExit("Set SOM_LLM_KEY first")

client = OpenAI(api_key=api_key, base_url=BASE_URL)
model = current_model(client)

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": (
                "A fund returns 3x on $100M over 10 years. It charges 2% annual "
                "management fees and 20% carry after returning capital. Estimate GP compensation."
            ),
        }
    ],
    temperature=0,
    max_tokens=4096,
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

message = response.choices[0].message
reasoning = getattr(message, "reasoning_content", None)

if reasoning:
    print("=== Reasoning preview ===")
    print(reasoning[:500].strip())
    print("...\n")

print("=== Answer ===")
print(message.content or "(No final answer; increase max_tokens.)")
