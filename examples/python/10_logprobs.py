#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Inspect token probabilities for a short classification response."""

import math
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

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "Classify the headline. Output exactly one label: "
                "business, science, technology, sports, or other."
            ),
        },
        {"role": "user", "content": "New chip design cuts training cost by 30%."},
    ],
    temperature=0,
    max_tokens=8,
    logprobs=True,
    top_logprobs=5,
    extra_body=no_thinking(),
)

choice = response.choices[0]
content = choice.message.content
if content is None:
    raise SystemExit("The server did not return message content")

print(f"Classification: {content.strip()}")

if choice.logprobs is None or choice.logprobs.content is None:
    raise SystemExit("The server did not return token log probabilities")

remaining_content = content
for position, token_info in enumerate(choice.logprobs.content, start=1):
    if not remaining_content:
        break
    if not remaining_content.startswith(token_info.token):
        continue

    remaining_content = remaining_content.removeprefix(token_info.token)
    probability = math.exp(token_info.logprob)
    print(f"\nToken {position}: {token_info.token!r} ({probability:.2%})")

    for alternative in token_info.top_logprobs:
        alternative_probability = math.exp(alternative.logprob)
        print(f"  {alternative.token!r:<20} {alternative_probability:.2%}")

if remaining_content:
    raise SystemExit("Token log probabilities did not reconstruct the visible response")

print(
    "\nTreat these probabilities as model confidence signals, not guarantees of correctness."
)
