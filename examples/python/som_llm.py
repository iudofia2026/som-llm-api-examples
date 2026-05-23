"""Small helper used by the example scripts."""

from __future__ import annotations

import os

from openai import OpenAI

BASE_URL = os.environ.get("SOM_LLM_BASE_URL", "https://api.som.chat/v1")


def client() -> OpenAI:
    key = os.environ.get("SOM_LLM_KEY")
    if not key:
        raise SystemExit("Set SOM_LLM_KEY first")
    return OpenAI(api_key=key, base_url=BASE_URL)


def current_model(client: OpenAI, *, prefer: str = "general") -> str:
    """Return SOM_LLM_MODEL, or choose from /v1/models.

    `prefer="coding"` chooses an advertised model with "coder" in its id when
    available. Otherwise we use the first model returned by the service.
    """
    if model := os.environ.get("SOM_LLM_MODEL"):
        return model

    models = [model.id for model in client.models.list().data]
    if not models:
        raise SystemExit("No models returned from /v1/models")

    if prefer == "coding":
        for model in models:
            if "coder" in model.lower() or "code" in model.lower():
                return model

    return models[0]


def no_thinking() -> dict:
    """Qwen chat-template setting for short direct answers."""
    return {"chat_template_kwargs": {"enable_thinking": False}}
