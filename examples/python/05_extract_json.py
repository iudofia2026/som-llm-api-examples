#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai", "pydantic"]
# ///

"""Extract typed JSON from text and validate it with Pydantic."""

import json
import os

from openai import OpenAI
from pydantic import BaseModel

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


class PaperSummary(BaseModel):
    title: str
    main_finding: str
    methodology: str
    jel_codes: list[str]


api_key = os.environ.get("SOM_LLM_KEY")
if not api_key:
    raise SystemExit("Set SOM_LLM_KEY first")

client = OpenAI(api_key=api_key, base_url=BASE_URL)
model = current_model(client)

abstract = """
We study how venture capital contracts have evolved over the past two decades.
Using a dataset of VC term sheets, we document that participating preferred
stock has declined since 2005 while simple preferred structures have become
more common. We show this shift is driven by increased competition among VCs
and the rising bargaining power of experienced founders.
""".strip()

response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "Extract paper metadata. Return only valid JSON matching this schema: "
                + json.dumps(PaperSummary.model_json_schema())
            ),
        },
        {"role": "user", "content": abstract},
    ],
    temperature=0,
    max_tokens=512,
    response_format={"type": "json_object"},
    extra_body=no_thinking(),
)

paper = PaperSummary.model_validate_json(response.choices[0].message.content)
print(f"Title: {paper.title}")
print(f"Finding: {paper.main_finding}")
print(f"Method: {paper.methodology}")
print(f"JEL: {paper.jel_codes}")
