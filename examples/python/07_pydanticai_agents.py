#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai", "pydantic-ai"]
# ///

"""Run a tiny two-agent PydanticAI workflow with local tools.

The workflow is intentionally small:

1. An intake agent reads a project request and calls a local dataset-catalog tool.
2. A reviewer agent reads the intake brief and calls a local row-count estimator.

Both agents use the SOM OpenAI-compatible endpoint.
"""

from __future__ import annotations

import json
import os
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

BASE_URL = os.environ.get("SOM_LLM_BASE_URL", "https://api.som.chat/v1")

DATASET_CATALOG = {
    "acs": {
        "name": "American Community Survey",
        "unit": "county-year",
        "years": "2012-2023",
        "fields": ["population", "median_income", "college_share", "unemployment_rate"],
    },
    "bea": {
        "name": "BEA Regional Economic Accounts",
        "unit": "county-year",
        "years": "2001-2023",
        "fields": ["employment", "personal_income", "gdp", "industry"],
    },
    "ipeds": {
        "name": "IPEDS Institutional Characteristics",
        "unit": "institution-year",
        "years": "2000-2023",
        "fields": ["tuition", "enrollment", "sector", "completion_rate"],
    },
}

ROWS_PER_YEAR = {"acs": 3144, "bea": 3144, "ipeds": 6400}


class ProjectBrief(BaseModel):
    dataset: Literal["acs", "bea", "ipeds"]
    research_question: str = Field(max_length=240)
    outcome: str = Field(max_length=80)
    covariates: list[str] = Field(max_length=5)
    next_step: str = Field(max_length=160)


class ReviewNote(BaseModel):
    feasible: bool
    estimated_rows: int
    caution: str = Field(max_length=180)
    suggested_check: str = Field(max_length=180)


def api_key() -> str:
    key = os.environ.get("SOM_LLM_KEY")
    if not key:
        raise SystemExit("Set SOM_LLM_KEY first")
    return key


def current_model(client: OpenAI, *, prefer: str = "general") -> str:
    """Return SOM_LLM_MODEL, or choose from /v1/models."""
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
    """Qwen chat-template setting for short tool/JSON workflows."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def som_model(model_name: str) -> OpenAIChatModel:
    provider = OpenAIProvider(base_url=BASE_URL, api_key=api_key())
    return OpenAIChatModel(model_name, provider=provider)


intake_agent = Agent(
    instructions=(
        "You are a research-project intake agent. Choose the best dataset for the request. "
        "You must call lookup_dataset before writing the final brief. Use only fields that "
        "appear in the tool result."
    ),
    output_type=ProjectBrief,
    model_settings={"temperature": 0, "max_tokens": 512, "extra_body": no_thinking()},
)


@intake_agent.tool_plain
def lookup_dataset(dataset: Literal["acs", "bea", "ipeds"]) -> str:
    """Return catalog metadata for one local dataset."""
    return json.dumps(DATASET_CATALOG[dataset])


reviewer_agent = Agent(
    instructions=(
        "You are a cautious workflow reviewer. Check whether the proposed dataset is feasible. "
        "You must call estimate_rows before writing the final review. Keep caution and suggested_check "
        "short, complete, and actionable."
    ),
    output_type=ReviewNote,
    model_settings={"temperature": 0, "max_tokens": 512, "extra_body": no_thinking()},
)


@reviewer_agent.tool_plain
def estimate_rows(dataset: Literal["acs", "bea", "ipeds"], start_year: int, end_year: int) -> int:
    """Estimate rows for a dataset-year panel."""
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    return ROWS_PER_YEAR[dataset] * (end_year - start_year + 1)


def main() -> None:
    discovery_client = OpenAI(api_key=api_key(), base_url=BASE_URL)
    model_name = current_model(discovery_client)
    model = som_model(model_name)

    request = (
        "I want to study whether county income and education are associated with "
        "unemployment after 2018. Suggest a small first workflow."
    )

    brief = intake_agent.run_sync(request, model=model).output
    review = reviewer_agent.run_sync(
        (
            "Review this brief for a 2018-2023 county panel. "
            f"Brief JSON: {brief.model_dump_json()}"
        ),
        model=model,
    ).output

    print("=== Intake brief ===")
    print(brief.model_dump_json(indent=2))
    print("\n=== Reviewer note ===")
    print(review.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
