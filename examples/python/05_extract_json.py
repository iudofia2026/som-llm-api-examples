#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai", "pydantic"]
# ///

"""Extract typed JSON from text and validate it with Pydantic."""

import json

from pydantic import BaseModel

from som_llm import client, current_model, no_thinking


class PaperSummary(BaseModel):
    title: str
    main_finding: str
    methodology: str
    jel_codes: list[str]


abstract = """
We study how venture capital contracts have evolved over the past two decades.
Using a dataset of VC term sheets, we document that participating preferred
stock has declined since 2005 while simple preferred structures have become
more common. We show this shift is driven by increased competition among VCs
and the rising bargaining power of experienced founders.
""".strip()

llm = client()
model = current_model(llm)

response = llm.chat.completions.create(
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
