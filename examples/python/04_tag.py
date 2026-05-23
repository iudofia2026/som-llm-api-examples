#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Tag a text with multiple topics."""

import json

from som_llm import client, current_model, no_thinking

llm = client()
model = current_model(llm)

allowed_tags = ["finance", "labor", "education", "health", "technology", "policy"]
text = "A school district approved funding for new tutoring software and teacher training."

response = llm.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "Return only JSON like {\"tags\": [...]} using only these tags: "
                + ", ".join(allowed_tags)
            ),
        },
        {"role": "user", "content": text},
    ],
    temperature=0,
    max_tokens=128,
    response_format={"type": "json_object"},
    extra_body=no_thinking(),
)

data = json.loads(response.choices[0].message.content)
print(data["tags"])
