#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Classify text into one of a few labels."""

from som_llm import client, current_model, no_thinking

llm = client()
model = current_model(llm)

text = "New chip design cuts training cost by 30%."
labels = "business|science|technology|sports|other"

response = llm.chat.completions.create(
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
