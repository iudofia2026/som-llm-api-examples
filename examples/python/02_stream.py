#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Stream tokens as they arrive."""

from som_llm import client, current_model, no_thinking

llm = client()
model = current_model(llm)

stream = llm.chat.completions.create(
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
