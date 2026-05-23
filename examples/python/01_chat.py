#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Ask a normal chat question."""

from som_llm import client, current_model, no_thinking

llm = client()
model = current_model(llm)

response = llm.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": "You are a concise research assistant."},
        {"role": "user", "content": "What are two uses of LLMs in empirical research?"},
    ],
    temperature=0,
    max_tokens=256,
    extra_body=no_thinking(),
)

print(response.choices[0].message.content)
