#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Use thinking mode for a multi-step reasoning question."""

from som_llm import client, current_model

llm = client()
model = current_model(llm)

response = llm.chat.completions.create(
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
