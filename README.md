# SOM LLM API examples

Human-friendly examples for the Yale SOM LLM API at `https://api.som.chat`.

The API supports:

- OpenAI-compatible chat completions at `https://api.som.chat/v1`
- Anthropic-compatible Messages at `https://api.som.chat/v1/messages`

## Setup

Get an API key from the SOM LLM API dashboard, then:

```sh
export SOM_LLM_KEY=sk-som-...
```

The hosted model changes over time. Each Python example is standalone: it creates an OpenAI client, asks `/v1/models`, and chooses an advertised model automatically. Normally leave `SOM_LLM_MODEL` unset. If you need to pin the current advertised model for a shell session:

```sh
export SOM_LLM_MODEL="$(scripts/som-current-model.py --purpose general)"
```

Do not commit API keys, private prompts, or model outputs from sensitive work.

## Python examples

Start here: [`examples/python`](examples/python/)

```sh
cd examples/python
./01_chat.py
./02_stream.py
./03_classify.py
./04_tag.py
./05_extract_json.py
./06_thinking.py
./07_pydanticai_agents.py
./08_bulk_jobs.py
```

Examples included:

| File | Shows |
|---|---|
| `01_chat.py` | basic chat completion |
| `02_stream.py` | streaming tokens |
| `03_classify.py` | single-label classification |
| `04_tag.py` | multi-label tagging |
| `05_extract_json.py` | JSON extraction + Pydantic validation |
| `06_thinking.py` | Qwen thinking mode for harder reasoning |
| `07_pydanticai_agents.py` | two-agent PydanticAI workflow with local tools |
| `08_bulk_jobs.py` | polite bulk jobs with bounded concurrency, `Retry-After`, and exponential backoff |

## Backpressure etiquette

When the service is busy, retryable responses include `Retry-After` plus advisory `X-SOM-*` scheduler headers. Use `examples/python/08_bulk_jobs.py` as the starting point for bulk clients that should keep concurrency bounded, honor `Retry-After`, and fall back to exponential backoff during transient downtime.

## Agent CLI setup

See [`docs/agent-cli-tutorial.md`](docs/agent-cli-tutorial.md) for Pi, Claude Code, and Codex status.

Short version:

- Pi works directly via OpenAI Chat Completions.
- Claude Code works through the Anthropic-compatible endpoint.
- Codex is not directly supported yet because current Codex custom providers require `/v1/responses`.

Use [`scripts/som-current-model.py`](scripts/som-current-model.py) if you want the current advertised model in shell scripts:

```sh
export SOM_LLM_MODEL="$(scripts/som-current-model.py --purpose coding)"
```

## Structured output guidance

For extraction/classification:

- Put `response_format` at the top level of the OpenAI request, not inside `extra_body`.
- Use `response_format={"type": "json_object"}` for long or open-ended JSON extraction.
- Disable Qwen thinking for mechanical JSON formatting:
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`.
- Consider two passes for rich documents: free-text notes first, then JSON conversion.
- Use strict JSON Schema only when every array/string is bounded with `maxItems`, `maxLength`, enums, or numeric bounds.
- Validate responses client-side when correctness matters.
