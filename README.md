# SOM LLM API examples

Runnable examples for the Yale SOM HPC LLM API at `https://api.som.chat`.

The service exposes:

- an OpenAI-compatible API at `https://api.som.chat/v1`
- an Anthropic-compatible Messages API at `https://api.som.chat/v1/messages`

## Setup

Use an API key from your SOM LLM API dashboard:

```sh
export SOM_LLM_KEY=sk-som-...
```

Optional environment variables:

```sh
export SOM_LLM_BASE_URL=https://api.som.chat/v1
export SOM_LLM_MODEL=Qwen3.5-122B-A10B-FP8
```

Do not commit API keys, prompts containing sensitive data, or model outputs from private workloads. Also **note that our hosted models change often** and the code here might not work if you just copy/paste. 

## Agent CLI setup

See [`docs/agent-cli-tutorial.md`](docs/agent-cli-tutorial.md) for Pi, Claude Code, and Codex status.

Short version:

- Pi works directly via OpenAI Chat Completions.
- Claude Code works through the Anthropic-compatible endpoint.
- Codex is not directly supported yet because current Codex custom providers require `/v1/responses`.

Use [`scripts/som-current-model.py`](scripts/som-current-model.py) to avoid hard-coding model ids:

```sh
export SOM_LLM_MODEL="$(scripts/som-current-model.py --purpose coding)"
```

## Examples

| Example | Shows |
|---|---|
| [`examples/openai-chat`](examples/openai-chat/) | Basic OpenAI-compatible chat call |
| [`examples/anthropic-messages`](examples/anthropic-messages/) | Basic Anthropic-compatible Messages call |
| [`examples/efficient-client`](examples/efficient-client/) | Bounded concurrency, retries, and HTTP client reuse |
| [`examples/json-object-extraction`](examples/json-object-extraction/) | JSON object extraction, two-pass extraction, bounded strict JSON Schema |

Every example has an offline `--self-test` mode that does not call the API:

```sh
scripts/run-self-tests.sh
```

Run a live example with:

```sh
examples/openai-chat/example.py --live
examples/json-object-extraction/example.py --live
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

## Live-test caveat

Live tests require network access to `api.som.chat` and a valid `SOM_LLM_KEY`. Offline self-tests are the default CI gate.
