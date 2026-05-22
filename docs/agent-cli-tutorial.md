# Agent CLI setup: Pi, Claude Code, and Codex

This guide shows how to point terminal coding agents at the SOM LLM API without hard-coding the current model id. Model ids change over time; fetch the current advertised model from `/v1/models`.

## Common setup

```sh
export SOM_LLM_KEY=sk-som-...
export SOM_LLM_BASE_URL=https://api.som.chat/v1

# Prefer a coding model if the service advertises one; otherwise use the first model.
export SOM_LLM_MODEL="$(scripts/som-current-model.py --purpose coding)"
```

To inspect all advertised model ids:

```sh
scripts/som-current-model.py --all
```

## Pi

Pi is the simplest direct integration because it can speak OpenAI Chat Completions.

Install or update the SOM provider in `~/.pi/agent/models.json`:

```sh
export SOM_LLM_KEY=sk-som-...
scripts/configure-pi.py
```

This writes `apiKey: "SOM_LLM_KEY"` to Pi config, so the plaintext key stays in your shell environment rather than in `models.json`.

Run Pi with the current coding model:

```sh
pi --model "som-chat/$(scripts/som-current-model.py --purpose coding):high"
```

Fast/no-thinking mode:

```sh
pi --model "som-chat/$(scripts/som-current-model.py --purpose coding):off"
```

One-shot smoke:

```sh
pi --no-tools --no-session -p \
  --model "som-chat/$(scripts/som-current-model.py --purpose coding):off" \
  "Reply with exactly one word: pong"
```

### Reasoning default

Use `reasoning: true` in Pi model entries for Qwen models. That does **not** force every request to think. It lets Pi's `:off` / `:high` selector send Qwen chat-template controls:

- `:off` -> `chat_template_kwargs.enable_thinking=false`
- `:high` -> `chat_template_kwargs.enable_thinking=true`

If `reasoning` is false, Pi does not send those thinking controls, so the model may use the backend default.

The generated Pi provider uses conservative compatibility settings:

```json
{
  "supportsDeveloperRole": false,
  "supportsReasoningEffort": false,
  "supportsUsageInStreaming": true,
  "maxTokensField": "max_tokens",
  "thinkingFormat": "qwen-chat-template"
}
```

## Claude Code

Claude Code can use the SOM API through its Anthropic-compatible endpoint. Configuration is just environment variables.

```sh
export SOM_LLM_KEY=sk-som-...
export SOM_LLM_MODEL="$(scripts/som-current-model.py --purpose coding)"

export ANTHROPIC_BASE_URL=https://api.som.chat
export ANTHROPIC_API_KEY="$SOM_LLM_KEY"
export ANTHROPIC_MODEL="$SOM_LLM_MODEL"

claude
```

For Claude Code's built-in model aliases, pin all three aliases to the currently advertised SOM model:

```sh
export ANTHROPIC_DEFAULT_SONNET_MODEL="$SOM_LLM_MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$SOM_LLM_MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$SOM_LLM_MODEL"
```

One-shot smoke:

```sh
ANTHROPIC_BASE_URL=https://api.som.chat \
ANTHROPIC_API_KEY="$SOM_LLM_KEY" \
ANTHROPIC_MODEL="$(scripts/som-current-model.py --purpose coding)" \
claude --bare --no-session-persistence --tools "" -p \
  "Reply with exactly one word: pong"
```

Notes:

- `ANTHROPIC_BASE_URL` expects an Anthropic Messages-compatible API, not raw OpenAI Chat Completions. `api.som.chat` provides this compatibility layer.
- The SOM Anthropic compatibility layer disables Qwen thinking unless the Anthropic request asks for thinking. This keeps normal Claude Code calls faster by default.

## Codex

Current status: **not directly supported yet**.

Codex custom providers use the OpenAI Responses API. The SOM API currently exposes OpenAI Chat Completions and Anthropic Messages compatibility, but not `/v1/responses`. A direct Codex custom-provider config reaches:

```text
https://api.som.chat/v1/responses
```

and currently gets `404 Not Found`.

Do not publish Codex as supported until the service implements Responses API or we add a Responses-to-Chat-Completions translation layer.

The expected future Codex shape is:

```toml
# ~/.codex/config.toml
profile = "som"

[profiles.som]
model = "<model from scripts/som-current-model.py>"
model_provider = "som"
model_reasoning_effort = "medium"
model_reasoning_summary = "none"
model_supports_reasoning_summaries = false

[model_providers.som]
name = "SOM LLM API"
base_url = "https://api.som.chat/v1"
env_key = "SOM_LLM_KEY"
wire_api = "responses"
```

Until `/v1/responses` exists, use Pi or Claude Code for SOM-backed coding agents.
