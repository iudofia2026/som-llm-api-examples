# Python examples

Set your key once:

```sh
export SOM_LLM_KEY=sk-som-...
```

Optional: pin a model. If omitted, the examples ask `/v1/models` and use the first advertised model. Prefer discovering the current model instead of hardcoding one:

```sh
export SOM_LLM_MODEL="$(../../scripts/som-current-model.py --purpose general)"
```

Run any example directly. The files are intentionally standalone, so you can copy one script into your own project without importing a local helper package:

```sh
./01_chat.py
./02_stream.py
./03_classify.py
./04_tag.py
./05_extract_json.py
./06_thinking.py
./07_pydanticai_agents.py
./08_bulk_jobs.py
./09_audit_outputs.py ./outputs
```

## What each example shows

- `01_chat.py` — normal OpenAI-style chat completion.
- `02_stream.py` — streaming tokens as they arrive.
- `03_classify.py` — single-label classification with a regex constraint.
- `04_tag.py` — multi-label tagging with `json_object`.
- `05_extract_json.py` — typed JSON extraction validated by Pydantic.
- `06_thinking.py` — enabling Qwen thinking for multi-step reasoning.
- `07_pydanticai_agents.py` — a two-agent PydanticAI workflow with local tool calls.
- `08_bulk_jobs.py` — polite bulk jobs with bounded concurrency, `Retry-After`, and exponential backoff.
- `09_audit_outputs.py` — scan saved JSON sidecars for truncation and token usage.

For short classification/extraction jobs, the examples disable thinking:

```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

For harder reasoning, turn thinking on and give the model enough `max_tokens` for reasoning plus the final answer.

For bulk extraction, read [`../../docs/workload-shaping.md`](../../docs/workload-shaping.md). Use per-field caps: labels often need only a few tokens, normal JSON extraction often fits in 1K-4K, and high caps should be reserved for fields that actually truncate.

## Backpressure and polite retries

`08_bulk_jobs.py` shows how to run a small batch without increasing load when the service is busy or temporarily unavailable:

- `429` means a caller/key/user policy or queue limit was hit.
- OpenAI-compatible overloads use `503`.
- Anthropic-compatible overloads use `529 overloaded_error`.
- Keep concurrency bounded; the example defaults to eight workers and can be tuned with `SOM_LLM_BULK_WORKERS`.
- Always honor `Retry-After` when the server sends it.
- Use capped exponential backoff with jitter for connection errors, timeouts, or retryable responses without `Retry-After`.
- Stop or slow a batch after sustained throttling; hot retries waste shared capacity.
- `X-SOM-Admission-Decision`, `X-SOM-Reject-Reason`, and `X-SOM-Queue-Wait-Ms` are safe advisory metadata for logging/debugging.

## Auditing saved outputs

`09_audit_outputs.py` helps decide whether a larger cap is worth it:

```sh
./09_audit_outputs.py ./outputs --warn-over 8192 --show-length-files
```

If only a small fraction of outputs finish with `finish_reason=length`, keep the normal pass small and retry only those truncated sidecars with a higher cap.

## Agent workflow example

`07_pydanticai_agents.py` shows a tiny two-agent workflow:

1. an intake agent calls a local dataset-catalog tool and writes a structured project brief;
2. a reviewer agent calls a local row-count estimator and writes a feasibility note.

It is still a single standalone file. Copy it when you want a starting point for PydanticAI agents, local tools, and structured outputs against the SOM OpenAI-compatible endpoint.
