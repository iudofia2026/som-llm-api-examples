# Workload shaping for bulk LLM jobs

Shared GPU services work best when clients describe the job size honestly and avoid hot retries. For bulk extraction, small choices in `max_tokens`, concurrency, and retry behavior often matter more than model parameters.

## Practical defaults

Always set `max_tokens` intentionally. Do not rely on server or SDK defaults for bulk jobs.

Suggested starting caps:

| Workload | Starting `max_tokens` |
|---|---:|
| one-label classification | 8-32 |
| short field extraction | 64-256 |
| short summary | 512-1,024 |
| ordinary JSON extraction | 1,024-4,096 |
| large arrays / long records | try 4,096, then 8,192 if needed |
| very large extraction | split the job or run as a slower/off-hours batch |

Use the smallest cap that fits the field. A high cap reserves shared capacity even when the final answer is short.

## Keep routine jobs routine

Requests with very high output caps, strict structured output, reasoning/thinking, or high fanout may be scheduled through more constrained capacity. To keep ordinary extraction fast:

- keep routine extraction at or below a few thousand output tokens;
- avoid `16K+` caps unless data proves they are needed;
- disable Qwen thinking for mechanical extraction:

  ```python
  extra_body={"chat_template_kwargs": {"enable_thinking": False}}
  ```

- use `temperature=0` for deterministic extraction/classification;
- keep worker concurrency modest and increase only after observing healthy throughput.

## Structured output tips

Strict JSON Schema is useful, but unbounded schemas can keep constrained decoding alive until `max_tokens`.

Prefer bounded schemas:

- add `maxItems` to arrays;
- add `maxLength` to strings;
- use enums for small closed sets;
- add numeric `minimum` / `maximum` where meaningful;
- avoid asking for giant arrays in one response if the data can be chunked.

For long or open-ended JSON, consider `response_format={"type": "json_object"}` plus client-side validation instead of strict schema.

## Handle 429/503/529 politely

Retryable status codes are not data loss:

- `429`: your key/user/job hit a policy, fairness, or queue limit;
- `503`: OpenAI-compatible backend overload or temporary unavailability;
- `529`: Anthropic-compatible overload.

When this happens:

1. Persist successful outputs before retrying.
2. Honor `Retry-After` exactly when present.
3. Use capped exponential backoff with jitter when `Retry-After` is absent.
4. Cap the exponent; do not let `2 ** consecutive_429s` grow without bound.
5. Stop or slow the batch after sustained throttling instead of spinning retries.
6. Resume only missing work from disk.

See [`examples/python/08_bulk_jobs.py`](../examples/python/08_bulk_jobs.py) for a bounded-concurrency retry pattern.

## Data-driven cap tuning

Before raising caps globally, inspect completed outputs:

- How often did `finish_reason` equal `length`?
- How often did a retry with a higher cap succeed?
- Among successful high-cap retries, how many actually used more than the smaller cap?
- Are some extraction fields always small enough for much lower caps?

If a large cap is needed in less than a few percent of cases, keep the normal pass small and retry only truncated records with a higher cap. This protects everyone else and often improves total throughput.

Use [`examples/python/09_audit_outputs.py`](../examples/python/09_audit_outputs.py) as a starting point for scanning saved JSON responses.

## Recommended bulk pattern

For each input record:

1. Save one sidecar file per extraction step.
2. Skip sidecars that already completed cleanly.
3. Use per-field `max_tokens`, not one large cap for every field.
4. Run normal/small fields first.
5. Retry only truncated fields with a larger cap.
6. Keep high-cap retries in a separate, lower-concurrency queue.
7. Pause after sustained throttling and resume later from disk.

This pattern avoids wasting shared GPU capacity and makes long jobs recoverable after local crashes, network failures, or service backpressure.
