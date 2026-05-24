#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Run a small batch politely with Retry-After and exponential backoff.

This is the pattern to copy when you have many independent jobs. It keeps
concurrency bounded, honors SOM scheduler Retry-After guidance when present,
and falls back to capped exponential backoff for transient downtime.
"""

from __future__ import annotations

import email.utils
import os
import random
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

BASE_URL = os.environ.get("SOM_LLM_BASE_URL", "https://api.som.chat/v1")
MAX_WORKERS = int(os.environ.get("SOM_LLM_BULK_WORKERS", "2"))
MAX_ATTEMPTS = 6
MAX_SLEEP_SECONDS = 60.0
RETRYABLE_STATUS_CODES = {429, 503, 529}


@dataclass(frozen=True)
class Job:
    id: str
    text: str


JOBS = [
    Job("paper-001", "A study of market power, markups, and productivity in manufacturing."),
    Job("paper-002", "Evidence on student peer effects from randomized classroom assignment."),
    Job("paper-003", "A field experiment on email reminders and appointment attendance."),
    Job("paper-004", "An analysis of venture capital contracts and founder bargaining power."),
    Job("paper-005", "Local labor market effects of a new commuter rail station."),
    Job("paper-006", "The role of information frictions in household refinancing decisions."),
]


class RetryableBackpressure(Exception):
    """The API returned a retryable status and optional scheduler headers."""

    def __init__(self, status_code: int, headers: Mapping[str, str]):
        self.status_code = status_code
        self.headers = headers
        super().__init__(f"retryable HTTP {status_code}")


class RetryableDowntime(Exception):
    """The request failed before an HTTP response was available."""


def current_model(client: OpenAI) -> str:
    """Return SOM_LLM_MODEL, or choose the first model from /v1/models."""
    if model := os.environ.get("SOM_LLM_MODEL"):
        return model

    models = [model.id for model in client.models.list().data]
    if not models:
        raise SystemExit("No models returned from /v1/models")
    return models[0]


def no_thinking() -> dict:
    """Qwen chat-template setting for short direct answers."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    """Parse Retry-After if present; return seconds, capped."""
    retry_after = headers.get("retry-after")
    if not retry_after:
        return None

    try:
        return min(float(retry_after), MAX_SLEEP_SECONDS)
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return min(max(delay, 0.0), MAX_SLEEP_SECONDS)


def exponential_backoff_seconds(attempt: int) -> float:
    """Capped exponential backoff with jitter for downtime/no-header cases."""
    base = min(2 ** (attempt - 1), MAX_SLEEP_SECONDS)
    jitter = random.uniform(0, min(base * 0.25, 2.0))
    return min(base + jitter, MAX_SLEEP_SECONDS)


def delay_for_retry(attempt: int, headers: Mapping[str, str] | None = None) -> float:
    """Prefer server Retry-After; otherwise use exponential backoff."""
    if headers:
        retry_after = retry_after_seconds(headers)
        if retry_after is not None:
            return retry_after

    return exponential_backoff_seconds(attempt)


def scheduler_summary(headers: Mapping[str, str]) -> str:
    """Return safe SOM scheduler metadata for logs/debugging."""
    names = [
        "x-som-admission-decision",
        "x-som-reject-reason",
        "x-som-queue-wait-ms",
        "x-som-queue-position",
        "x-som-scheduler-policy",
    ]
    parts = [f"{name}={headers[name]}" for name in names if name in headers]
    return "; ".join(parts)


def classify_job(client: OpenAI, model: str, job: Job) -> str:
    """Classify one job and return a short label."""
    raw_response = client.chat.completions.with_raw_response.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the research topic with one short label such as "
                    "finance, labor, education, health, macro, or methods. "
                    "Return only the label."
                ),
            },
            {"role": "user", "content": job.text},
        ],
        temperature=0,
        max_tokens=32,
        extra_body=no_thinking(),
    )

    if summary := scheduler_summary(raw_response.headers):
        print(f"{job.id}: {summary}")

    response = raw_response.parse()
    return (response.choices[0].message.content or "").strip()


def classify_with_retries(client: OpenAI, model: str, job: Job) -> tuple[str, str]:
    """Run one job, retrying politely for backpressure or transient downtime."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return job.id, classify_job(client, model, job)
        except APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS_CODES:
                raise

            if summary := scheduler_summary(exc.response.headers):
                print(f"{job.id}: retryable HTTP {exc.status_code}; {summary}")

            delay = delay_for_retry(attempt, exc.response.headers)
        except (APIConnectionError, APITimeoutError) as exc:
            # No reliable HTTP response: use exponential backoff. This covers
            # network blips, restarts, and other transient downtime.
            if attempt == MAX_ATTEMPTS:
                raise RetryableDowntime(f"{job.id} failed after downtime retries") from exc

            delay = delay_for_retry(attempt)
            print(f"{job.id}: transient connection issue; retrying")

        if attempt == MAX_ATTEMPTS:
            raise RuntimeError(f"{job.id} still busy after {MAX_ATTEMPTS} attempts")

        print(f"{job.id}: sleeping {delay:.1f}s before attempt {attempt + 1}")
        time.sleep(delay)

    raise AssertionError("unreachable")


def main() -> None:
    api_key = os.environ.get("SOM_LLM_KEY")
    if not api_key:
        raise SystemExit("Set SOM_LLM_KEY first")

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60.0)
    model = current_model(client)

    # Keep this small. For real jobs, start with low concurrency and increase
    # only if the service remains responsive.
    workers = max(1, min(MAX_WORKERS, len(JOBS)))
    print(f"Running {len(JOBS)} jobs with {workers} worker(s) on {model}")

    results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(classify_with_retries, client, model, job) for job in JOBS]

        for future in as_completed(futures):
            job_id, label = future.result()
            results[job_id] = label
            print(f"{job_id}: {label}")

    print("\nResults")
    for job in JOBS:
        print(f"{job.id}: {results[job.id]}")


if __name__ == "__main__":
    main()
