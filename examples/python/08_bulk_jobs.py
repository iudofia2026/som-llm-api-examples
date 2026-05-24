#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai"]
# ///

"""Run many independent jobs without hammering the API.

Copy this pattern for bulk work:

1. Limit concurrency with a small number of worker threads.
2. If the server sends Retry-After, wait that long before retrying.
3. If there is a timeout or connection problem, use exponential backoff.

The default is intentionally gentle: two workers.
"""

from __future__ import annotations

import email.utils
import os
import random
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from threading import Thread

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

BASE_URL = os.environ.get("SOM_LLM_BASE_URL", "https://api.som.chat/v1")
NUM_WORKERS = int(os.environ.get("SOM_LLM_BULK_WORKERS", "2"))
MAX_ATTEMPTS_PER_JOB = 6
MAX_WAIT_SECONDS = 60.0
RETRYABLE_STATUS_CODES = {429, 503, 529}


@dataclass(frozen=True)
class Job:
    job_id: str
    text: str


@dataclass(frozen=True)
class JobResult:
    job_id: str
    label: str | None = None
    error: BaseException | None = None


SAMPLE_JOBS = [
    Job("paper-001", "A study of market power, markups, and productivity in manufacturing."),
    Job("paper-002", "Evidence on student peer effects from randomized classroom assignment."),
    Job("paper-003", "A field experiment on email reminders and appointment attendance."),
    Job("paper-004", "An analysis of venture capital contracts and founder bargaining power."),
    Job("paper-005", "Local labor market effects of a new commuter rail station."),
    Job("paper-006", "The role of information frictions in household refinancing decisions."),
]


def iter_jobs() -> Iterator[Job]:
    """Yield jobs one at a time.

    In real bulk work, replace this with a CSV reader, database cursor, or
    other iterator. Do not load a huge dataset into memory first.
    """
    yield from SAMPLE_JOBS


def get_model_name(client: OpenAI) -> str:
    """Use SOM_LLM_MODEL if set; otherwise choose the first advertised model."""
    if model_name := os.environ.get("SOM_LLM_MODEL"):
        return model_name

    model_names = [model.id for model in client.models.list().data]
    if not model_names:
        raise SystemExit("No models returned from /v1/models")
    return model_names[0]


def disable_thinking() -> dict:
    """Qwen chat-template setting for short direct answers."""
    return {"chat_template_kwargs": {"enable_thinking": False}}


def parse_retry_after_header(headers: Mapping[str, str]) -> float | None:
    """Return Retry-After seconds if the header is present and parseable."""
    retry_after = headers.get("retry-after")
    if not retry_after:
        return None

    try:
        return min(float(retry_after), MAX_WAIT_SECONDS)
    except ValueError:
        pass

    try:
        retry_at = email.utils.parsedate_to_datetime(retry_after)
    except (TypeError, ValueError):
        return None

    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)

    seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return min(max(seconds, 0.0), MAX_WAIT_SECONDS)


def fallback_backoff_seconds(attempt_number: int) -> float:
    """Exponential backoff with a little jitter."""
    base_seconds = min(2 ** (attempt_number - 1), MAX_WAIT_SECONDS)
    jitter_seconds = random.uniform(0, min(base_seconds * 0.25, 2.0))
    return min(base_seconds + jitter_seconds, MAX_WAIT_SECONDS)


def retry_delay_seconds(attempt_number: int, headers: Mapping[str, str] | None = None) -> float:
    """Prefer server Retry-After; otherwise use local exponential backoff."""
    if headers:
        retry_after_seconds = parse_retry_after_header(headers)
        if retry_after_seconds is not None:
            return retry_after_seconds

    return fallback_backoff_seconds(attempt_number)


def safe_scheduler_headers(headers: Mapping[str, str]) -> str:
    """Return safe SOM scheduler headers for debugging.

    These headers do not contain prompts, responses, API keys, or request bodies.
    """
    header_names = [
        "x-som-admission-decision",
        "x-som-reject-reason",
        "x-som-queue-wait-ms",
        "x-som-queue-position",
        "x-som-scheduler-policy",
    ]
    parts = [f"{name}={headers[name]}" for name in header_names if name in headers]
    return "; ".join(parts)


def run_one_job_once(client: OpenAI, model_name: str, job: Job) -> str:
    """Send one request and return the model's label."""
    response_with_headers = client.chat.completions.with_raw_response.create(
        model=model_name,
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
        extra_body=disable_thinking(),
    )

    if scheduler_headers := safe_scheduler_headers(response_with_headers.headers):
        print(f"{job.job_id}: {scheduler_headers}")

    response = response_with_headers.parse()
    return (response.choices[0].message.content or "").strip()


def run_one_job_with_retries(client: OpenAI, model_name: str, job: Job) -> tuple[str, str]:
    """Run one job, retrying politely when the service says to try later."""
    for attempt_number in range(1, MAX_ATTEMPTS_PER_JOB + 1):
        try:
            label = run_one_job_once(client, model_name, job)
            return job.job_id, label
        except APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS_CODES:
                raise

            if scheduler_headers := safe_scheduler_headers(exc.response.headers):
                print(f"{job.job_id}: retryable HTTP {exc.status_code}; {scheduler_headers}")

            wait_seconds = retry_delay_seconds(attempt_number, exc.response.headers)
        except (APIConnectionError, APITimeoutError):
            print(f"{job.job_id}: transient connection issue")
            wait_seconds = retry_delay_seconds(attempt_number)

        if attempt_number == MAX_ATTEMPTS_PER_JOB:
            raise RuntimeError(f"{job.job_id} failed after {MAX_ATTEMPTS_PER_JOB} attempts")

        print(f"{job.job_id}: waiting {wait_seconds:.1f}s before attempt {attempt_number + 1}")
        time.sleep(wait_seconds)

    raise AssertionError("unreachable")


def worker_loop(
    api_key: str,
    model_name: str,
    job_queue: Queue[Job | None],
    result_queue: Queue[JobResult | None],
) -> None:
    """Process jobs until a None sentinel asks this worker to stop."""
    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60.0)

    while True:
        job = job_queue.get()
        try:
            if job is None:
                return

            job_id, label = run_one_job_with_retries(client, model_name, job)
            result_queue.put(JobResult(job_id=job_id, label=label))
        except Exception as exc:
            job_id = job.job_id if job is not None else "unknown"
            result_queue.put(JobResult(job_id=job_id, error=exc))
        finally:
            job_queue.task_done()


def result_printer(result_queue: Queue[JobResult | None], counts: dict[str, int]) -> None:
    """Print results as they finish so they do not build up in memory."""
    while True:
        result = result_queue.get()
        try:
            if result is None:
                return

            if result.error is None:
                counts["completed"] += 1
                print(f"{result.job_id}: {result.label}")
            else:
                counts["failed"] += 1
                print(f"{result.job_id}: failed: {result.error}")
        finally:
            result_queue.task_done()


def process_jobs(api_key: str, model_name: str, jobs: Iterable[Job], num_workers: int) -> int:
    """Process a job iterator with bounded memory.

    This is the important bulk pattern:

    - the producer reads one job at a time;
    - job_queue.maxsize prevents the producer from outrunning the workers;
    - workers make API calls and retry politely;
    - the result printer consumes results as they finish.
    """
    job_queue: Queue[Job | None] = Queue(maxsize=max(1, num_workers * 2))
    result_queue: Queue[JobResult | None] = Queue()
    counts = {"completed": 0, "failed": 0}

    workers = [
        Thread(
            target=worker_loop,
            args=(api_key, model_name, job_queue, result_queue),
            name=f"som-bulk-worker-{worker_number}",
        )
        for worker_number in range(1, num_workers + 1)
    ]
    printer = Thread(target=result_printer, args=(result_queue, counts), name="som-bulk-printer")

    for worker in workers:
        worker.start()
    printer.start()

    submitted_count = 0
    try:
        for job in jobs:
            job_queue.put(job)
            submitted_count += 1
            # Optional rate limit hook for strict quotas:
            # time.sleep(seconds_between_submissions)
    finally:
        for _ in workers:
            job_queue.put(None)

    job_queue.join()
    for worker in workers:
        worker.join()

    result_queue.put(None)
    result_queue.join()
    printer.join()

    if counts["failed"]:
        raise RuntimeError(f"{counts['failed']} job(s) failed")

    return submitted_count


def main() -> None:
    api_key = os.environ.get("SOM_LLM_KEY")
    if not api_key:
        raise SystemExit("Set SOM_LLM_KEY first")

    client = OpenAI(api_key=api_key, base_url=BASE_URL, timeout=60.0)
    model_name = get_model_name(client)

    # Keep this small. For real jobs, start low and increase only if the
    # service stays responsive.
    num_workers = max(1, NUM_WORKERS)
    print(f"Running with up to {num_workers} worker(s) on {model_name}")

    completed_count = process_jobs(api_key, model_name, iter_jobs(), num_workers)
    print(f"\nDone: {completed_count} job(s)")


if __name__ == "__main__":
    main()
