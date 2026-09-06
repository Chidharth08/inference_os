"""Concurrent and sequential multi-request benchmark runner with warm-up support."""

import asyncio
import time
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    Awaitable,
    Callable,
    Optional,
    Sequence,
)

from inference_os.metrics.request import RequestMeasurement
from inference_os.metrics.summary import BenchmarkSummary, compute_benchmark_summary
from inference_os.runner.request import ClockFn, run_single_request

# Type alias for request factory:
# (request_id, index, is_warmup) ->
# Awaitable[(stream, input_tokens, optional_output_tokens)]
RequestFactory = Callable[
    [str, int, bool],
    Awaitable[tuple[AsyncIterable[Any], int, Optional[int | Callable[[], int]]]],
]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete results from a benchmark run including warmup and measured requests."""

    warmup_measurements: Sequence[RequestMeasurement]
    measured_requests: Sequence[RequestMeasurement]
    summary: BenchmarkSummary
    warmup_summary: Optional[BenchmarkSummary] = None


async def run_benchmark(
    request_factory: RequestFactory,
    num_requests: int,
    concurrency: int = 1,
    warmup_requests: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> BenchmarkResult:
    """Execute a benchmark with closed-loop client concurrency.

    Maintains up to `concurrency` requests in flight simultaneously using an
    asynchronous worker pool pulling from a queue. When one request completes,
    the worker immediately dequeues the next request until all `num_requests`
    are processed.

    Args:
        request_factory: Async callable (request_id, index, is_warmup) returning
            (stream, input_tokens, optional_output_tokens).
        num_requests: Number of measured benchmark requests to run (must be > 0).
        concurrency: Maximum number of concurrent in-flight requests (must be > 0).
        warmup_requests: Number of warm-up requests to run prior to measurement (>= 0).
        clock_fn: Monotonic nanosecond clock function.

    Returns:
        BenchmarkResult containing raw warmup/measured records and aggregate summary.

    Raises:
        ValueError: If num_requests <= 0, warmup_requests < 0, or concurrency <= 0.
    """
    if num_requests <= 0:
        raise ValueError(f"num_requests must be positive, got {num_requests}")
    if warmup_requests < 0:
        raise ValueError(f"warmup_requests cannot be negative, got {warmup_requests}")
    if concurrency <= 0:
        raise ValueError(f"concurrency must be positive, got {concurrency}")

    # Phase 1: Warm-up (executed sequentially to prime caches safely)
    warmup_measurements: list[RequestMeasurement] = []
    warmup_start_time_ns = clock_fn() if warmup_requests > 0 else 0

    for i in range(warmup_requests):
        request_id = f"warmup-{i + 1}"
        stream, input_tokens, output_tokens = await request_factory(request_id, i, True)
        measurement = await run_single_request(
            request_id=request_id,
            stream=stream,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            clock_fn=clock_fn,
        )
        warmup_measurements.append(measurement)

    warmup_end_time_ns = clock_fn() if warmup_requests > 0 else 0
    warmup_duration_seconds = (
        (warmup_end_time_ns - warmup_start_time_ns) / 1e9
        if warmup_requests > 0
        else 0.0
    )
    warmup_summary = (
        compute_benchmark_summary(warmup_measurements, warmup_duration_seconds)
        if warmup_requests > 0
        else None
    )

    # Phase 2: Benchmark Measured Requests (Closed-loop concurrency)
    queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
    for i in range(num_requests):
        queue.put_nowait((f"req-{i + 1}", i))

    measured_requests: list[Optional[RequestMeasurement]] = [None] * num_requests

    async def worker() -> None:
        while True:
            try:
                request_id, idx = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            try:
                stream, input_tokens, output_tokens = await request_factory(
                    request_id, idx, False
                )
                measurement = await run_single_request(
                    request_id=request_id,
                    stream=stream,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    clock_fn=clock_fn,
                )
            except Exception as exc:
                now = clock_fn()
                measurement = RequestMeasurement(
                    request_id=request_id,
                    start_time_ns=now,
                    completion_time_ns=now,
                    input_tokens=0,
                    output_tokens=0,
                    success=False,
                    error_message=str(exc) or exc.__class__.__name__,
                )

            measured_requests[idx] = measurement
            queue.task_done()

    num_workers = min(concurrency, num_requests)
    benchmark_start_time_ns = clock_fn()

    workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
    await asyncio.gather(*workers)

    benchmark_end_time_ns = clock_fn()
    benchmark_duration_seconds = (benchmark_end_time_ns - benchmark_start_time_ns) / 1e9

    # Collect and ensure all requests completed
    final_measured = [m for m in measured_requests if m is not None]

    # Phase 3: Compute Summary for Measured Requests
    summary = compute_benchmark_summary(final_measured, benchmark_duration_seconds)

    return BenchmarkResult(
        warmup_measurements=warmup_measurements,
        measured_requests=final_measured,
        summary=summary,
        warmup_summary=warmup_summary,
    )


async def run_sequential_benchmark(
    request_factory: RequestFactory,
    num_requests: int,
    warmup_requests: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> BenchmarkResult:
    """Execute a sequential benchmark at concurrency 1 (backwards-compatible wrapper).

    Args:
        request_factory: Async callable (request_id, index, is_warmup) returning
            (stream, input_tokens, optional_output_tokens).
        num_requests: Number of measured benchmark requests to run (must be > 0).
        warmup_requests: Number of warm-up requests to run prior to measurement (>= 0).
        clock_fn: Monotonic nanosecond clock function.

    Returns:
        BenchmarkResult containing raw warmup/measured records and aggregate summary.
    """
    return await run_benchmark(
        request_factory=request_factory,
        num_requests=num_requests,
        concurrency=1,
        warmup_requests=warmup_requests,
        clock_fn=clock_fn,
    )
