"""Sequential multi-request benchmark runner with warm-up support."""

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
    Awaitable[tuple[AsyncIterable[Any], int, Optional[int]]],
]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Complete results from a benchmark run including warmup and measured requests."""

    warmup_measurements: Sequence[RequestMeasurement]
    measured_requests: Sequence[RequestMeasurement]
    summary: BenchmarkSummary
    warmup_summary: Optional[BenchmarkSummary] = None


async def run_sequential_benchmark(
    request_factory: RequestFactory,
    num_requests: int,
    warmup_requests: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> BenchmarkResult:
    """Execute a sequential benchmark at concurrency 1.

    Args:
        request_factory: Async callable (request_id, index, is_warmup) returning
            (stream, input_tokens, optional_output_tokens).
        num_requests: Number of measured benchmark requests to run (must be > 0).
        warmup_requests: Number of warm-up requests to run prior to measurement (>= 0).
        clock_fn: Monotonic nanosecond clock function.

    Returns:
        BenchmarkResult containing raw warmup/measured records and aggregate summary.

    Raises:
        ValueError: If num_requests <= 0 or warmup_requests < 0.
    """
    if num_requests <= 0:
        raise ValueError(f"num_requests must be positive, got {num_requests}")
    if warmup_requests < 0:
        raise ValueError(f"warmup_requests cannot be negative, got {warmup_requests}")

    warmup_measurements: list[RequestMeasurement] = []
    warmup_start_time_ns = clock_fn() if warmup_requests > 0 else 0

    # Phase 1: Warm-up
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

    # Phase 2: Benchmark Measured Requests
    measured_requests: list[RequestMeasurement] = []
    benchmark_start_time_ns = clock_fn()

    for i in range(num_requests):
        request_id = f"req-{i + 1}"
        stream, input_tokens, output_tokens = await request_factory(
            request_id, i, False
        )
        measurement = await run_single_request(
            request_id=request_id,
            stream=stream,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            clock_fn=clock_fn,
        )
        measured_requests.append(measurement)

    benchmark_end_time_ns = clock_fn()
    benchmark_duration_seconds = (benchmark_end_time_ns - benchmark_start_time_ns) / 1e9

    # Phase 3: Compute Summary for Measured Requests
    summary = compute_benchmark_summary(measured_requests, benchmark_duration_seconds)

    return BenchmarkResult(
        warmup_measurements=warmup_measurements,
        measured_requests=measured_requests,
        summary=summary,
        warmup_summary=warmup_summary,
    )
