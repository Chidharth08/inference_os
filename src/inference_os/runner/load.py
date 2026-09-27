"""Duration-based open-loop request scheduling."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, replace
from typing import Awaitable, Callable, Sequence

from inference_os.metrics.request import RequestMeasurement
from inference_os.metrics.summary import BenchmarkSummary, compute_benchmark_summary
from inference_os.runner.benchmark import RequestFactory
from inference_os.runner.request import ClockFn, run_single_request

SleepFn = Callable[[float], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class InFlightSample:
    """A change in the number of dispatched requests still in progress."""

    timestamp_ns: int
    in_flight: int
    event: str
    request_id: str


@dataclass(frozen=True, slots=True)
class OpenLoopResult:
    """Measurements and load-specific accounting for one offered-rate point."""

    warmup_measurements: Sequence[RequestMeasurement]
    measured_requests: Sequence[RequestMeasurement]
    summary: BenchmarkSummary
    warmup_summary: BenchmarkSummary | None
    in_flight_samples: Sequence[InFlightSample]
    offered_rate: float
    offered_duration_seconds: float
    measured_duration_seconds: float
    dispatched_requests: int
    dropped_requests: int
    timed_out_requests: int
    max_observed_in_flight: int


def constant_rate_arrival_offsets(
    request_rate: float, duration_seconds: float
) -> tuple[float, ...]:
    """Return deterministic arrivals at i/rate for every offset below duration."""
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    count = int(math.ceil(request_rate * duration_seconds - 1e-12))
    return tuple(index / request_rate for index in range(count))


async def run_open_loop(
    request_factory: RequestFactory,
    request_rate: float,
    duration_seconds: float,
    *,
    max_in_flight: int,
    max_drain_seconds: float,
    warmup_requests: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
    sleep_fn: SleepFn = asyncio.sleep,
) -> OpenLoopResult:
    """Dispatch constant-rate arrivals independently of request completions.

    The in-flight cap is a client safety boundary, not closed-loop throttling. An
    arrival that finds the cap full is recorded immediately as ``client_overload``.
    """
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    if max_drain_seconds <= 0:
        raise ValueError("max_drain_seconds must be positive")
    if warmup_requests < 0:
        raise ValueError("warmup_requests cannot be negative")
    offsets = constant_rate_arrival_offsets(request_rate, duration_seconds)

    warmups: list[RequestMeasurement] = []
    warmup_start = clock_fn() if warmup_requests else 0
    for index in range(warmup_requests):
        request_id = f"warmup-{index + 1}"
        stream, input_tokens, output_tokens = await request_factory(
            request_id, index, True
        )
        warmups.append(
            await run_single_request(
                request_id,
                stream,
                input_tokens,
                output_tokens,
                clock_fn,
            )
        )
    warmup_duration = (
        (clock_fn() - warmup_start) / 1e9 if warmup_requests else 0.0
    )
    warmup_summary = (
        compute_benchmark_summary(warmups, warmup_duration) if warmups else None
    )

    measurements: list[RequestMeasurement | None] = [None] * len(offsets)
    in_flight_samples: list[InFlightSample] = []
    active: set[asyncio.Task[None]] = set()
    in_flight = 0
    max_observed = 0
    dispatched = 0
    dropped = 0
    timed_out = 0
    start_ns = clock_fn()

    async def execute(index: int, scheduled_ns: int, dispatch_ns: int) -> None:
        nonlocal in_flight, timed_out
        request_id = f"req-{index + 1}"
        try:
            stream, input_tokens, output_tokens = await request_factory(
                request_id, index, False
            )
            measurement = await run_single_request(
                request_id,
                stream,
                input_tokens,
                output_tokens,
                clock_fn,
            )
            measurements[index] = replace(
                measurement,
                scheduled_time_ns=scheduled_ns,
                dispatch_time_ns=dispatch_ns,
            )
        except asyncio.CancelledError:
            now = max(clock_fn(), dispatch_ns)
            measurements[index] = RequestMeasurement(
                request_id=request_id,
                start_time_ns=dispatch_ns,
                completion_time_ns=now,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message="drain_timeout",
                scheduled_time_ns=scheduled_ns,
                dispatch_time_ns=dispatch_ns,
            )
            timed_out += 1
        except Exception as exc:
            now = max(clock_fn(), dispatch_ns)
            measurements[index] = RequestMeasurement(
                request_id=request_id,
                start_time_ns=dispatch_ns,
                completion_time_ns=now,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message=str(exc) or exc.__class__.__name__,
                scheduled_time_ns=scheduled_ns,
                dispatch_time_ns=dispatch_ns,
            )
        finally:
            in_flight -= 1
            in_flight_samples.append(
                InFlightSample(clock_fn(), in_flight, "complete", request_id)
            )

    for index, offset in enumerate(offsets):
        target_ns = start_ns + round(offset * 1e9)
        remaining = (target_ns - clock_fn()) / 1e9
        if remaining > 0:
            await sleep_fn(remaining)
        now = max(clock_fn(), target_ns)
        active = {task for task in active if not task.done()}
        request_id = f"req-{index + 1}"
        if in_flight >= max_in_flight:
            dropped += 1
            measurements[index] = RequestMeasurement(
                request_id=request_id,
                start_time_ns=now,
                completion_time_ns=now,
                input_tokens=0,
                output_tokens=0,
                success=False,
                error_message="client_overload",
                scheduled_time_ns=target_ns,
                dispatch_time_ns=now,
            )
            in_flight_samples.append(
                InFlightSample(now, in_flight, "drop", request_id)
            )
            continue
        dispatched += 1
        in_flight += 1
        max_observed = max(max_observed, in_flight)
        in_flight_samples.append(InFlightSample(now, in_flight, "dispatch", request_id))
        task = asyncio.create_task(execute(index, target_ns, now))
        active.add(task)

    if active:
        done, pending = await asyncio.wait(active, timeout=max_drain_seconds)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.gather(*done, return_exceptions=True)

    end_ns = clock_fn()
    measured_duration = max((end_ns - start_ns) / 1e9, 1e-9)
    final_measurements = [item for item in measurements if item is not None]
    if len(final_measurements) != len(offsets):
        raise RuntimeError("open-loop scheduler lost request measurements")
    summary = compute_benchmark_summary(final_measurements, measured_duration)
    return OpenLoopResult(
        warmup_measurements=warmups,
        measured_requests=final_measurements,
        summary=summary,
        warmup_summary=warmup_summary,
        in_flight_samples=in_flight_samples,
        offered_rate=request_rate,
        offered_duration_seconds=duration_seconds,
        measured_duration_seconds=measured_duration,
        dispatched_requests=dispatched,
        dropped_requests=dropped,
        timed_out_requests=timed_out,
        max_observed_in_flight=max_observed,
    )
