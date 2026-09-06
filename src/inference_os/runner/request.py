"""Single-request benchmark runner for recording request lifecycle observations."""

import time
from typing import AsyncIterable, Callable, Optional, TypeVar

from inference_os.metrics.request import RequestMeasurement

T = TypeVar("T")
ClockFn = Callable[[], int]


async def run_single_request(
    request_id: str,
    stream: AsyncIterable[T],
    input_tokens: int,
    output_tokens: Optional[int | Callable[[], int]] = None,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> RequestMeasurement:
    """Consume an async stream of outputs and record lifecycle timestamps.

    Args:
        request_id: Unique identifier for the request.
        stream: Async iterable yielding output chunks.
        input_tokens: Number of prompt/input tokens.
        output_tokens: Optional explicit output token count or callable returning
            token count. If None, defaults to the number of observed stream chunks.
        clock_fn: Callable returning current monotonic time in nanoseconds.
            Defaults to time.perf_counter_ns.

    Returns:
        RequestMeasurement containing raw lifecycle timestamps and status.
    """
    start_time_ns = clock_fn()
    first_token_time_ns: Optional[int] = None
    observed_chunks = 0
    success = True
    error_message: Optional[str] = None

    try:
        async for _ in stream:
            if first_token_time_ns is None:
                first_token_time_ns = clock_fn()
            observed_chunks += 1
    except Exception as exc:
        success = False
        error_message = str(exc) or exc.__class__.__name__
    finally:
        completion_time_ns = clock_fn()

    if callable(output_tokens):
        try:
            final_output_tokens = output_tokens()
        except Exception:
            final_output_tokens = observed_chunks
    elif output_tokens is not None:
        final_output_tokens = output_tokens
    else:
        final_output_tokens = observed_chunks

    return RequestMeasurement(
        request_id=request_id,
        start_time_ns=start_time_ns,
        completion_time_ns=completion_time_ns,
        input_tokens=input_tokens,
        output_tokens=final_output_tokens,
        success=success,
        first_token_time_ns=first_token_time_ns,
        error_message=error_message,
    )
