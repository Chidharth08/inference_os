"""Unit tests for run_single_request with deterministic fake clocks."""

import asyncio
from typing import AsyncGenerator, Callable, List, Optional, TypeVar

import pytest

from inference_os.runner import run_single_request

T = TypeVar("T")


def make_fake_clock(timestamps: List[int]) -> Callable[[], int]:
    """Helper creating a deterministic nanosecond clock from a sequence."""
    times = iter(timestamps)

    def _clock() -> int:
        return next(times)

    return _clock


async def async_stream(
    items: List[T],
    raise_error_index: Optional[int] = None,
    error_msg: str = "Stream error",
) -> AsyncGenerator[T, None]:
    """Helper async generator yielding items with optional error injection."""
    for idx, item in enumerate(items):
        if raise_error_index == idx:
            raise RuntimeError(error_msg)
        yield item
    if raise_error_index == len(items):
        raise RuntimeError(error_msg)


def test_successful_multi_chunk_request() -> None:
    """Verify lifecycle recording for a multi-chunk successful stream."""

    async def _test() -> None:
        clock = make_fake_clock([100, 250, 900])

        stream = async_stream(["chunk1", "chunk2", "chunk3"])
        measurement = await run_single_request(
            request_id="req-1",
            stream=stream,
            input_tokens=100,
            clock_fn=clock,
        )

        assert measurement.request_id == "req-1"
        assert measurement.start_time_ns == 100
        assert measurement.first_token_time_ns == 250
        assert measurement.completion_time_ns == 900
        assert measurement.input_tokens == 100
        assert measurement.output_tokens == 3
        assert measurement.success is True
        assert measurement.error_message is None

        assert measurement.ttft_seconds == pytest.approx(0.000000150)
        assert measurement.e2e_latency_seconds == pytest.approx(0.000000800)

    asyncio.run(_test())


def test_successful_single_chunk_request() -> None:
    """Verify first token timestamp is captured exactly once on item 1."""

    async def _test() -> None:
        clock = make_fake_clock([1_000_000_000, 1_200_000_000, 1_800_000_000])

        stream = async_stream(["token1"])
        measurement = await run_single_request(
            request_id="req-2",
            stream=stream,
            input_tokens=50,
            output_tokens=10,  # explicit output count override
            clock_fn=clock,
        )

        assert measurement.start_time_ns == 1_000_000_000
        assert measurement.first_token_time_ns == 1_200_000_000
        assert measurement.completion_time_ns == 1_800_000_000
        assert measurement.output_tokens == 10
        assert measurement.success is True
        assert measurement.ttft_seconds == pytest.approx(0.2)
        assert measurement.e2e_latency_seconds == pytest.approx(0.8)

    asyncio.run(_test())


def test_empty_stream() -> None:
    """Verify behavior when stream yields no outputs."""

    async def _test() -> None:
        clock = make_fake_clock([500, 900])

        stream = async_stream([])
        measurement = await run_single_request(
            request_id="req-empty",
            stream=stream,
            input_tokens=20,
            clock_fn=clock,
        )

        assert measurement.start_time_ns == 500
        assert measurement.first_token_time_ns is None
        assert measurement.completion_time_ns == 900
        assert measurement.output_tokens == 0
        assert measurement.success is True
        assert measurement.error_message is None
        assert measurement.ttft_seconds is None
        assert measurement.e2e_latency_seconds == pytest.approx(0.000000400)

    asyncio.run(_test())


def test_exception_before_first_output() -> None:
    """Verify error handling when exception occurs prior to yielding any chunk."""

    async def _test() -> None:
        clock = make_fake_clock([1_000, 5_000])

        stream = async_stream(
            ["chunk1"], raise_error_index=0, error_msg="Connection failed"
        )
        measurement = await run_single_request(
            request_id="req-fail-prefill",
            stream=stream,
            input_tokens=30,
            clock_fn=clock,
        )

        assert measurement.start_time_ns == 1_000
        assert measurement.first_token_time_ns is None
        assert measurement.completion_time_ns == 5_000
        assert measurement.success is False
        assert measurement.error_message == "Connection failed"
        assert measurement.ttft_seconds is None
        assert measurement.e2e_latency_seconds == pytest.approx(0.000004000)

    asyncio.run(_test())


def test_exception_after_first_output() -> None:
    """Verify error handling when exception occurs after first token emission."""

    async def _test() -> None:
        clock = make_fake_clock([10_000, 20_000, 60_000])

        stream = async_stream(
            ["chunk1", "chunk2"],
            raise_error_index=1,
            error_msg="Stream interrupted",
        )
        measurement = await run_single_request(
            request_id="req-fail-decode",
            stream=stream,
            input_tokens=30,
            clock_fn=clock,
        )

        assert measurement.start_time_ns == 10_000
        assert measurement.first_token_time_ns == 20_000
        assert measurement.completion_time_ns == 60_000
        assert measurement.output_tokens == 1
        assert measurement.success is False
        assert measurement.error_message == "Stream interrupted"
        assert measurement.ttft_seconds == pytest.approx(0.000010000)
        assert measurement.e2e_latency_seconds == pytest.approx(0.000050000)

    asyncio.run(_test())
