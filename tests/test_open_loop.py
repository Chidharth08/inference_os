"""Tests for constant-rate scheduling and overload accounting."""

import asyncio

import pytest

from inference_os.runner.load import constant_rate_arrival_offsets, run_open_loop


def test_constant_rate_arrivals_exclude_duration_boundary() -> None:
    assert constant_rate_arrival_offsets(2.0, 2.0) == (0.0, 0.5, 1.0, 1.5)
    assert len(constant_rate_arrival_offsets(0.5, 3.0)) == 2


def test_constant_rate_arrivals_validate_inputs() -> None:
    with pytest.raises(ValueError, match="request_rate"):
        constant_rate_arrival_offsets(0.0, 1.0)
    with pytest.raises(ValueError, match="duration"):
        constant_rate_arrival_offsets(1.0, 0.0)


def test_open_loop_records_overload_instead_of_self_throttling() -> None:
    asyncio.run(_assert_open_loop_records_overload())


async def _assert_open_loop_records_overload() -> None:
    async def request_factory(request_id: str, index: int, is_warmup: bool):
        async def stream():
            await asyncio.sleep(0.04)
            yield "token"

        return stream(), 10, 1

    result = await run_open_loop(
        request_factory,
        request_rate=100.0,
        duration_seconds=0.02,
        max_in_flight=1,
        max_drain_seconds=1.0,
    )

    assert len(result.measured_requests) == 2
    assert result.dispatched_requests == 1
    assert result.dropped_requests == 1
    assert result.max_observed_in_flight == 1
    assert result.measured_requests[1].error_message == "client_overload"
    assert all(item.scheduled_time_ns is not None for item in result.measured_requests)
