"""Unit tests for sequential benchmark runner."""

import asyncio
from typing import AsyncGenerator, Callable, List, Optional

import pytest

from inference_os.runner import run_sequential_benchmark


def make_fake_clock(timestamps: List[int]) -> Callable[[], int]:
    """Helper creating a deterministic nanosecond clock from a sequence."""
    times = iter(timestamps)

    def _clock() -> int:
        return next(times)

    return _clock


async def mock_async_stream(tokens: List[str]) -> AsyncGenerator[str, None]:
    """Yield mock tokens asynchronously."""
    for tok in tokens:
        yield tok


def test_sequential_benchmark_warmup_separation() -> None:
    """Verify warm-up requests are excluded from primary summary."""

    async def _test() -> None:
        # Timestamps sequence (W=2, N=3):
        # 1. Warmup start: 0
        # 2. Warmup-1: start=100, first_token=150, completion=200
        # 3. Warmup-2: start=210, first_token=260, completion=300
        # 4. Warmup end: 310
        # 5. Benchmark start: 320
        # 6. Req-1: start=330, first_token=380, completion=450
        # 7. Req-2: start=460, first_token=510, completion=600
        # 8. Req-3: start=610, first_token=660, completion=800
        # 9. Benchmark end: 850
        timestamps = [
            0,  # warmup start
            100,
            150,
            200,  # warmup 1
            210,
            260,
            300,  # warmup 2
            310,  # warmup end
            320,  # benchmark start
            330,
            380,
            450,  # req 1
            460,
            510,
            600,  # req 2
            610,
            660,
            800,  # req 3
            850,  # benchmark end
        ]
        clock = make_fake_clock(timestamps)
        executed_requests: List[str] = []

        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            executed_requests.append(request_id)
            tokens = ["hello", "world"]
            return mock_async_stream(tokens), 10, len(tokens)

        result = await run_sequential_benchmark(
            request_factory=request_factory,
            num_requests=3,
            warmup_requests=2,
            clock_fn=clock,
        )

        # Execution order check
        assert executed_requests == [
            "warmup-1",
            "warmup-2",
            "req-1",
            "req-2",
            "req-3",
        ]

        # Raw records check
        assert len(result.warmup_measurements) == 2
        assert len(result.measured_requests) == 3

        # Summary check (warmup excluded)
        assert result.summary.total_requests == 3
        assert result.summary.successful_requests == 3
        assert result.summary.failed_requests == 0
        assert result.summary.total_output_tokens == 6  # 3 reqs * 2 tokens
        assert result.summary.total_input_tokens == 30  # 3 reqs * 10 tokens

        # Benchmark duration = (850 - 320) ns = 530 ns = 0.000000530 s
        assert result.summary.total_duration_seconds == pytest.approx(0.000000530)
        assert result.summary.request_throughput == pytest.approx(3 / 0.000000530)

        # Warmup summary check
        assert result.warmup_summary is not None
        assert result.warmup_summary.total_requests == 2
        assert result.warmup_summary.total_output_tokens == 4

    asyncio.run(_test())


def test_sequential_benchmark_cold_run() -> None:
    """Verify cold benchmark with 0 warm-up requests."""

    async def _test() -> None:
        # W=0, N=2
        timestamps = [
            1_000_000_000,  # benchmark start
            1_100_000_000,
            1_200_000_000,
            1_500_000_000,  # req 1
            1_600_000_000,
            1_700_000_000,
            2_000_000_000,  # req 2
            2_100_000_000,  # benchmark end
        ]
        clock = make_fake_clock(timestamps)

        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            assert is_warmup is False
            return mock_async_stream(["cold"]), 5, 1

        result = await run_sequential_benchmark(
            request_factory=request_factory,
            num_requests=2,
            warmup_requests=0,
            clock_fn=clock,
        )

        assert len(result.warmup_measurements) == 0
        assert result.warmup_summary is None
        assert len(result.measured_requests) == 2
        assert result.summary.total_requests == 2
        # (2.1 - 1.0)s = 1.1s
        assert result.summary.total_duration_seconds == pytest.approx(1.1)

    asyncio.run(_test())


def test_sequential_benchmark_validation_errors() -> None:
    """Verify ValueError on non-positive num_requests or negative warmup_requests."""

    async def dummy_factory(
        request_id: str, index: int, is_warmup: bool
    ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
        return mock_async_stream([]), 0, 0

    with pytest.raises(ValueError, match="num_requests must be positive"):
        asyncio.run(
            run_sequential_benchmark(request_factory=dummy_factory, num_requests=0)
        )

    with pytest.raises(ValueError, match="warmup_requests cannot be negative"):
        asyncio.run(
            run_sequential_benchmark(
                request_factory=dummy_factory,
                num_requests=2,
                warmup_requests=-1,
            )
        )
