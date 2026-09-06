"""Deterministic unit tests for closed-loop concurrent benchmark runner."""

import asyncio
from typing import AsyncGenerator, Callable, List, Optional

import pytest

from inference_os.config import BenchmarkConfig, SweepConfig
from inference_os.runner import run_benchmark


def make_fake_clock(timestamps: List[int]) -> Callable[[], int]:
    """Helper creating a deterministic nanosecond clock from a sequence."""
    times = iter(timestamps)

    def _clock() -> int:
        return next(times)

    return _clock


async def mock_async_stream(
    tokens: List[str],
    delay: float = 0.0,
) -> AsyncGenerator[str, None]:
    """Yield mock tokens asynchronously with optional delay."""
    for tok in tokens:
        if delay > 0:
            await asyncio.sleep(delay)
        yield tok


def test_concurrency_1_no_overlap() -> None:
    """Verify that concurrency=1 executes strictly sequentially with zero overlap."""

    async def _test() -> None:
        active_count = 0
        max_active = 0
        overlap_detected = False
        completed_order: list[str] = []

        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            nonlocal active_count, max_active, overlap_detected
            if active_count > 0:
                overlap_detected = True
            active_count += 1
            if active_count > max_active:
                max_active = active_count

            async def stream():
                try:
                    await asyncio.sleep(0.005)
                    yield "token"
                finally:
                    nonlocal active_count
                    active_count -= 1
                    completed_order.append(request_id)

            return stream(), 10, 1

        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=5,
            concurrency=1,
            warmup_requests=0,
        )

        assert not overlap_detected
        assert max_active == 1
        assert len(result.measured_requests) == 5
        assert completed_order == ["req-1", "req-2", "req-3", "req-4", "req-5"]
        assert result.summary.successful_requests == 5

    asyncio.run(_test())


def test_concurrency_c_never_exceeds_cap() -> None:
    """Verify that concurrency=C never exceeds C active in-flight requests."""

    async def _test() -> None:
        target_concurrency = 4
        total_requests = 12
        active_count = 0
        max_active = 0
        lock = asyncio.Lock()

        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            nonlocal active_count, max_active
            async with lock:
                active_count += 1
                if active_count > max_active:
                    max_active = active_count
                assert active_count <= target_concurrency

            async def stream():
                try:
                    await asyncio.sleep(0.01)
                    yield "chunk"
                finally:
                    async with lock:
                        nonlocal active_count
                        active_count -= 1

            return stream(), 10, 1

        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=total_requests,
            concurrency=target_concurrency,
            warmup_requests=0,
        )

        assert max_active == target_concurrency
        assert len(result.measured_requests) == total_requests
        assert result.summary.successful_requests == total_requests

    asyncio.run(_test())


def test_all_requests_finish() -> None:
    """Verify that all N requests are processed and returned in order."""

    async def _test() -> None:
        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            return mock_async_stream(["tok"]), 5, 1

        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=10,
            concurrency=3,
            warmup_requests=0,
        )

        assert len(result.measured_requests) == 10
        for i, m in enumerate(result.measured_requests):
            assert m.request_id == f"req-{i + 1}"
            assert m.success is True

        assert result.summary.total_requests == 10
        assert result.summary.successful_requests == 10
        assert result.summary.failed_requests == 0

    asyncio.run(_test())


def test_failures_do_not_deadlock() -> None:
    """Verify that failing requests do not hang or deadlock the worker queue."""

    async def _test() -> None:
        failing_indices = {1, 4}  # 0-indexed: req-2 and req-5

        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            if index in failing_indices:
                raise RuntimeError(f"Simulated network error on {request_id}")

            async def stream():
                yield "ok"

            return stream(), 10, 1

        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=6,
            concurrency=3,
            warmup_requests=0,
        )

        assert len(result.measured_requests) == 6
        assert result.summary.total_requests == 6
        assert result.summary.successful_requests == 4
        assert result.summary.failed_requests == 2
        assert result.summary.error_rate == pytest.approx(2 / 6)
        assert result.summary.errors is not None
        assert len(result.summary.errors) == 2
        assert "Simulated network error on req-2" in result.summary.errors[0]
        assert "Simulated network error on req-5" in result.summary.errors[1]

    asyncio.run(_test())


def test_throughput_uses_wall_clock_duration() -> None:
    """Verify throughput uses wall-clock time, not summed request latencies."""

    async def _test() -> None:
        # Simulate 2 concurrent workers running 4 requests:
        # Each request takes 100ms.
        # With concurrency=2, 4 requests take 200ms wall-clock time.
        # Sum of request latencies = 4 * 100ms = 400ms.
        # Correct throughput = 4 / 0.2s = 20.0 req/s (NOT 4 / 0.4s = 10.0 req/s).
        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            return mock_async_stream(["tok1", "tok2"], delay=0.05), 10, 2

        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=4,
            concurrency=2,
            warmup_requests=0,
        )

        summed_e2e = sum(m.e2e_latency_seconds for m in result.measured_requests)
        wall_clock = result.summary.total_duration_seconds

        # Wall clock duration must be strictly less than summed latencies
        assert wall_clock < summed_e2e * 0.8
        expected_req_tp = 4 / wall_clock
        assert result.summary.request_throughput == pytest.approx(expected_req_tp)
        assert result.summary.output_token_throughput == pytest.approx(8 / wall_clock)

    asyncio.run(_test())


def test_summaries_contain_only_measured_requests() -> None:
    """Verify warmup requests are isolated from measured summary under concurrency."""

    async def _test() -> None:
        async def request_factory(
            request_id: str, index: int, is_warmup: bool
        ) -> tuple[AsyncGenerator[str, None], int, Optional[int]]:
            return mock_async_stream(["t1", "t2"]), 10, 2

        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=4,
            concurrency=2,
            warmup_requests=2,
        )

        assert len(result.warmup_measurements) == 2
        assert len(result.measured_requests) == 4
        assert result.warmup_measurements[0].request_id == "warmup-1"
        assert result.warmup_measurements[1].request_id == "warmup-2"
        assert result.measured_requests[0].request_id == "req-1"

        assert result.summary.total_requests == 4
        assert result.summary.total_output_tokens == 8  # 4 reqs * 2 tokens
        assert result.warmup_summary is not None
        assert result.warmup_summary.total_requests == 2
        assert result.warmup_summary.total_output_tokens == 4

    asyncio.run(_test())


def test_concurrency_config_validation() -> None:
    """Verify validation of concurrency parameter in BenchmarkConfig and SweepConfig."""
    # Invalid concurrency <= 0
    with pytest.raises(ValueError, match="concurrency must be positive"):
        BenchmarkConfig(model="test-model", concurrency=0)

    with pytest.raises(ValueError, match="concurrency must be positive"):
        BenchmarkConfig(model="test-model", concurrency=-5)

    # Valid config
    cfg = BenchmarkConfig(model="test-model", concurrency=16)
    assert cfg.concurrency == 16

    # Serialization roundtrip
    d = cfg.to_dict()
    assert d["concurrency"] == 16
    cfg2 = BenchmarkConfig.from_dict(d)
    assert cfg2.concurrency == 16

    # SweepConfig validation
    sweep_cfg = SweepConfig(
        sweep_param="concurrency",
        sweep_values=(1, 2, 4, 8, 16, 32),
        base_config=cfg,
        experiment_id="E002",
    )
    assert sweep_cfg.sweep_param == "concurrency"
    assert sweep_cfg.sweep_values == (1, 2, 4, 8, 16, 32)
    point_cfgs = sweep_cfg.generate_point_configs()
    assert len(point_cfgs) == 6
    assert point_cfgs[0][1].concurrency == 1
    assert point_cfgs[5][1].concurrency == 32
