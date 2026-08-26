"""Unit tests for GPU telemetry collection and background sampler."""

import asyncio
from unittest.mock import patch

import pytest

from inference_os.telemetry import (
    GPUSample,
    GPUTelemetrySampler,
    compute_gpu_summary,
    parse_gpu_sample_output,
    query_gpu_sample,
)


def test_parse_gpu_sample_output_valid() -> None:
    """Verify parsing valid nvidia-smi sample output."""
    raw = "14200, 24576, 85, 42\n"
    sample = parse_gpu_sample_output(raw, timestamp_ns=1_000_000_000)

    assert sample is not None
    assert isinstance(sample, GPUSample)
    assert sample.timestamp_ns == 1_000_000_000
    assert sample.memory_used_mb == 14200
    assert sample.memory_total_mb == 24576
    assert sample.utilization_gpu_pct == 85
    assert sample.utilization_memory_pct == 42


def test_parse_gpu_sample_output_invalid() -> None:
    """Verify invalid and partial CSV strings return None."""
    assert parse_gpu_sample_output("") is None
    assert parse_gpu_sample_output("   \n") is None
    assert parse_gpu_sample_output("not_a_number, 24576, 80") is None
    assert parse_gpu_sample_output("14200, 24576") is None  # missing utilization


def test_compute_gpu_summary_empty() -> None:
    """Verify compute_gpu_summary returns None for empty sequence."""
    assert compute_gpu_summary([]) is None


def test_compute_gpu_summary_aggregation() -> None:
    """Verify peak and average metric calculations across samples."""
    s1 = GPUSample(
        timestamp_ns=100,
        memory_used_mb=10000,
        memory_total_mb=24576,
        utilization_gpu_pct=50,
    )
    s2 = GPUSample(
        timestamp_ns=200,
        memory_used_mb=16000,
        memory_total_mb=24576,
        utilization_gpu_pct=90,
    )
    s3 = GPUSample(
        timestamp_ns=300,
        memory_used_mb=12000,
        memory_total_mb=24576,
        utilization_gpu_pct=40,
    )

    summary = compute_gpu_summary([s1, s2, s3])
    assert summary is not None
    assert summary.sample_count == 3
    assert summary.peak_memory_used_mb == 16000
    assert summary.avg_memory_used_mb == pytest.approx((10000 + 16000 + 12000) / 3)
    assert summary.peak_utilization_gpu_pct == 90
    assert summary.avg_utilization_gpu_pct == pytest.approx(60.0)
    assert summary.total_memory_mb == 24576


def test_gpu_telemetry_sampler_lifecycle() -> None:
    """Verify async sampler collects periodic samples and terminates cleanly."""

    async def _test() -> None:
        count = 0

        def mock_query(device_idx: int) -> GPUSample:
            nonlocal count
            count += 1
            return GPUSample(
                timestamp_ns=count * 1_000_000,
                memory_used_mb=8000 + count * 500,
                memory_total_mb=24576,
                utilization_gpu_pct=30 + count * 10,
            )

        async with GPUTelemetrySampler(
            interval_seconds=0.01, query_fn=mock_query
        ) as sampler:
            await asyncio.sleep(0.04)

        samples = sampler.get_samples()
        assert len(samples) >= 2
        summary = sampler.get_summary()
        assert summary is not None
        assert summary.sample_count == len(samples)
        assert summary.peak_memory_used_mb > 8000
        assert summary.total_memory_mb == 24576

    asyncio.run(_test())


def test_query_gpu_sample_fallback() -> None:
    """Verify query_gpu_sample returns None when nvidia-smi is unavailable."""
    with patch("shutil.which", return_value=None):
        assert query_gpu_sample() is None
