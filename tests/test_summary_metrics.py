"""Unit tests for statistical summary metrics and aggregation."""

import math

import pytest

from inference_os.metrics import (
    RequestMeasurement,
    calculate_metric_stats,
    compute_benchmark_summary,
)


def test_calculate_metric_stats_empty() -> None:
    """Verify calculate_metric_stats returns None for empty sequence."""
    assert calculate_metric_stats([]) is None


def test_calculate_metric_stats_single_value() -> None:
    """Verify statistics for a single value."""
    stats = calculate_metric_stats([42.0])
    assert stats is not None
    assert stats.count == 1
    assert stats.mean == 42.0
    assert stats.std_dev == 0.0
    assert stats.min == 42.0
    assert stats.max == 42.0
    assert stats.p50 == 42.0
    assert stats.p90 == 42.0
    assert stats.p95 == 42.0
    assert stats.p99 == 42.0


def test_calculate_metric_stats_known_distribution() -> None:
    """Verify exact percentile and variance math for a known sequence."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    stats = calculate_metric_stats(values)

    assert stats is not None
    assert stats.count == 5
    assert stats.mean == pytest.approx(3.0)
    assert stats.min == pytest.approx(1.0)
    assert stats.max == pytest.approx(5.0)
    assert stats.std_dev == pytest.approx(math.sqrt(2.5))

    # Linear interpolation percentiles for 5 items (indices 0, 1, 2, 3, 4)
    # p50 = rank 2.0 -> value 3.0
    assert stats.p50 == pytest.approx(3.0)
    # p90 = rank 3.6 -> (1 - 0.6)*4.0 + 0.6*5.0 = 4.6
    assert stats.p90 == pytest.approx(4.6)
    # p95 = rank 3.8 -> (1 - 0.8)*4.0 + 0.8*5.0 = 4.8
    assert stats.p95 == pytest.approx(4.8)
    # p99 = rank 3.96 -> (1 - 0.96)*4.0 + 0.96*5.0 = 4.96
    assert stats.p99 == pytest.approx(4.96)


def test_compute_benchmark_summary_success_and_throughput() -> None:
    """Verify aggregate summary and throughput math for successful requests."""
    # 3 successful requests:
    # req 1: start 1s, first 1.2s (TTFT 0.2), comp 2.0s (E2E 1.0), in 100, out 20
    # req 2: start 2s, first 2.3s (TTFT 0.3), comp 3.5s (E2E 1.5), in 100, out 30
    # req 3: start 4s, first 4.4s (TTFT 0.4), comp 6.0s (E2E 2.0), in 100, out 50
    m1 = RequestMeasurement(
        request_id="r1",
        start_time_ns=1_000_000_000,
        first_token_time_ns=1_200_000_000,
        completion_time_ns=2_000_000_000,
        input_tokens=100,
        output_tokens=20,
        success=True,
    )
    m2 = RequestMeasurement(
        request_id="r2",
        start_time_ns=2_000_000_000,
        first_token_time_ns=2_300_000_000,
        completion_time_ns=3_500_000_000,
        input_tokens=100,
        output_tokens=30,
        success=True,
    )
    m3 = RequestMeasurement(
        request_id="r3",
        start_time_ns=4_000_000_000,
        first_token_time_ns=4_400_000_000,
        completion_time_ns=6_000_000_000,
        input_tokens=100,
        output_tokens=50,
        success=True,
    )

    total_duration = 5.0  # seconds
    summary = compute_benchmark_summary([m1, m2, m3], total_duration)

    assert summary.total_requests == 3
    assert summary.successful_requests == 3
    assert summary.failed_requests == 0
    assert summary.total_input_tokens == 300
    assert summary.total_output_tokens == 100
    assert summary.total_duration_seconds == 5.0

    # Throughputs
    assert summary.request_throughput == pytest.approx(3 / 5.0)  # 0.6 req/s
    assert summary.output_token_throughput == pytest.approx(100 / 5.0)  # 20.0 tok/s

    # TTFT stats (0.2, 0.3, 0.4)
    assert summary.ttft_stats is not None
    assert summary.ttft_stats.count == 3
    assert summary.ttft_stats.mean == pytest.approx(0.3)
    assert summary.ttft_stats.p50 == pytest.approx(0.3)

    # E2E stats (1.0, 1.5, 2.0)
    assert summary.e2e_latency_stats is not None
    assert summary.e2e_latency_stats.count == 3
    assert summary.e2e_latency_stats.mean == pytest.approx(1.5)
    assert summary.e2e_latency_stats.p50 == pytest.approx(1.5)

    # TPOT stats:
    # m1: (2.0 - 1.2) / 19 = 0.8 / 19 = ~0.042105s
    # m2: (3.5 - 2.3) / 29 = 1.2 / 29 = ~0.041379s
    # m3: (6.0 - 4.4) / 49 = 1.6 / 49 = ~0.032653s
    assert summary.tpot_stats is not None
    assert summary.tpot_stats.count == 3
    expected_tpot_mean = (0.8 / 19 + 1.2 / 29 + 1.6 / 49) / 3
    assert summary.tpot_stats.mean == pytest.approx(expected_tpot_mean)


def test_compute_benchmark_summary_partial_failures() -> None:
    """Verify failed requests are excluded from latency metrics and counted properly."""
    m_success = RequestMeasurement(
        request_id="ok-1",
        start_time_ns=1_000_000_000,
        first_token_time_ns=1_200_000_000,
        completion_time_ns=2_000_000_000,
        input_tokens=50,
        output_tokens=20,
        success=True,
    )
    m_fail = RequestMeasurement(
        request_id="err-1",
        start_time_ns=2_000_000_000,
        completion_time_ns=2_500_000_000,
        input_tokens=50,
        output_tokens=0,
        success=False,
        error_message="HTTP 500",
    )

    summary = compute_benchmark_summary([m_success, m_fail], total_duration_seconds=2.0)

    assert summary.total_requests == 2
    assert summary.successful_requests == 1
    assert summary.failed_requests == 1
    assert summary.total_input_tokens == 100
    assert summary.total_output_tokens == 20

    # Only 1 successful request contributed to throughput
    assert summary.request_throughput == pytest.approx(1 / 2.0)
    assert summary.output_token_throughput == pytest.approx(20 / 2.0)

    # Latency stats should only reflect m_success
    assert summary.ttft_stats is not None
    assert summary.ttft_stats.count == 1
    assert summary.ttft_stats.mean == pytest.approx(0.2)

    assert summary.e2e_latency_stats is not None
    assert summary.e2e_latency_stats.count == 1
    assert summary.e2e_latency_stats.mean == pytest.approx(1.0)

    # Errors should be preserved
    assert summary.errors == ["HTTP 500"]


def test_compute_benchmark_summary_all_failures() -> None:
    """Verify metrics when all requests fail."""
    m_fail1 = RequestMeasurement(
        request_id="err-1",
        start_time_ns=1_000_000_000,
        completion_time_ns=1_100_000_000,
        input_tokens=50,
        output_tokens=0,
        success=False,
        error_message="Connection refused",
    )
    m_fail2 = RequestMeasurement(
        request_id="err-2",
        start_time_ns=2_000_000_000,
        completion_time_ns=2_100_000_000,
        input_tokens=50,
        output_tokens=0,
        success=False,
        error_message="HTTP 502",
    )

    summary = compute_benchmark_summary([m_fail1, m_fail2], total_duration_seconds=1.0)
    assert summary.total_requests == 2
    assert summary.successful_requests == 0
    assert summary.failed_requests == 2
    assert summary.request_throughput == 0.0
    assert summary.output_token_throughput == 0.0
    assert summary.ttft_stats is None
    assert summary.e2e_latency_stats is None
    assert summary.tpot_stats is None
    assert summary.errors == ["Connection refused", "HTTP 502"]
