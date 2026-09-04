"""Statistical and aggregate summary metrics for benchmark runs."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from inference_os.metrics.request import RequestMeasurement


@dataclass(frozen=True, slots=True)
class MetricStats:
    """Distribution statistics for a continuous numerical metric (e.g. latency)."""

    count: int
    mean: float
    std_dev: float
    min: float
    max: float
    p50: float
    p90: float
    p95: float
    p99: float


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Aggregate performance and throughput summary for a benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_duration_seconds: float
    request_throughput: float
    output_token_throughput: float
    ttft_stats: Optional[MetricStats] = None
    e2e_latency_stats: Optional[MetricStats] = None
    tpot_stats: Optional[MetricStats] = None
    errors: Optional[list[str]] = None


def _calculate_percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Calculate percentile using linear interpolation between closest ranks."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]

    rank = (percentile / 100.0) * (n - 1)
    lower_idx = math.floor(rank)
    upper_idx = math.ceil(rank)
    weight = rank - lower_idx

    return (1.0 - weight) * sorted_values[lower_idx] + weight * sorted_values[upper_idx]


def calculate_metric_stats(values: Sequence[float]) -> Optional[MetricStats]:
    """Calculate distribution statistics for a sequence of float values.

    Returns None if the values sequence is empty.
    """
    if not values:
        return None

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / n

    if n > 1:
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / (n - 1)
        std_dev_val = math.sqrt(max(0.0, variance))
    else:
        std_dev_val = 0.0

    return MetricStats(
        count=n,
        mean=mean_val,
        std_dev=std_dev_val,
        min=sorted_vals[0],
        max=sorted_vals[-1],
        p50=_calculate_percentile(sorted_vals, 50.0),
        p90=_calculate_percentile(sorted_vals, 90.0),
        p95=_calculate_percentile(sorted_vals, 95.0),
        p99=_calculate_percentile(sorted_vals, 99.0),
    )


def compute_benchmark_summary(
    measurements: Sequence[RequestMeasurement],
    total_duration_seconds: float,
) -> BenchmarkSummary:
    """Compute aggregate performance and throughput metrics for a set of requests.

    Args:
        measurements: Sequence of RequestMeasurement objects.
        total_duration_seconds: Wall-clock / benchmark duration in seconds.

    Returns:
        BenchmarkSummary containing request stats, throughputs, and
        latency distributions.
    """
    total_requests = len(measurements)
    successful = [m for m in measurements if m.success]
    failed_requests = total_requests - len(successful)

    total_input_tokens = sum(m.input_tokens for m in measurements)
    total_output_tokens = sum(m.output_tokens for m in successful)

    safe_duration = max(total_duration_seconds, 1e-9)
    request_throughput = len(successful) / safe_duration
    output_token_throughput = total_output_tokens / safe_duration

    ttft_values = [m.ttft_seconds for m in successful if m.ttft_seconds is not None]
    e2e_values = [m.e2e_latency_seconds for m in successful]
    tpot_values = [m.tpot_seconds for m in successful if m.tpot_seconds is not None]

    ttft_stats = calculate_metric_stats(ttft_values)
    e2e_stats = calculate_metric_stats(e2e_values)
    tpot_stats = calculate_metric_stats(tpot_values)

    errors = [
        m.error_message
        for m in measurements
        if not m.success and m.error_message is not None
    ]

    return BenchmarkSummary(
        total_requests=total_requests,
        successful_requests=len(successful),
        failed_requests=failed_requests,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
        total_duration_seconds=total_duration_seconds,
        request_throughput=request_throughput,
        output_token_throughput=output_token_throughput,
        ttft_stats=ttft_stats,
        e2e_latency_stats=e2e_stats,
        tpot_stats=tpot_stats,
        errors=errors if errors else None,
    )
