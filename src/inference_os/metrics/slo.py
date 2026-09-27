"""Service-level-objective evaluation for open-loop load tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from inference_os.config import SLOConfig
from inference_os.metrics.request import RequestMeasurement
from inference_os.metrics.summary import _calculate_percentile


@dataclass(frozen=True, slots=True)
class SLOSummary:
    offered_requests: int
    compliant_requests: int
    compliance_rate: float
    goodput: float
    ttft_percentile_seconds: float | None
    e2e_percentile_seconds: float | None
    error_rate: float
    ttft_objective_met: bool
    e2e_objective_met: bool
    error_objective_met: bool
    all_objectives_met: bool


def request_meets_slo(measurement: RequestMeasurement, slo: SLOConfig) -> bool:
    """Return whether one offered request completed within both latency limits."""
    return bool(
        measurement.success
        and measurement.ttft_seconds is not None
        and measurement.ttft_seconds <= slo.max_ttft_seconds
        and measurement.e2e_latency_seconds <= slo.max_e2e_latency_seconds
    )


def evaluate_slo(
    measurements: Sequence[RequestMeasurement],
    slo: SLOConfig,
    measured_duration_seconds: float,
) -> SLOSummary:
    """Evaluate per-request goodput and aggregate percentile objectives."""
    total = len(measurements)
    successful = [measurement for measurement in measurements if measurement.success]
    ttfts = sorted(
        measurement.ttft_seconds
        for measurement in successful
        if measurement.ttft_seconds is not None
    )
    e2es = sorted(measurement.e2e_latency_seconds for measurement in successful)
    ttft_percentile = _calculate_percentile(ttfts, slo.percentile) if ttfts else None
    e2e_percentile = _calculate_percentile(e2es, slo.percentile) if e2es else None
    compliant = sum(request_meets_slo(measurement, slo) for measurement in measurements)
    error_rate = (total - len(successful)) / total if total else 0.0
    ttft_met = ttft_percentile is not None and ttft_percentile <= slo.max_ttft_seconds
    e2e_met = (
        e2e_percentile is not None and e2e_percentile <= slo.max_e2e_latency_seconds
    )
    error_met = error_rate <= slo.max_error_rate
    return SLOSummary(
        offered_requests=total,
        compliant_requests=compliant,
        compliance_rate=compliant / total if total else 0.0,
        goodput=compliant / max(measured_duration_seconds, 1e-9),
        ttft_percentile_seconds=ttft_percentile,
        e2e_percentile_seconds=e2e_percentile,
        error_rate=error_rate,
        ttft_objective_met=ttft_met,
        e2e_objective_met=e2e_met,
        error_objective_met=error_met,
        all_objectives_met=ttft_met and e2e_met and error_met,
    )
