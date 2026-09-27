"""Tests for E005 SLO and goodput semantics."""

import pytest

from inference_os.config import SLOConfig
from inference_os.metrics.request import RequestMeasurement
from inference_os.metrics.slo import evaluate_slo, request_meets_slo


def measurement(
    request_id: str, ttft: float, e2e: float, *, success: bool = True
) -> RequestMeasurement:
    start = 1_000_000_000
    return RequestMeasurement(
        request_id=request_id,
        start_time_ns=start,
        first_token_time_ns=start + round(ttft * 1e9) if success else None,
        completion_time_ns=start + round(e2e * 1e9),
        input_tokens=10,
        output_tokens=2 if success else 0,
        success=success,
        error_message=None if success else "failed",
    )


def test_goodput_counts_only_per_request_slo_completions() -> None:
    slo = SLOConfig(
        max_ttft_seconds=1.0,
        max_e2e_latency_seconds=5.0,
        max_error_rate=0.5,
        percentile=95.0,
    )
    records = [
        measurement("fast", 0.5, 2.0),
        measurement("late", 1.5, 4.0),
        measurement("failed", 0.0, 0.0, success=False),
    ]

    summary = evaluate_slo(records, slo, measured_duration_seconds=2.0)

    assert request_meets_slo(records[0], slo)
    assert not request_meets_slo(records[1], slo)
    assert summary.compliant_requests == 1
    assert summary.compliance_rate == pytest.approx(1 / 3)
    assert summary.goodput == pytest.approx(0.5)
    assert not summary.ttft_objective_met
    assert summary.e2e_objective_met
    assert summary.error_objective_met
    assert not summary.all_objectives_met
