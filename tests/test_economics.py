"""Tests for benchmark-normalized serving economics metrics."""

import pytest

from inference_os.metrics.economics import CostAssumptions, compute_cost_metrics


def test_compute_cost_metrics_applies_billing_and_gpu_count() -> None:
    assumptions = CostAssumptions(
        currency="USD",
        gpu_price_per_hour=0.36,
        gpu_count=2,
        billing_increment_seconds=1.0,
    )

    result = compute_cost_metrics(
        measured_duration_seconds=10.2,
        completed_requests=10,
        slo_compliant_requests=8,
        input_tokens=1000,
        output_tokens=500,
        assumptions=assumptions,
    )

    assert result.billable_duration_seconds == 11.0
    assert result.measured_gpu_seconds == pytest.approx(20.4)
    assert result.billed_gpu_seconds == 22.0
    assert result.estimated_cost == pytest.approx(0.0022)
    assert result.gpu_seconds_per_completed_request == pytest.approx(2.04)
    assert result.cost_per_1000_completed_requests == pytest.approx(0.22)
    assert result.cost_per_1000_slo_compliant_requests == pytest.approx(0.275)
    assert result.cost_per_million_output_tokens == pytest.approx(4.4)
    assert result.cost_per_million_total_tokens == pytest.approx(1.4666666667)


def test_zero_denominators_are_none_not_zero_cost() -> None:
    result = compute_cost_metrics(
        measured_duration_seconds=5.0,
        completed_requests=0,
        slo_compliant_requests=0,
        input_tokens=0,
        output_tokens=0,
        assumptions=CostAssumptions("USD", 0.19),
    )

    assert result.estimated_cost > 0
    assert result.cost_per_completed_request is None
    assert result.cost_per_slo_compliant_request is None
    assert result.cost_per_million_output_tokens is None


def test_cost_metrics_reject_invalid_counts() -> None:
    assumptions = CostAssumptions("USD", 0.19)
    with pytest.raises(ValueError, match="cannot exceed"):
        compute_cost_metrics(
            measured_duration_seconds=1.0,
            completed_requests=1,
            slo_compliant_requests=2,
            input_tokens=1,
            output_tokens=1,
            assumptions=assumptions,
        )

    with pytest.raises(ValueError, match="gpu_price"):
        CostAssumptions("USD", -0.01)
