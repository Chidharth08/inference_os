"""Unit tests for RequestMeasurement dataclass and derived metrics."""

import pytest

from inference_os.metrics import RequestMeasurement


def test_valid_request_measurement_metrics() -> None:
    """Verify TTFT and E2E latency calculations for a valid request."""
    # start = 1.0s, first_token = 1.2s, completion = 2.5s (in ns)
    req = RequestMeasurement(
        request_id="req-1",
        start_time_ns=1_000_000_000,
        first_token_time_ns=1_200_000_000,
        completion_time_ns=2_500_000_000,
        input_tokens=128,
        output_tokens=32,
        success=True,
    )

    assert req.ttft_seconds == pytest.approx(0.2)
    assert req.e2e_latency_seconds == pytest.approx(1.5)


def test_absent_first_token_timestamp() -> None:
    """Verify behavior when no first token timestamp exists (e.g. failed prefill)."""
    req = RequestMeasurement(
        request_id="req-2",
        start_time_ns=1_000_000_000,
        completion_time_ns=1_500_000_000,
        input_tokens=128,
        output_tokens=0,
        success=False,
        error_message="Connection timed out during prefill",
    )

    assert req.ttft_seconds is None
    assert req.e2e_latency_seconds == pytest.approx(0.5)


def test_zero_token_output() -> None:
    """Verify measurement with zero output tokens and no first token timestamp."""
    req = RequestMeasurement(
        request_id="req-3",
        start_time_ns=500_000_000,
        completion_time_ns=600_000_000,
        input_tokens=64,
        output_tokens=0,
        success=True,
    )

    assert req.ttft_seconds is None
    assert req.e2e_latency_seconds == pytest.approx(0.1)


def test_invalid_completion_before_start() -> None:
    """Verify ValueError when completion timestamp precedes start timestamp."""
    with pytest.raises(ValueError, match="completion_time_ns.*cannot precede"):
        RequestMeasurement(
            request_id="req-err-1",
            start_time_ns=2_000_000_000,
            completion_time_ns=1_000_000_000,
            input_tokens=10,
            output_tokens=5,
            success=False,
        )


def test_invalid_first_token_before_start() -> None:
    """Verify ValueError when first token timestamp precedes start timestamp."""
    with pytest.raises(ValueError, match="first_token_time_ns.*cannot precede"):
        RequestMeasurement(
            request_id="req-err-2",
            start_time_ns=2_000_000_000,
            first_token_time_ns=1_500_000_000,
            completion_time_ns=3_000_000_000,
            input_tokens=10,
            output_tokens=5,
            success=True,
        )


def test_invalid_first_token_after_completion() -> None:
    """Verify ValueError when first token timestamp is after completion timestamp."""
    with pytest.raises(ValueError, match="first_token_time_ns.*cannot be after"):
        RequestMeasurement(
            request_id="req-err-3",
            start_time_ns=1_000_000_000,
            first_token_time_ns=4_000_000_000,
            completion_time_ns=3_000_000_000,
            input_tokens=10,
            output_tokens=5,
            success=True,
        )


def test_invalid_negative_timestamps_or_counts() -> None:
    """Verify ValueError on negative timestamps or token counts."""
    with pytest.raises(ValueError, match="start_time_ns must be non-negative"):
        RequestMeasurement(
            request_id="req-err-4",
            start_time_ns=-1,
            completion_time_ns=100,
            input_tokens=10,
            output_tokens=5,
            success=True,
        )

    with pytest.raises(ValueError, match="input_tokens must be non-negative"):
        RequestMeasurement(
            request_id="req-err-5",
            start_time_ns=100,
            completion_time_ns=200,
            input_tokens=-10,
            output_tokens=5,
            success=True,
        )

    with pytest.raises(ValueError, match="output_tokens must be non-negative"):
        RequestMeasurement(
            request_id="req-err-6",
            start_time_ns=100,
            completion_time_ns=200,
            input_tokens=10,
            output_tokens=-5,
            success=True,
        )
