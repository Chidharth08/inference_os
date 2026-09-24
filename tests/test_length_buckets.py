"""Tests for heterogeneous workload length-bucket summaries."""

import pytest

from inference_os.metrics import summarize_length_buckets


def _request(
    request_id: str,
    *,
    input_tokens: int,
    output_tokens: int,
    ttft_seconds: float,
    e2e_seconds: float,
) -> dict[str, object]:
    start = 1_000_000_000
    first = start + int(ttft_seconds * 1e9)
    completion = start + int(e2e_seconds * 1e9)
    return {
        "request_id": request_id,
        "is_warmup": False,
        "start_time_ns": start,
        "first_token_time_ns": first,
        "completion_time_ns": completion,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "success": True,
        "ttft_seconds": ttft_seconds,
        "e2e_latency_seconds": e2e_seconds,
    }


def test_summarize_length_buckets_joins_and_groups_records() -> None:
    requests = [
        _request(
            "req-1",
            input_tokens=128,
            output_tokens=4,
            ttft_seconds=0.1,
            e2e_seconds=1.0,
        ),
        _request(
            "req-2",
            input_tokens=512,
            output_tokens=8,
            ttft_seconds=0.4,
            e2e_seconds=2.0,
        ),
        _request(
            "req-3",
            input_tokens=128,
            output_tokens=4,
            ttft_seconds=0.2,
            e2e_seconds=1.2,
        ),
    ]
    workload = [
        {
            "request_id": "req-1",
            "is_warmup": False,
            "target_input_tokens": 128,
            "max_output_tokens": 4,
        },
        {
            "request_id": "req-2",
            "is_warmup": False,
            "target_input_tokens": 512,
            "max_output_tokens": 8,
        },
        {
            "request_id": "req-3",
            "is_warmup": False,
            "target_input_tokens": 128,
            "max_output_tokens": 4,
        },
    ]

    buckets = summarize_length_buckets(
        requests,
        workload,
        bucket_field="target_input_tokens",
    )

    assert [bucket["token_length"] for bucket in buckets] == [128, 512]
    assert buckets[0]["request_count"] == 2
    assert buckets[0]["ttft_stats"]["p50"] == pytest.approx(0.15)
    assert buckets[0]["e2e_latency_stats"]["p95"] == pytest.approx(1.19)
    assert buckets[1]["actual_output_tokens"]["mean"] == 8
    assert buckets[1]["tpot_stats"] is not None


def test_summarize_length_buckets_rejects_mismatched_records() -> None:
    with pytest.raises(ValueError, match="do not match"):
        summarize_length_buckets(
            [
                _request(
                    "req-1",
                    input_tokens=8,
                    output_tokens=2,
                    ttft_seconds=0.1,
                    e2e_seconds=0.2,
                )
            ],
            [],
            bucket_field="target_input_tokens",
        )


def test_summarize_length_buckets_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="bucket_field"):
        summarize_length_buckets([], [], bucket_field="unknown")
