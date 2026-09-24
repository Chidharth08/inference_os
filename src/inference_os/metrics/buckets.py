"""Per-length workload bucket summaries for heterogeneous benchmark runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Any, Sequence

from inference_os.metrics.summary import calculate_metric_stats

_SUPPORTED_BUCKET_FIELDS = {"target_input_tokens", "max_output_tokens"}
_NANOSECONDS_PER_SECOND = 1_000_000_000.0


def summarize_length_buckets(
    request_records: Sequence[dict[str, Any]],
    workload_records: Sequence[dict[str, Any]],
    *,
    bucket_field: str,
) -> list[dict[str, Any]]:
    """Join persisted requests to their plan and summarize exact length buckets."""
    if bucket_field not in _SUPPORTED_BUCKET_FIELDS:
        raise ValueError(
            "bucket_field must be 'target_input_tokens' or 'max_output_tokens'"
        )

    measured_requests = {
        str(record["request_id"]): record
        for record in request_records
        if not record.get("is_warmup", False)
    }
    measured_workload = {
        str(record["request_id"]): record
        for record in workload_records
        if not record.get("is_warmup", False)
    }
    if measured_requests.keys() != measured_workload.keys():
        raise ValueError("measured request and workload records do not match")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for request_id, workload in measured_workload.items():
        token_length = workload.get(bucket_field)
        if isinstance(token_length, bool) or not isinstance(token_length, int):
            raise ValueError(f"{bucket_field} must contain integer token lengths")
        grouped[token_length].append(measured_requests[request_id])

    summaries: list[dict[str, Any]] = []
    for token_length in sorted(grouped):
        requests = grouped[token_length]
        successful = [request for request in requests if request.get("success")]

        def stats(values: list[float]) -> dict[str, Any] | None:
            result = calculate_metric_stats(values)
            return asdict(result) if result is not None else None

        ttft_values = [
            float(request["ttft_seconds"])
            for request in successful
            if request.get("ttft_seconds") is not None
        ]
        e2e_values = [float(request["e2e_latency_seconds"]) for request in successful]
        tpot_values: list[float] = []
        for request in successful:
            first_token_ns = request.get("first_token_time_ns")
            output_tokens = int(request.get("output_tokens", 0))
            if first_token_ns is None or output_tokens <= 1:
                continue
            decode_seconds = (
                int(request["completion_time_ns"]) - int(first_token_ns)
            ) / _NANOSECONDS_PER_SECOND
            tpot_values.append(decode_seconds / (output_tokens - 1))

        summaries.append(
            {
                "token_length": token_length,
                "request_count": len(requests),
                "successful_requests": len(successful),
                "failed_requests": len(requests) - len(successful),
                "actual_input_tokens": stats(
                    [float(request["input_tokens"]) for request in successful]
                ),
                "actual_output_tokens": stats(
                    [float(request["output_tokens"]) for request in successful]
                ),
                "ttft_stats": stats(ttft_values),
                "tpot_stats": stats(tpot_values),
                "e2e_latency_stats": stats(e2e_values),
            }
        )
    return summaries
