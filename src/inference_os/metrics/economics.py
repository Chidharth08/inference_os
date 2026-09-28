"""Transparent, benchmark-normalized serving economics metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CostAssumptions:
    """Explicit pricing and measurement-boundary assumptions."""

    currency: str
    gpu_price_per_hour: float
    gpu_count: int = 1
    billing_increment_seconds: float = 1.0
    idle_time_included: bool = True
    actual_session_cost: float | None = None
    actual_session_cost_is_approximate: bool = True

    def __post_init__(self) -> None:
        if not self.currency or not self.currency.strip():
            raise ValueError("currency cannot be empty")
        if self.gpu_price_per_hour < 0:
            raise ValueError("gpu_price_per_hour cannot be negative")
        if self.gpu_count <= 0:
            raise ValueError("gpu_count must be positive")
        if self.billing_increment_seconds <= 0:
            raise ValueError("billing_increment_seconds must be positive")
        if self.actual_session_cost is not None and self.actual_session_cost < 0:
            raise ValueError("actual_session_cost cannot be negative")

    @property
    def gpu_price_per_second(self) -> float:
        return self.gpu_price_per_hour / 3600.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostAssumptions:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EconomicsConfig:
    """Offline E006 analysis configuration."""

    source_summary: str
    output_dir: str
    assumptions: CostAssumptions
    experiment_id: str = "E006"
    measurement_scope: str = (
        "per-point measured wall time, including arrival gaps and backlog drain; "
        "excluding warm-up, setup, model loading, and time between points"
    )

    def __post_init__(self) -> None:
        if not self.source_summary or not self.source_summary.strip():
            raise ValueError("source_summary cannot be empty")
        if not self.output_dir or not self.output_dir.strip():
            raise ValueError("output_dir cannot be empty")
        if not self.measurement_scope or not self.measurement_scope.strip():
            raise ValueError("measurement_scope cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["assumptions"] = self.assumptions.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EconomicsConfig:
        raw = dict(data)
        assumptions = raw.get("assumptions")
        if not isinstance(assumptions, dict):
            raise ValueError("assumptions must be a mapping")
        raw["assumptions"] = CostAssumptions.from_dict(assumptions)
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class CostMetrics:
    """Cost metrics for one measured load point."""

    measured_duration_seconds: float
    billable_duration_seconds: float
    measured_gpu_seconds: float
    billed_gpu_seconds: float
    completed_requests: int
    slo_compliant_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost: float
    gpu_seconds_per_completed_request: float | None
    gpu_seconds_per_slo_compliant_request: float | None
    cost_per_completed_request: float | None
    cost_per_1000_completed_requests: float | None
    cost_per_slo_compliant_request: float | None
    cost_per_1000_slo_compliant_requests: float | None
    cost_per_million_input_tokens: float | None
    cost_per_million_output_tokens: float | None
    cost_per_million_total_tokens: float | None


def compute_cost_metrics(
    *,
    measured_duration_seconds: float,
    completed_requests: int,
    slo_compliant_requests: int,
    input_tokens: int,
    output_tokens: int,
    assumptions: CostAssumptions,
) -> CostMetrics:
    """Calculate normalized costs using one explicit measured time window."""
    if measured_duration_seconds < 0:
        raise ValueError("measured_duration_seconds cannot be negative")
    for name, value in (
        ("completed_requests", completed_requests),
        ("slo_compliant_requests", slo_compliant_requests),
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
    ):
        if value < 0:
            raise ValueError(f"{name} cannot be negative")
    if slo_compliant_requests > completed_requests:
        raise ValueError("slo_compliant_requests cannot exceed completed_requests")

    increment = assumptions.billing_increment_seconds
    billable_duration = (
        math.ceil(measured_duration_seconds / increment) * increment
        if measured_duration_seconds > 0
        else 0.0
    )
    measured_gpu_seconds = measured_duration_seconds * assumptions.gpu_count
    billed_gpu_seconds = billable_duration * assumptions.gpu_count
    estimated_cost = billed_gpu_seconds * assumptions.gpu_price_per_second
    total_tokens = input_tokens + output_tokens

    def divide(numerator: float, denominator: int) -> float | None:
        return numerator / denominator if denominator > 0 else None

    cost_per_completed = divide(estimated_cost, completed_requests)
    cost_per_compliant = divide(estimated_cost, slo_compliant_requests)
    return CostMetrics(
        measured_duration_seconds=measured_duration_seconds,
        billable_duration_seconds=billable_duration,
        measured_gpu_seconds=measured_gpu_seconds,
        billed_gpu_seconds=billed_gpu_seconds,
        completed_requests=completed_requests,
        slo_compliant_requests=slo_compliant_requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        estimated_cost=estimated_cost,
        gpu_seconds_per_completed_request=divide(
            measured_gpu_seconds, completed_requests
        ),
        gpu_seconds_per_slo_compliant_request=divide(
            measured_gpu_seconds, slo_compliant_requests
        ),
        cost_per_completed_request=cost_per_completed,
        cost_per_1000_completed_requests=(
            cost_per_completed * 1000.0 if cost_per_completed is not None else None
        ),
        cost_per_slo_compliant_request=cost_per_compliant,
        cost_per_1000_slo_compliant_requests=(
            cost_per_compliant * 1000.0 if cost_per_compliant is not None else None
        ),
        cost_per_million_input_tokens=(
            divide(estimated_cost, input_tokens) * 1_000_000.0
            if input_tokens > 0
            else None
        ),
        cost_per_million_output_tokens=(
            divide(estimated_cost, output_tokens) * 1_000_000.0
            if output_tokens > 0
            else None
        ),
        cost_per_million_total_tokens=(
            divide(estimated_cost, total_tokens) * 1_000_000.0
            if total_tokens > 0
            else None
        ),
    )
