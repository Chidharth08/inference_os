"""Deterministic request specifications for synthetic workload profiles."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True, slots=True)
class RequestSpec:
    """Requested token shape for one planned inference request."""

    target_input_tokens: int
    max_output_tokens: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.target_input_tokens, bool)
            or not isinstance(self.target_input_tokens, int)
            or self.target_input_tokens <= 0
        ):
            raise ValueError("target_input_tokens must be positive")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class TokenLengthDistribution:
    """A validated finite weighted distribution of token lengths."""

    values: tuple[int, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("distribution values cannot be empty")
        if len(self.values) != len(self.weights):
            raise ValueError("distribution values and weights must have equal lengths")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.values
        ):
            raise ValueError("distribution values must be positive integers")
        if any(
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight < 0
            for weight in self.weights
        ):
            raise ValueError("distribution weights must be finite and non-negative")
        if sum(self.weights) <= 0:
            raise ValueError("distribution weights must have a positive sum")

    @classmethod
    def fixed(cls, value: int) -> TokenLengthDistribution:
        """Represent a fixed V1 token length as a one-point distribution."""
        return cls(values=(value,), weights=(1.0,))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenLengthDistribution:
        """Construct a distribution from a YAML/JSON mapping."""
        raw_values = data.get("values")
        raw_weights = data.get("weights")
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, str):
            raise ValueError("distribution values must be a list")
        if not isinstance(raw_weights, Sequence) or isinstance(raw_weights, str):
            raise ValueError("distribution weights must be a list")
        return cls(values=tuple(raw_values), weights=tuple(raw_weights))

    def sample(self, rng: random.Random) -> int:
        """Draw one value with a caller-owned deterministic random generator."""
        threshold = rng.random() * sum(self.weights)
        cumulative = 0.0
        for value, weight in zip(self.values, self.weights):
            cumulative += weight
            if threshold < cumulative:
                return value
        return self.values[-1]


@dataclass(frozen=True, slots=True)
class PromptReuseConfig:
    """Prompt-reuse metadata reserved for a future caching experiment."""

    mode: str = "none"

    def __post_init__(self) -> None:
        if self.mode != "none":
            raise ValueError("only prompt_reuse mode 'none' is supported in V2")


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """Configuration for one named synthetic workload profile."""

    name: str
    input_tokens: TokenLengthDistribution
    max_output_tokens: TokenLengthDistribution
    prompt_reuse: PromptReuseConfig = field(default_factory=PromptReuseConfig)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("workload name cannot be empty")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadConfig:
        """Construct a workload configuration from a YAML/JSON mapping."""
        input_data = data.get("input_tokens")
        output_data = data.get("max_output_tokens")
        prompt_reuse_data = data.get("prompt_reuse", {"mode": "none"})
        if not isinstance(input_data, dict):
            raise ValueError("workload.input_tokens must be a mapping")
        if not isinstance(output_data, dict):
            raise ValueError("workload.max_output_tokens must be a mapping")
        if not isinstance(prompt_reuse_data, dict):
            raise ValueError("workload.prompt_reuse must be a mapping")
        return cls(
            name=str(data.get("name", "")).strip(),
            input_tokens=TokenLengthDistribution.from_dict(input_data),
            max_output_tokens=TokenLengthDistribution.from_dict(output_data),
            prompt_reuse=PromptReuseConfig(
                mode=str(prompt_reuse_data.get("mode", "none"))
            ),
        )


def generate_request_specs(
    *,
    num_requests: int,
    seed: int,
    input_tokens: TokenLengthDistribution,
    max_output_tokens: TokenLengthDistribution,
) -> list[RequestSpec]:
    """Generate a deterministic, ordered request plan."""
    if num_requests < 0:
        raise ValueError("num_requests cannot be negative")

    rng = random.Random(seed)
    return [
        RequestSpec(
            target_input_tokens=input_tokens.sample(rng),
            max_output_tokens=max_output_tokens.sample(rng),
        )
        for _ in range(num_requests)
    ]
