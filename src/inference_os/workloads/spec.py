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

    @property
    def normalized_weights(self) -> tuple[float, ...]:
        """Return weights normalized to probabilities that sum to one."""
        total = sum(self.weights)
        return tuple(weight / total for weight in self.weights)

    @property
    def mean(self) -> float:
        """Return the configured distribution mean."""
        return sum(
            value * probability
            for value, probability in zip(self.values, self.normalized_weights)
        )

    @property
    def std_dev(self) -> float:
        """Return the configured population standard deviation."""
        mean = self.mean
        variance = sum(
            probability * (value - mean) ** 2
            for value, probability in zip(self.values, self.normalized_weights)
        )
        return math.sqrt(max(0.0, variance))

    def sample_many(
        self,
        count: int,
        rng: random.Random,
        *,
        mode: str = "iid",
    ) -> list[int]:
        """Realize multiple values using IID or stratified sampling.

        Stratified sampling converts expected category counts using the largest
        remainder method, then shuffles them. It reduces finite-sample mean drift
        while preserving heterogeneous request order.
        """
        if count < 0:
            raise ValueError("sample count cannot be negative")
        if mode == "iid":
            return [self.sample(rng) for _ in range(count)]
        if mode == "sequence":
            return [self.values[index % len(self.values)] for index in range(count)]
        if mode != "stratified":
            raise ValueError("sampling mode must be 'iid', 'stratified', or 'sequence'")
        if count == 0:
            return []

        quotas = [count * probability for probability in self.normalized_weights]
        counts = [math.floor(quota) for quota in quotas]
        remaining = count - sum(counts)
        ranked = sorted(
            range(len(quotas)),
            key=lambda index: (-(quotas[index] - counts[index]), index),
        )
        for index in ranked[:remaining]:
            counts[index] += 1

        realized = [
            value
            for value, category_count in zip(self.values, counts)
            for _ in range(category_count)
        ]
        rng.shuffle(realized)
        return realized


@dataclass(frozen=True, slots=True)
class PromptReuseConfig:
    """Configuration for deterministic prompt identity and prefix reuse."""

    mode: str = "none"
    shared_prefix_tokens: int = 0
    reuse_group_id: str = "prefix-1"
    cache_block_size_tokens: int = 16
    application_profile: str | None = None
    prompt_layout: str = "default"
    relationship_group_count: int = 1
    stable_prefix_tokens: int = 0
    document_tokens: int = 0

    def __post_init__(self) -> None:
        valid_modes = {"none", "unique_prefix", "shared_prefix", "application"}
        if self.mode not in valid_modes:
            raise ValueError(f"prompt_reuse mode must be one of {sorted(valid_modes)}")
        if (
            isinstance(self.shared_prefix_tokens, bool)
            or not isinstance(self.shared_prefix_tokens, int)
            or self.shared_prefix_tokens < 0
        ):
            raise ValueError("shared_prefix_tokens must be a non-negative integer")
        if (
            isinstance(self.cache_block_size_tokens, bool)
            or not isinstance(self.cache_block_size_tokens, int)
            or self.cache_block_size_tokens <= 0
        ):
            raise ValueError("cache_block_size_tokens must be a positive integer")
        if self.mode == "shared_prefix":
            if self.shared_prefix_tokens <= 0:
                raise ValueError(
                    "shared_prefix mode requires positive shared_prefix_tokens"
                )
            if not self.reuse_group_id.strip():
                raise ValueError("shared_prefix mode requires a reuse_group_id")
        elif self.mode != "application" and self.shared_prefix_tokens != 0:
            raise ValueError(
                "shared_prefix_tokens is only valid for shared_prefix mode"
            )
        if self.mode == "application":
            valid_profiles = {
                "chat_like",
                "rag_like",
                "summarization_like",
                "agent_like",
            }
            if self.application_profile not in valid_profiles:
                raise ValueError(
                    f"application_profile must be one of {sorted(valid_profiles)}"
                )
            if self.prompt_layout not in {"default", "friendly", "hostile"}:
                raise ValueError(
                    "application prompt_layout must be default, friendly, or hostile"
                )
            if self.application_profile == "rag_like":
                if self.prompt_layout not in {"friendly", "hostile"}:
                    raise ValueError("rag_like requires friendly or hostile layout")
            elif self.prompt_layout != "default":
                raise ValueError(
                    "prompt_layout is only configurable for rag_like profiles"
                )
            if self.relationship_group_count <= 0:
                raise ValueError("relationship_group_count must be positive")
            if self.stable_prefix_tokens <= 0:
                raise ValueError("application mode requires stable_prefix_tokens")
            if self.document_tokens < 0:
                raise ValueError("document_tokens must be non-negative")
            if self.application_profile == "rag_like" and self.document_tokens <= 0:
                raise ValueError("rag_like requires positive document_tokens")
        elif self.application_profile is not None:
            raise ValueError("application_profile requires application mode")


@dataclass(frozen=True, slots=True)
class WorkloadConfig:
    """Configuration for one named synthetic workload profile."""

    name: str
    input_tokens: TokenLengthDistribution
    max_output_tokens: TokenLengthDistribution
    prompt_reuse: PromptReuseConfig = field(default_factory=PromptReuseConfig)
    sampling_mode: str = "iid"

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("workload name cannot be empty")
        if self.sampling_mode not in {"iid", "stratified", "sequence"}:
            raise ValueError(
                "workload sampling_mode must be 'iid', 'stratified', or 'sequence'"
            )

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
                mode=str(prompt_reuse_data.get("mode", "none")),
                shared_prefix_tokens=int(
                    prompt_reuse_data.get("shared_prefix_tokens", 0)
                ),
                reuse_group_id=str(prompt_reuse_data.get("reuse_group_id", "prefix-1")),
                cache_block_size_tokens=int(
                    prompt_reuse_data.get("cache_block_size_tokens", 16)
                ),
                application_profile=(
                    str(prompt_reuse_data["application_profile"])
                    if prompt_reuse_data.get("application_profile") is not None
                    else None
                ),
                prompt_layout=str(prompt_reuse_data.get("prompt_layout", "default")),
                relationship_group_count=int(
                    prompt_reuse_data.get("relationship_group_count", 1)
                ),
                stable_prefix_tokens=int(
                    prompt_reuse_data.get("stable_prefix_tokens", 0)
                ),
                document_tokens=int(prompt_reuse_data.get("document_tokens", 0)),
            ),
            sampling_mode=str(data.get("sampling_mode", "iid")),
        )


def generate_request_specs(
    *,
    num_requests: int,
    seed: int,
    input_tokens: TokenLengthDistribution,
    max_output_tokens: TokenLengthDistribution,
    sampling_mode: str = "iid",
) -> list[RequestSpec]:
    """Generate a deterministic, ordered request plan."""
    if num_requests < 0:
        raise ValueError("num_requests cannot be negative")

    rng = random.Random(seed)
    if sampling_mode == "iid":
        # Preserve the original interleaved draw order for existing V2 profiles.
        return [
            RequestSpec(
                target_input_tokens=input_tokens.sample(rng),
                max_output_tokens=max_output_tokens.sample(rng),
            )
            for _ in range(num_requests)
        ]

    input_plan = input_tokens.sample_many(
        num_requests,
        rng,
        mode=sampling_mode,
    )
    output_plan = max_output_tokens.sample_many(
        num_requests,
        rng,
        mode=sampling_mode,
    )
    return [
        RequestSpec(target_input_tokens=input_length, max_output_tokens=output_length)
        for input_length, output_length in zip(input_plan, output_plan)
    ]
