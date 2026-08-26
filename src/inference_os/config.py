"""Experiment configuration models and validation."""

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration specification for a benchmark experiment run."""

    model: str
    base_url: str = "http://localhost:8000"
    prompt_tokens: int = 128
    max_output_tokens: int = 64
    num_requests: int = 10
    warmup_requests: int = 2
    seed: int = 42
    temperature: float = 0.0
    telemetry_interval_seconds: float = 0.1
    device_index: int = 0
    experiment_id: str = "E000"
    output_dir: str = "runs"

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if not self.model or not self.model.strip():
            raise ValueError("model cannot be empty")
        if self.prompt_tokens <= 0:
            raise ValueError(
                f"prompt_tokens must be positive, got {self.prompt_tokens}"
            )
        if self.max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be positive, got {self.max_output_tokens}"
            )
        if self.num_requests <= 0:
            raise ValueError(f"num_requests must be positive, got {self.num_requests}")
        if self.warmup_requests < 0:
            raise ValueError(
                f"warmup_requests cannot be negative, got {self.warmup_requests}"
            )
        if self.telemetry_interval_seconds <= 0:
            raise ValueError(
                "telemetry_interval_seconds must be positive, got "
                f"{self.telemetry_interval_seconds}"
            )
        if self.temperature < 0.0:
            raise ValueError(f"temperature cannot be negative, got {self.temperature}")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Construct a BenchmarkConfig instance from a dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "BenchmarkConfig":
        """Construct a BenchmarkConfig instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
