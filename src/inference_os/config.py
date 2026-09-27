"""Experiment configuration models and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from inference_os.workloads.spec import WorkloadConfig


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Configuration specification for a single benchmark experiment run."""

    model: str
    base_url: str = "http://localhost:8000"
    prompt_tokens: int = 128
    max_output_tokens: int = 64
    num_requests: int = 10
    warmup_requests: int = 2
    seed: int = 42
    temperature: float = 0.0
    request_timeout_seconds: float = 300.0
    telemetry_interval_seconds: float = 0.1
    device_index: int = 0
    experiment_id: str = "E000"
    output_dir: str = "runs"
    enable_prefix_caching: bool = False
    chunked_prefill: Optional[int] = None
    enable_chunked_prefill: bool = False
    concurrency: int = 1
    workload: Optional[WorkloadConfig] = None

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
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {self.concurrency}")
        if self.telemetry_interval_seconds <= 0:
            raise ValueError(
                "telemetry_interval_seconds must be positive, got "
                f"{self.telemetry_interval_seconds}"
            )
        if self.temperature < 0.0:
            raise ValueError(f"temperature cannot be negative, got {self.temperature}")
        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be positive, got "
                f"{self.request_timeout_seconds}"
            )
        if self.chunked_prefill is not None and self.chunked_prefill <= 0:
            raise ValueError(
                f"chunked_prefill must be positive if set, got {self.chunked_prefill}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialize configuration to a YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Construct a BenchmarkConfig instance from a dictionary."""
        known_keys = {
            "model",
            "base_url",
            "prompt_tokens",
            "max_output_tokens",
            "num_requests",
            "warmup_requests",
            "seed",
            "temperature",
            "request_timeout_seconds",
            "telemetry_interval_seconds",
            "device_index",
            "experiment_id",
            "output_dir",
            "enable_prefix_caching",
            "chunked_prefill",
            "enable_chunked_prefill",
            "concurrency",
            "workload",
        }
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        workload_data = filtered_data.get("workload")
        if isinstance(workload_data, dict):
            filtered_data["workload"] = WorkloadConfig.from_dict(workload_data)
        elif workload_data is not None and not isinstance(
            workload_data, WorkloadConfig
        ):
            raise ValueError("workload must be a mapping")
        return cls(**filtered_data)

    @classmethod
    def from_json(cls, json_str: str) -> BenchmarkConfig:
        """Construct a BenchmarkConfig instance from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> BenchmarkConfig:
        """Construct a BenchmarkConfig instance from a YAML string."""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a key-value mapping")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """1D parameter sweep configuration for controlled scaling experiments."""

    sweep_param: str
    sweep_values: tuple[int, ...]
    base_config: BenchmarkConfig
    experiment_id: str = "E001A"

    def __post_init__(self) -> None:
        """Validate sweep invariants."""
        if not self.sweep_param or not self.sweep_param.strip():
            raise ValueError("sweep_param cannot be empty")
        valid_params = {
            "prompt_tokens",
            "max_output_tokens",
            "num_requests",
            "concurrency",
        }
        if self.sweep_param not in valid_params:
            options_str = sorted(valid_params)
            raise ValueError(
                f"sweep_param must be one of {options_str}, got '{self.sweep_param}'"
            )
        if not self.sweep_values:
            raise ValueError("sweep_values cannot be empty")
        for val in self.sweep_values:
            if val <= 0:
                raise ValueError(f"sweep_values entries must be positive, got {val}")

    def generate_point_configs(self) -> list[tuple[int, BenchmarkConfig]]:
        """Generate a list of (sweep_value, point_config) pairs."""
        point_configs: list[tuple[int, BenchmarkConfig]] = []
        for val in self.sweep_values:
            kwargs = {
                self.sweep_param: val,
                "experiment_id": self.experiment_id,
            }
            cfg = replace(self.base_config, **kwargs)
            point_configs.append((val, cfg))
        return point_configs

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary representation."""
        data = self.base_config.to_dict()
        data["experiment_id"] = self.experiment_id
        data["sweep_param"] = self.sweep_param
        data["sweep_values"] = list(self.sweep_values)
        return data

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SweepConfig:
        """Construct SweepConfig from a dictionary (flat or nested)."""
        data = dict(data)
        sweep_param = data.pop("sweep_param", "prompt_tokens")
        raw_values = data.pop("sweep_values", [128, 512, 2048, 4096])
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, str):
            sweep_values = tuple(int(v) for v in raw_values)
        else:
            raise ValueError("sweep_values must be a list of integers")

        experiment_id = data.get("experiment_id", "E001A")

        if "base_config" in data and isinstance(data["base_config"], dict):
            base_config = BenchmarkConfig.from_dict(data["base_config"])
        else:
            base_config = BenchmarkConfig.from_dict(data)

        return cls(
            sweep_param=sweep_param,
            sweep_values=sweep_values,
            base_config=base_config,
            experiment_id=experiment_id,
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> SweepConfig:
        """Construct SweepConfig from a YAML string."""
        data = yaml.safe_load(yaml_str)
        if not isinstance(data, dict):
            raise ValueError("YAML content must be a key-value mapping")
        return cls.from_dict(data)


@dataclass(frozen=True, slots=True)
class SLOConfig:
    """Latency and reliability objectives evaluated at each load point."""

    max_ttft_seconds: float = 1.0
    max_e2e_latency_seconds: float = 5.0
    max_error_rate: float = 0.01
    percentile: float = 95.0

    def __post_init__(self) -> None:
        if self.max_ttft_seconds <= 0:
            raise ValueError("max_ttft_seconds must be positive")
        if self.max_e2e_latency_seconds <= 0:
            raise ValueError("max_e2e_latency_seconds must be positive")
        if not 0.0 <= self.max_error_rate <= 1.0:
            raise ValueError("max_error_rate must be between 0 and 1")
        if not 0.0 < self.percentile <= 100.0:
            raise ValueError("percentile must be in (0, 100]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SLOConfig:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class OpenLoopSweepConfig:
    """Duration-based constant-rate load sweep configuration."""

    request_rates: tuple[float, ...]
    max_in_flight: int
    max_drain_seconds: float
    base_config: BenchmarkConfig
    slo: SLOConfig
    duration_seconds: Optional[float] = None
    requests_per_rate: Optional[int] = None
    experiment_id: str = "E005"

    def __post_init__(self) -> None:
        if not self.request_rates:
            raise ValueError("request_rates cannot be empty")
        if any(rate <= 0 for rate in self.request_rates):
            raise ValueError("request_rates entries must be positive")
        if len(set(self.request_rates)) != len(self.request_rates):
            raise ValueError("request_rates entries must be unique")
        if (self.duration_seconds is None) == (self.requests_per_rate is None):
            raise ValueError("set exactly one of duration_seconds or requests_per_rate")
        if self.duration_seconds is not None and self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if self.requests_per_rate is not None and self.requests_per_rate <= 0:
            raise ValueError("requests_per_rate must be positive")
        if self.max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        if self.max_drain_seconds <= 0:
            raise ValueError("max_drain_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        data = self.base_config.to_dict()
        data.update(
            {
                "experiment_id": self.experiment_id,
                "request_rates": list(self.request_rates),
                "duration_seconds": self.duration_seconds,
                "requests_per_rate": self.requests_per_rate,
                "max_in_flight": self.max_in_flight,
                "max_drain_seconds": self.max_drain_seconds,
                "slo": asdict(self.slo),
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OpenLoopSweepConfig:
        data = dict(data)
        raw_rates = data.pop("request_rates", None)
        if not isinstance(raw_rates, Sequence) or isinstance(raw_rates, str):
            raise ValueError("request_rates must be a list of numbers")
        raw_duration = data.pop("duration_seconds", None)
        duration_seconds = float(raw_duration) if raw_duration is not None else None
        raw_request_count = data.pop("requests_per_rate", None)
        requests_per_rate = (
            int(raw_request_count) if raw_request_count is not None else None
        )
        max_in_flight = int(data.pop("max_in_flight", 128))
        max_drain_seconds = float(data.pop("max_drain_seconds", 300.0))
        slo_data = data.pop("slo", {})
        if not isinstance(slo_data, dict):
            raise ValueError("slo must be a mapping")
        experiment_id = str(data.get("experiment_id", "E005"))
        base_config = BenchmarkConfig.from_dict(data)
        return cls(
            request_rates=tuple(float(rate) for rate in raw_rates),
            duration_seconds=duration_seconds,
            requests_per_rate=requests_per_rate,
            max_in_flight=max_in_flight,
            max_drain_seconds=max_drain_seconds,
            base_config=base_config,
            slo=SLOConfig.from_dict(slo_data),
            experiment_id=experiment_id,
        )

    def duration_for_rate(self, request_rate: float) -> float:
        """Return the configured or request-count-derived arrival duration."""
        if request_rate <= 0:
            raise ValueError("request_rate must be positive")
        if self.requests_per_rate is not None:
            return self.requests_per_rate / request_rate
        assert self.duration_seconds is not None
        return self.duration_seconds


def load_config(
    file_path: Path | str,
) -> BenchmarkConfig | SweepConfig | OpenLoopSweepConfig:
    """Load configuration from a YAML or JSON file.

    Automatically discriminates between single-run BenchmarkConfig and SweepConfig.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(content)
    elif path.suffix.lower() == ".json":
        data = json.loads(content)
    else:
        # Attempt YAML parser first as superset
        try:
            data = yaml.safe_load(content)
        except Exception:
            data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} did not contain a valid mapping")

    if "request_rates" in data:
        return OpenLoopSweepConfig.from_dict(data)
    if "sweep_param" in data or "sweep_values" in data:
        return SweepConfig.from_dict(data)
    return BenchmarkConfig.from_dict(data)
