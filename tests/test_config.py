"""Unit tests for BenchmarkConfig data model and validation."""

import pytest

from inference_os import BenchmarkConfig


def test_benchmark_config_defaults() -> None:
    """Verify default configuration attributes."""
    cfg = BenchmarkConfig(model="Qwen/Qwen2.5-7B-Instruct")
    assert cfg.model == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.base_url == "http://localhost:8000"
    assert cfg.prompt_tokens == 128
    assert cfg.max_output_tokens == 64
    assert cfg.num_requests == 10
    assert cfg.warmup_requests == 2
    assert cfg.temperature == 0.0
    assert cfg.telemetry_interval_seconds == 0.1
    assert cfg.seed == 42


def test_benchmark_config_validation_errors() -> None:
    """Verify ValueError on invalid configuration parameters."""
    with pytest.raises(ValueError, match="model cannot be empty"):
        BenchmarkConfig(model="")

    with pytest.raises(ValueError, match="model cannot be empty"):
        BenchmarkConfig(model="   ")

    with pytest.raises(ValueError, match="prompt_tokens must be positive"):
        BenchmarkConfig(model="test", prompt_tokens=0)

    with pytest.raises(ValueError, match="max_output_tokens must be positive"):
        BenchmarkConfig(model="test", max_output_tokens=-5)

    with pytest.raises(ValueError, match="num_requests must be positive"):
        BenchmarkConfig(model="test", num_requests=0)

    with pytest.raises(ValueError, match="warmup_requests cannot be negative"):
        BenchmarkConfig(model="test", warmup_requests=-1)

    with pytest.raises(ValueError, match="telemetry_interval_seconds must be positive"):
        BenchmarkConfig(model="test", telemetry_interval_seconds=0.0)

    with pytest.raises(ValueError, match="temperature cannot be negative"):
        BenchmarkConfig(model="test", temperature=-0.5)


def test_benchmark_config_json_roundtrip() -> None:
    """Verify JSON serialization and deserialization."""
    cfg = BenchmarkConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt_tokens=512,
        max_output_tokens=128,
        num_requests=20,
        warmup_requests=3,
        temperature=0.7,
        seed=999,
        experiment_id="E001",
    )

    json_str = cfg.to_json()
    loaded = BenchmarkConfig.from_json(json_str)

    assert loaded == cfg
