"""Unit tests for BenchmarkConfig data model and validation."""

import pytest

from inference_os import BenchmarkConfig, SweepConfig


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


def test_benchmark_config_yaml_roundtrip(tmp_path) -> None:
    """Verify YAML serialization, deserialization, and load_config."""
    from inference_os.config import load_config

    cfg = BenchmarkConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt_tokens=512,
        max_output_tokens=128,
        enable_prefix_caching=False,
        chunked_prefill=512,
    )

    yaml_str = cfg.to_yaml()
    loaded = BenchmarkConfig.from_yaml(yaml_str)
    assert loaded == cfg

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml_str, encoding="utf-8")
    loaded_from_file = load_config(cfg_file)
    assert isinstance(loaded_from_file, BenchmarkConfig)
    assert loaded_from_file == cfg


def test_sweep_config_validation_and_points() -> None:
    """Verify SweepConfig validation and point config generation."""
    base = BenchmarkConfig(model="Qwen/Qwen2.5-7B-Instruct", max_output_tokens=128)
    sweep = SweepConfig(
        sweep_param="prompt_tokens",
        sweep_values=(128, 512, 2048),
        base_config=base,
        experiment_id="E001A",
    )

    points = sweep.generate_point_configs()
    assert len(points) == 3
    assert points[0][0] == 128
    assert points[0][1].prompt_tokens == 128
    assert points[0][1].max_output_tokens == 128
    assert points[0][1].experiment_id == "E001A"

    assert points[1][0] == 512
    assert points[1][1].prompt_tokens == 512

    assert points[2][0] == 2048
    assert points[2][1].prompt_tokens == 2048

    with pytest.raises(ValueError, match="sweep_param cannot be empty"):
        SweepConfig(sweep_param="", sweep_values=(128,), base_config=base)

    with pytest.raises(ValueError, match="sweep_param must be one of"):
        SweepConfig(sweep_param="invalid_param", sweep_values=(128,), base_config=base)

    with pytest.raises(ValueError, match="sweep_values cannot be empty"):
        SweepConfig(sweep_param="prompt_tokens", sweep_values=(), base_config=base)


def test_sweep_config_yaml_and_load_config(tmp_path) -> None:
    """Verify SweepConfig YAML roundtrip and load_config helper."""
    from inference_os.config import load_config

    yaml_content = """
experiment_id: E001A
sweep_param: prompt_tokens
sweep_values: [128, 512, 2048, 4096]
model: Qwen/Qwen2.5-7B-Instruct
base_url: http://localhost:18000
max_output_tokens: 128
enable_prefix_caching: false
enable_chunked_prefill: false
"""
    yaml_file = tmp_path / "sweep.yaml"
    yaml_file.write_text(yaml_content, encoding="utf-8")

    loaded = load_config(yaml_file)
    assert isinstance(loaded, SweepConfig)
    assert loaded.sweep_param == "prompt_tokens"
    assert loaded.sweep_values == (128, 512, 2048, 4096)
    assert loaded.base_config.model == "Qwen/Qwen2.5-7B-Instruct"
    assert loaded.base_config.base_url == "http://localhost:18000"
    assert loaded.base_config.max_output_tokens == 128
    assert loaded.base_config.enable_prefix_caching is False
    assert loaded.base_config.enable_chunked_prefill is False


def test_load_e001a_actual_config_file() -> None:
    """Verify loading the actual repository configs/e001a_input_scaling.yaml."""
    from pathlib import Path
    from inference_os.config import load_config

    config_path = Path("configs/e001a_input_scaling.yaml")
    assert config_path.is_file()
    cfg = load_config(config_path)
    assert isinstance(cfg, SweepConfig)
    assert cfg.experiment_id == "E001A"
    assert cfg.sweep_param == "prompt_tokens"
    assert cfg.sweep_values == (128, 512, 2048, 4096)
    assert cfg.base_config.base_url == "http://localhost:18000"
    assert cfg.base_config.enable_prefix_caching is False
    assert cfg.base_config.enable_chunked_prefill is False


