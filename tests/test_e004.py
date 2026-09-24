"""Configuration-contract tests for the E004 experiment runner."""

import importlib.util
from pathlib import Path

import pytest

from inference_os.config import BenchmarkConfig


def _load_runner():
    path = Path("experiments/E004-fixed-vs-variable/run_e004.py")
    spec = importlib.util.spec_from_file_location("run_e004", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e004_canonical_configs_match_means_and_vary_variance() -> None:
    runner = _load_runner()
    configs = runner.load_profile_configs(list(runner.DEFAULT_CONFIGS))

    assert len(configs) == 2
    workloads = [config.workload for config in configs]
    assert all(workload is not None for workload in workloads)
    fixed, variable = workloads
    assert fixed is not None and variable is not None
    assert fixed.input_tokens.mean == variable.input_tokens.mean == 4096
    assert fixed.max_output_tokens.mean == variable.max_output_tokens.mean == 512
    assert fixed.input_tokens.std_dev == 0
    assert variable.input_tokens.std_dev > 0


def test_e004_rejects_control_mismatch() -> None:
    runner = _load_runner()
    configs = runner.load_profile_configs(list(runner.DEFAULT_CONFIGS))
    mismatched = BenchmarkConfig.from_dict(
        {**configs[1].to_dict(), "concurrency": configs[0].concurrency + 1}
    )

    with pytest.raises(ValueError, match="controlled fields"):
        runner._validate_controlled_fields([configs[0], mismatched])
