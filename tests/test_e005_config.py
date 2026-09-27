"""Configuration tests for E005 duration-based open-loop sweeps."""

from pathlib import Path

import pytest

from inference_os.config import OpenLoopSweepConfig, SLOConfig, load_config


def test_load_e005_config() -> None:
    config = load_config(Path("configs/e005_open_loop.yaml"))
    assert isinstance(config, OpenLoopSweepConfig)
    assert config.request_rates == (0.5, 1.0, 2.0, 3.0, 4.0)
    assert config.duration_seconds is None
    assert config.requests_per_rate == 30
    assert config.duration_for_rate(2.0) == 15.0
    assert config.base_config.workload is not None
    assert config.slo == SLOConfig()


def test_open_loop_config_rejects_duplicate_rates() -> None:
    with pytest.raises(ValueError, match="unique"):
        OpenLoopSweepConfig.from_dict(
            {
                "model": "model",
                "request_rates": [1.0, 1.0],
                "requests_per_rate": 50,
                "max_in_flight": 2,
                "max_drain_seconds": 5,
            }
        )


def test_open_loop_config_requires_one_run_length_mode() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        OpenLoopSweepConfig.from_dict(
            {
                "model": "model",
                "request_rates": [1.0],
                "duration_seconds": 10,
                "requests_per_rate": 50,
            }
        )
