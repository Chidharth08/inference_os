"""Offline integration tests for E006 serving economics."""

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from inference_os.metrics.economics import EconomicsConfig


def _load_runner():
    path = Path("experiments/E006-serving-economics/run_e006.py")
    spec = importlib.util.spec_from_file_location("run_e006", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e006_canonical_analysis_preserves_failed_aggregate_slo() -> None:
    runner = _load_runner()
    config = runner.load_economics_config("configs/e006_economics.yaml")
    assert isinstance(config, EconomicsConfig)
    assert config.assumptions.gpu_price_per_hour == pytest.approx(0.190)
    source = json.loads(Path(config.source_summary).read_text(encoding="utf-8"))

    summary = runner.analyze_e005_summary(source, config)

    assert len(summary["points"]) == 5
    assert summary["aggregate_slo_passing_rates"] == []
    assert summary["highest_observed_aggregate_slo_passing_rate"] is None
    assert not summary["sustainable_cost_available"]
    assert summary["benchmark_normalized_totals"]["estimated_cost"] == pytest.approx(
        137 * 0.190 / 3600
    )
    assert all(not point["aggregate_slo_met"] for point in summary["points"])


def test_e006_executes_offline_and_writes_all_artifacts(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.load_economics_config("configs/e006_economics.yaml")
    config = replace(config, output_dir=str(tmp_path / "e006"))

    output_dir, summary = runner.execute_e006(config)

    assert summary["status"] == "SUCCESS"
    assert (output_dir / "config.json").is_file()
    assert (output_dir / "e006_summary.json").is_file()
    assert (output_dir / "e006_report.md").is_file()
    plot_names = {path.name for path in (output_dir / "plots").iterdir()}
    assert plot_names == {
        "cost_per_1000_requests_vs_offered_rate.png",
        "cost_per_million_tokens_vs_offered_rate.png",
        "cost_quality_tradeoff.png",
    }
    assert all(path.stat().st_size > 1000 for path in (output_dir / "plots").iterdir())
