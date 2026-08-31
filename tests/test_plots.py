"""Unit tests for E001-A plot generation."""

from pathlib import Path

from inference_os.reports.plots import generate_e001a_plots


def test_generate_e001a_plots(tmp_path: Path) -> None:
    """Verify all 4 E001-A plots are generated correctly as non-empty PNG files."""
    sweep_data = [
        {
            "param_value": 128,
            "benchmark": {
                "ttft_stats": {"mean": 0.040, "std_dev": 0.002, "p50": 0.039},
                "e2e_latency_stats": {"mean": 2.50, "std_dev": 0.05, "p50": 2.49},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 18914.0, "avg_memory_used_mb": 18500.0},
        },
        {
            "param_value": 512,
            "benchmark": {
                "ttft_stats": {"mean": 0.090, "std_dev": 0.004, "p50": 0.088},
                "e2e_latency_stats": {"mean": 2.55, "std_dev": 0.06, "p50": 2.54},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 19100.0, "avg_memory_used_mb": 18700.0},
        },
        {
            "param_value": 2048,
            "benchmark": {
                "ttft_stats": {"mean": 0.280, "std_dev": 0.010, "p50": 0.278},
                "e2e_latency_stats": {"mean": 2.74, "std_dev": 0.07, "p50": 2.73},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 19800.0, "avg_memory_used_mb": 19400.0},
        },
        {
            "param_value": 4096,
            "benchmark": {
                "ttft_stats": {"mean": 0.550, "std_dev": 0.020, "p50": 0.545},
                "e2e_latency_stats": {"mean": 3.01, "std_dev": 0.09, "p50": 3.00},
                "tpot_stats": {"mean": 0.020, "std_dev": 0.0005, "p50": 0.020},
            },
            "gpu": {"peak_memory_used_mb": 20500.0, "avg_memory_used_mb": 20100.0},
        },
    ]

    plots_dir = tmp_path / "plots"
    generated = generate_e001a_plots(sweep_data, plots_dir)

    assert len(generated) == 4
    expected_names = {
        "ttft_vs_input_tokens.png",
        "e2e_vs_input_tokens.png",
        "tpot_vs_input_tokens.png",
        "gpu_memory_vs_input_tokens.png",
    }
    actual_names = {p.name for p in generated}
    assert actual_names == expected_names

    for plot_path in generated:
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000
