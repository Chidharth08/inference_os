"""Unit tests for E001-A plot generation."""

from pathlib import Path

from inference_os.reports.plots import (
    generate_e001a_plots,
    generate_e001b_plots,
    generate_e002_plots,
)


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


def test_generate_e001b_plots(tmp_path: Path) -> None:
    """Verify all 4 E001-B plots are generated correctly as non-empty PNG files."""
    sweep_data = [
        {
            "param_value": 32,
            "benchmark": {
                "ttft_stats": {"mean": 0.040, "std_dev": 0.002, "p50": 0.039},
                "e2e_latency_stats": {"mean": 0.65, "std_dev": 0.01, "p50": 0.64},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 18914.0, "avg_memory_used_mb": 18500.0},
        },
        {
            "param_value": 128,
            "benchmark": {
                "ttft_stats": {"mean": 0.041, "std_dev": 0.002, "p50": 0.040},
                "e2e_latency_stats": {"mean": 2.50, "std_dev": 0.05, "p50": 2.49},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 19000.0, "avg_memory_used_mb": 18600.0},
        },
        {
            "param_value": 512,
            "benchmark": {
                "ttft_stats": {"mean": 0.040, "std_dev": 0.002, "p50": 0.039},
                "e2e_latency_stats": {"mean": 9.80, "std_dev": 0.10, "p50": 9.78},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 19500.0, "avg_memory_used_mb": 19100.0},
        },
        {
            "param_value": 1024,
            "benchmark": {
                "ttft_stats": {"mean": 0.041, "std_dev": 0.003, "p50": 0.040},
                "e2e_latency_stats": {"mean": 19.50, "std_dev": 0.20, "p50": 19.45},
                "tpot_stats": {"mean": 0.019, "std_dev": 0.0005, "p50": 0.019},
            },
            "gpu": {"peak_memory_used_mb": 20200.0, "avg_memory_used_mb": 19800.0},
        },
    ]

    plots_dir = tmp_path / "plots"
    generated = generate_e001b_plots(sweep_data, plots_dir)

    assert len(generated) == 4
    expected_names = {
        "ttft_vs_output_tokens.png",
        "e2e_vs_output_tokens.png",
        "tpot_vs_output_tokens.png",
        "gpu_memory_vs_output_tokens.png",
    }
    actual_names = {p.name for p in generated}
    assert actual_names == expected_names

    for plot_path in generated:
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000


def test_generate_e002_plots(tmp_path: Path) -> None:
    """Verify all 4 E002 plots are generated correctly as non-empty PNG files."""
    sweep_data = [
        {
            "param_value": 1,
            "benchmark": {
                "request_throughput": 0.8,
                "output_token_throughput": 50.0,
                "ttft_stats": {
                    "p50": 0.040,
                    "p95": 0.045,
                    "mean": 0.041,
                    "std_dev": 0.002,
                },
                "e2e_latency_stats": {
                    "p50": 2.50,
                    "p95": 2.55,
                    "mean": 2.51,
                    "std_dev": 0.03,
                },
                "tpot_stats": {
                    "p50": 0.019,
                    "p95": 0.020,
                    "mean": 0.019,
                    "std_dev": 0.0005,
                },
            },
            "gpu": {
                "peak_memory_used_mb": 18914.0,
                "avg_memory_used_mb": 18500.0,
                "peak_utilization_gpu_pct": 95,
                "avg_utilization_gpu_pct": 80.0,
            },
        },
        {
            "param_value": 4,
            "benchmark": {
                "request_throughput": 2.5,
                "output_token_throughput": 160.0,
                "ttft_stats": {
                    "p50": 0.065,
                    "p95": 0.090,
                    "mean": 0.070,
                    "std_dev": 0.010,
                },
                "e2e_latency_stats": {
                    "p50": 3.20,
                    "p95": 3.60,
                    "mean": 3.30,
                    "std_dev": 0.15,
                },
                "tpot_stats": {
                    "p50": 0.024,
                    "p95": 0.027,
                    "mean": 0.025,
                    "std_dev": 0.001,
                },
            },
            "gpu": {
                "peak_memory_used_mb": 19400.0,
                "avg_memory_used_mb": 19000.0,
                "peak_utilization_gpu_pct": 98,
                "avg_utilization_gpu_pct": 92.0,
            },
        },
        {
            "param_value": 16,
            "benchmark": {
                "request_throughput": 4.2,
                "output_token_throughput": 260.0,
                "ttft_stats": {
                    "p50": 0.150,
                    "p95": 0.280,
                    "mean": 0.170,
                    "std_dev": 0.040,
                },
                "e2e_latency_stats": {
                    "p50": 5.80,
                    "p95": 7.10,
                    "mean": 6.00,
                    "std_dev": 0.40,
                },
                "tpot_stats": {
                    "p50": 0.040,
                    "p95": 0.048,
                    "mean": 0.042,
                    "std_dev": 0.003,
                },
            },
            "gpu": {
                "peak_memory_used_mb": 20800.0,
                "avg_memory_used_mb": 20400.0,
                "peak_utilization_gpu_pct": 100,
                "avg_utilization_gpu_pct": 98.0,
            },
        },
    ]

    plots_dir = tmp_path / "plots"
    generated = generate_e002_plots(sweep_data, plots_dir)

    assert len(generated) == 4
    expected_names = {
        "throughput_requests_vs_concurrency.png",
        "throughput_tokens_vs_concurrency.png",
        "latency_vs_concurrency.png",
        "gpu_metrics_vs_concurrency.png",
    }
    actual_names = {p.name for p in generated}
    assert actual_names == expected_names

    for plot_path in generated:
        assert plot_path.is_file()
        assert plot_path.stat().st_size > 1000
