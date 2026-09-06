"""Matplotlib plot generation for E001-A scaling experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_e001a_plots(
    sweep_data: Sequence[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Generate publication-quality plots for E001-A input length scaling sweep.

    Plots generated:
    1. ttft_vs_input_tokens.png: TTFT (mean + std dev, P50) vs prompt tokens
    2. e2e_vs_input_tokens.png: End-to-end latency vs prompt tokens
    3. tpot_vs_input_tokens.png: Time per output token vs prompt tokens
    4. gpu_memory_vs_input_tokens.png: Peak and Avg VRAM vs prompt tokens

    Args:
        sweep_data: List of point summary dicts, each containing:
            - "param_value" (int): prompt_tokens count
            - "benchmark" (dict): BenchmarkSummary as dict
            - "gpu" (dict | None): Optional GPUSummary as dict
        output_dir: Target directory to write PNG images to.

    Returns:
        List of generated image file Paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_points = sorted(sweep_data, key=lambda p: p["param_value"])
    prompt_tokens = [p["param_value"] for p in sorted_points]

    ttft_means = []
    ttft_stds = []
    ttft_p50s = []

    e2e_means = []
    e2e_stds = []
    e2e_p50s = []

    tpot_means = []
    tpot_stds = []
    tpot_p50s = []

    peak_vram_mb = []
    avg_vram_mb = []

    for pt in sorted_points:
        bench = pt.get("benchmark", {})
        ttft_s = bench.get("ttft_stats")
        e2e_s = bench.get("e2e_latency_stats")
        tpot_s = bench.get("tpot_stats")
        gpu_s = pt.get("gpu")

        if ttft_s:
            ttft_means.append(ttft_s["mean"] * 1000.0)
            ttft_stds.append(ttft_s["std_dev"] * 1000.0)
            ttft_p50s.append(ttft_s["p50"] * 1000.0)
        else:
            ttft_means.append(0.0)
            ttft_stds.append(0.0)
            ttft_p50s.append(0.0)

        if e2e_s:
            e2e_means.append(e2e_s["mean"] * 1000.0)
            e2e_stds.append(e2e_s["std_dev"] * 1000.0)
            e2e_p50s.append(e2e_s["p50"] * 1000.0)
        else:
            e2e_means.append(0.0)
            e2e_stds.append(0.0)
            e2e_p50s.append(0.0)

        if tpot_s:
            tpot_means.append(tpot_s["mean"] * 1000.0)
            tpot_stds.append(tpot_s["std_dev"] * 1000.0)
            tpot_p50s.append(tpot_s["p50"] * 1000.0)
        else:
            tpot_means.append(0.0)
            tpot_stds.append(0.0)
            tpot_p50s.append(0.0)

        if gpu_s and "peak_memory_used_mb" in gpu_s:
            peak_vram_mb.append(gpu_s["peak_memory_used_mb"])
            avg_mem = gpu_s.get("avg_memory_used_mb", gpu_s["peak_memory_used_mb"])
            avg_vram_mb.append(avg_mem)
        else:
            peak_vram_mb.append(0.0)
            avg_vram_mb.append(0.0)

    generated_plots: list[Path] = []

    # Plot 1: TTFT vs Input Length
    p1 = out_path / "ttft_vs_input_tokens.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        prompt_tokens,
        ttft_means,
        yerr=ttft_stds,
        fmt="o-",
        capsize=4,
        color="#1f77b4",
        label="Mean TTFT (±1σ)",
        linewidth=2,
    )
    ax.plot(prompt_tokens, ttft_p50s, "s--", color="#ff7f0e", label="P50 TTFT")
    ax.set_title(
        "E001-A: Time to First Token (TTFT) vs Input Length",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Input Prompt Tokens", fontsize=10)
    ax.set_ylabel("TTFT (ms)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    generated_plots.append(p1)

    # Plot 2: E2E vs Input Length
    p2 = out_path / "e2e_vs_input_tokens.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        prompt_tokens,
        e2e_means,
        yerr=e2e_stds,
        fmt="o-",
        capsize=4,
        color="#2ca02c",
        label="Mean E2E Latency (±1σ)",
        linewidth=2,
    )
    ax.plot(prompt_tokens, e2e_p50s, "s--", color="#d62728", label="P50 E2E")
    ax.set_title(
        "E001-A: End-to-End Latency vs Input Length (Fixed Output=128)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Input Prompt Tokens", fontsize=10)
    ax.set_ylabel("E2E Latency (ms)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    generated_plots.append(p2)

    # Plot 3: TPOT vs Input Length
    p3 = out_path / "tpot_vs_input_tokens.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        prompt_tokens,
        tpot_means,
        yerr=tpot_stds,
        fmt="o-",
        capsize=4,
        color="#9467bd",
        label="Mean TPOT (±1σ)",
        linewidth=2,
    )
    ax.plot(prompt_tokens, tpot_p50s, "s--", color="#8c564b", label="P50 TPOT")
    ax.set_title(
        "E001-A: Time Per Output Token (TPOT) vs Input Length",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Input Prompt Tokens", fontsize=10)
    ax.set_ylabel("TPOT (ms / token)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    generated_plots.append(p3)

    # Plot 4: GPU Memory vs Input Length
    if any(v > 0 for v in peak_vram_mb):
        p4 = out_path / "gpu_memory_vs_input_tokens.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            prompt_tokens,
            peak_vram_mb,
            "o-",
            color="#e377c2",
            label="Peak VRAM",
            linewidth=2,
        )
        ax.plot(
            prompt_tokens,
            avg_vram_mb,
            "s--",
            color="#7f7f7f",
            label="Avg VRAM",
            linewidth=1.5,
        )
        ax.set_title(
            "E001-A: GPU Memory Usage vs Input Length",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Input Prompt Tokens", fontsize=10)
        ax.set_ylabel("VRAM Usage (MiB)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        generated_plots.append(p4)

    return generated_plots


def generate_e001b_plots(
    sweep_data: Sequence[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Generate publication-quality plots for E001-B output length scaling sweep.

    Plots generated:
    1. ttft_vs_output_tokens.png: TTFT (mean + std dev, P50) vs output tokens
    2. e2e_vs_output_tokens.png: End-to-end latency vs output tokens
    3. tpot_vs_output_tokens.png: Time per output token vs output tokens
    4. gpu_memory_vs_output_tokens.png: Peak and Avg VRAM vs output tokens

    Args:
        sweep_data: List of point summary dicts, each containing:
            - "param_value" (int): max_output_tokens count
            - "benchmark" (dict): BenchmarkSummary as dict
            - "gpu" (dict | None): Optional GPUSummary as dict
        output_dir: Target directory to write PNG images to.

    Returns:
        List of generated image file Paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_points = sorted(sweep_data, key=lambda p: p["param_value"])
    output_tokens = [p["param_value"] for p in sorted_points]

    ttft_means = []
    ttft_stds = []
    ttft_p50s = []

    e2e_means = []
    e2e_stds = []
    e2e_p50s = []

    tpot_means = []
    tpot_stds = []
    tpot_p50s = []

    peak_vram_mb = []
    avg_vram_mb = []

    for pt in sorted_points:
        bench = pt.get("benchmark", {})
        ttft_s = bench.get("ttft_stats")
        e2e_s = bench.get("e2e_latency_stats")
        tpot_s = bench.get("tpot_stats")
        gpu_s = pt.get("gpu")

        if ttft_s:
            ttft_means.append(ttft_s["mean"] * 1000.0)
            ttft_stds.append(ttft_s["std_dev"] * 1000.0)
            ttft_p50s.append(ttft_s["p50"] * 1000.0)
        else:
            ttft_means.append(0.0)
            ttft_stds.append(0.0)
            ttft_p50s.append(0.0)

        if e2e_s:
            e2e_means.append(e2e_s["mean"] * 1000.0)
            e2e_stds.append(e2e_s["std_dev"] * 1000.0)
            e2e_p50s.append(e2e_s["p50"] * 1000.0)
        else:
            e2e_means.append(0.0)
            e2e_stds.append(0.0)
            e2e_p50s.append(0.0)

        if tpot_s:
            tpot_means.append(tpot_s["mean"] * 1000.0)
            tpot_stds.append(tpot_s["std_dev"] * 1000.0)
            tpot_p50s.append(tpot_s["p50"] * 1000.0)
        else:
            tpot_means.append(0.0)
            tpot_stds.append(0.0)
            tpot_p50s.append(0.0)

        if gpu_s and "peak_memory_used_mb" in gpu_s:
            peak_vram_mb.append(gpu_s["peak_memory_used_mb"])
            avg_mem = gpu_s.get("avg_memory_used_mb", gpu_s["peak_memory_used_mb"])
            avg_vram_mb.append(avg_mem)
        else:
            peak_vram_mb.append(0.0)
            avg_vram_mb.append(0.0)

    generated_plots: list[Path] = []

    # Plot 1: TTFT vs Output Length
    p1 = out_path / "ttft_vs_output_tokens.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        output_tokens,
        ttft_means,
        yerr=ttft_stds,
        fmt="o-",
        capsize=4,
        color="#1f77b4",
        label="Mean TTFT (±1σ)",
        linewidth=2,
    )
    ax.plot(output_tokens, ttft_p50s, "s--", color="#ff7f0e", label="P50 TTFT")
    ax.set_title(
        "E001-B: Time to First Token (TTFT) vs Output Length (Fixed Input=128)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Output Tokens", fontsize=10)
    ax.set_ylabel("TTFT (ms)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    generated_plots.append(p1)

    # Plot 2: E2E vs Output Length
    p2 = out_path / "e2e_vs_output_tokens.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        output_tokens,
        e2e_means,
        yerr=e2e_stds,
        fmt="o-",
        capsize=4,
        color="#2ca02c",
        label="Mean E2E Latency (±1σ)",
        linewidth=2,
    )
    ax.plot(output_tokens, e2e_p50s, "s--", color="#d62728", label="P50 E2E")
    ax.set_title(
        "E001-B: End-to-End Latency vs Output Length (Fixed Input=128)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Output Tokens", fontsize=10)
    ax.set_ylabel("E2E Latency (ms)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    generated_plots.append(p2)

    # Plot 3: TPOT vs Output Length
    p3 = out_path / "tpot_vs_output_tokens.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        output_tokens,
        tpot_means,
        yerr=tpot_stds,
        fmt="o-",
        capsize=4,
        color="#9467bd",
        label="Mean TPOT (±1σ)",
        linewidth=2,
    )
    ax.plot(output_tokens, tpot_p50s, "s--", color="#8c564b", label="P50 TPOT")
    ax.set_title(
        "E001-B: Time Per Output Token (TPOT) vs Output Length",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Output Tokens", fontsize=10)
    ax.set_ylabel("TPOT (ms / token)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    generated_plots.append(p3)

    # Plot 4: GPU Memory vs Output Length
    if any(v > 0 for v in peak_vram_mb):
        p4 = out_path / "gpu_memory_vs_output_tokens.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(
            output_tokens,
            peak_vram_mb,
            "o-",
            color="#e377c2",
            label="Peak VRAM",
            linewidth=2,
        )
        ax.plot(
            output_tokens,
            avg_vram_mb,
            "s--",
            color="#7f7f7f",
            label="Avg VRAM",
            linewidth=1.5,
        )
        ax.set_title(
            "E001-B: GPU Memory Usage vs Output Length",
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Output Tokens", fontsize=10)
        ax.set_ylabel("VRAM Usage (MiB)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        generated_plots.append(p4)

    return generated_plots


def generate_e002_plots(
    sweep_data: Sequence[dict[str, Any]],
    output_dir: Path | str,
) -> list[Path]:
    """Generate publication-quality plots for E002 concurrency scaling sweep.

    Plots generated:
    1. throughput_requests_vs_concurrency.png: Request throughput (req/s)
    2. throughput_tokens_vs_concurrency.png: Output token throughput (tok/s)
    3. latency_vs_concurrency.png: TTFT, TPOT, and E2E latency percentiles
    4. gpu_metrics_vs_concurrency.png: GPU utilization (%) and VRAM (MiB)

    Args:
        sweep_data: List of point summary dicts.
        output_dir: Target directory to write PNG images to.

    Returns:
        List of generated image file Paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sorted_points = sorted(sweep_data, key=lambda p: p["param_value"])
    concurrency_values = [p["param_value"] for p in sorted_points]

    req_throughputs = []
    tok_throughputs = []

    ttft_p50s = []
    ttft_p95s = []

    tpot_p50s = []
    tpot_p95s = []

    e2e_p50s = []
    e2e_p95s = []

    peak_vram_mb = []
    avg_vram_mb = []
    peak_gpu_util = []
    avg_gpu_util = []

    for pt in sorted_points:
        bench = pt.get("benchmark", {})
        req_throughputs.append(bench.get("request_throughput", 0.0))
        tok_throughputs.append(bench.get("output_token_throughput", 0.0))

        ttft_s = bench.get("ttft_stats")
        if ttft_s:
            ttft_p50s.append(ttft_s["p50"] * 1000.0)
            ttft_p95s.append(ttft_s["p95"] * 1000.0)
        else:
            ttft_p50s.append(0.0)
            ttft_p95s.append(0.0)

        tpot_s = bench.get("tpot_stats")
        if tpot_s:
            tpot_p50s.append(tpot_s["p50"] * 1000.0)
            tpot_p95s.append(tpot_s["p95"] * 1000.0)
        else:
            tpot_p50s.append(0.0)
            tpot_p95s.append(0.0)

        e2e_s = bench.get("e2e_latency_stats")
        if e2e_s:
            e2e_p50s.append(e2e_s["p50"] * 1000.0)
            e2e_p95s.append(e2e_s["p95"] * 1000.0)
        else:
            e2e_p50s.append(0.0)
            e2e_p95s.append(0.0)

        gpu_s = pt.get("gpu")
        if gpu_s:
            peak_vram_mb.append(gpu_s.get("peak_memory_used_mb", 0))
            avg_vram_mb.append(gpu_s.get("avg_memory_used_mb", 0.0))
            peak_gpu_util.append(gpu_s.get("peak_utilization_gpu_pct", 0))
            avg_gpu_util.append(gpu_s.get("avg_utilization_gpu_pct", 0.0))
        else:
            peak_vram_mb.append(0)
            avg_vram_mb.append(0.0)
            peak_gpu_util.append(0)
            avg_gpu_util.append(0.0)

    generated_plots: list[Path] = []

    # Plot 1: Concurrency vs Requests / sec
    p1 = out_path / "throughput_requests_vs_concurrency.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        concurrency_values,
        req_throughputs,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=6,
        label="Request Throughput",
    )
    ax.set_title(
        "E002: Request Throughput vs Concurrency",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Client Concurrency (in-flight requests)", fontsize=10)
    ax.set_ylabel("Throughput (req/s)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    generated_plots.append(p1)

    # Plot 2: Concurrency vs Output Tokens / sec
    p2 = out_path / "throughput_tokens_vs_concurrency.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        concurrency_values,
        tok_throughputs,
        "s-",
        color="#2ca02c",
        linewidth=2,
        markersize=6,
        label="Output Token Throughput",
    )
    ax.set_title(
        "E002: Output Token Throughput vs Concurrency",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Client Concurrency (in-flight requests)", fontsize=10)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    generated_plots.append(p2)

    # Plot 3: Concurrency vs Latency Percentiles (TTFT, TPOT, E2E)
    p3 = out_path / "latency_vs_concurrency.png"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Subplot 3a: TTFT
    axes[0].plot(concurrency_values, ttft_p50s, "o-", color="#1f77b4", label="TTFT P50")
    axes[0].plot(
        concurrency_values, ttft_p95s, "s--", color="#ff7f0e", label="TTFT P95"
    )
    axes[0].set_title("Time to First Token (TTFT)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Concurrency", fontsize=10)
    axes[0].set_ylabel("Latency (ms)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left")

    # Subplot 3b: TPOT
    axes[1].plot(concurrency_values, tpot_p50s, "o-", color="#2ca02c", label="TPOT P50")
    axes[1].plot(
        concurrency_values, tpot_p95s, "s--", color="#d62728", label="TPOT P95"
    )
    axes[1].set_title("Time Per Output Token (TPOT)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Concurrency", fontsize=10)
    axes[1].set_ylabel("Latency (ms/token)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left")

    # Subplot 3c: E2E
    axes[2].plot(concurrency_values, e2e_p50s, "o-", color="#9467bd", label="E2E P50")
    axes[2].plot(concurrency_values, e2e_p95s, "s--", color="#8c564b", label="E2E P95")
    axes[2].set_title("End-to-End Latency (E2E)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Concurrency", fontsize=10)
    axes[2].set_ylabel("Latency (ms)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper left")

    fig.suptitle(
        "E002: Latency Percentiles vs Concurrency", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    generated_plots.append(p3)

    # Plot 4: Concurrency vs GPU Utilization / Memory
    if any(v > 0 for v in peak_vram_mb) or any(u > 0 for u in peak_gpu_util):
        p4 = out_path / "gpu_metrics_vs_concurrency.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 4a: GPU Compute Utilization
        axes[0].plot(
            concurrency_values,
            avg_gpu_util,
            "o-",
            color="#17becf",
            label="Avg GPU Util %",
        )
        axes[0].plot(
            concurrency_values,
            peak_gpu_util,
            "^--",
            color="#bcbd22",
            label="Peak GPU Util %",
        )
        axes[0].set_title("GPU Compute Utilization", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("Concurrency", fontsize=10)
        axes[0].set_ylabel("Utilization (%)", fontsize=10)
        axes[0].set_ylim(0, 105)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(loc="lower right")

        # 4b: GPU Memory
        axes[1].plot(
            concurrency_values,
            peak_vram_mb,
            "o-",
            color="#e377c2",
            label="Peak VRAM",
        )
        axes[1].plot(
            concurrency_values,
            avg_vram_mb,
            "s--",
            color="#7f7f7f",
            label="Avg VRAM",
        )
        axes[1].set_title("GPU Memory (VRAM) Usage", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("Concurrency", fontsize=10)
        axes[1].set_ylabel("Memory (MiB)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="upper left")

        fig.suptitle("E002: GPU Metrics vs Concurrency", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(p4, dpi=200)
        plt.close(fig)
        generated_plots.append(p4)

    return generated_plots
