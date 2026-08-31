"""E001-A — Input Length Scaling Benchmark Execution Script."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

from inference_os.config import SweepConfig, load_config
from inference_os.runner.sweep import execute_sweep


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for E001-A benchmark."""
    parser = argparse.ArgumentParser(
        description="E001-A — Input Length Scaling Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/e001a_input_scaling.yaml",
        help="Path to YAML/JSON configuration file for E001-A",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override base URL of the vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory for sweep artifacts",
    )
    return parser


def format_ms(val_seconds: float | None) -> str:
    """Format seconds value as milliseconds string."""
    if val_seconds is None:
        return "N/A"
    return f"{val_seconds * 1000.0:7.2f} ms"


def print_sweep_table(points: list[dict[str, Any]]) -> None:
    """Print ASCII comparison table across all sweep points."""
    print("=" * 95)
    print(" E001-A: INPUT LENGTH SCALING RESULTS SUMMARY")
    print("=" * 95)
    header = (
        f"{'Prompt Tokens':<14} | {'TTFT (P50)':<12} | {'TTFT (Mean)':<12} | "
        f"{'TPOT (P50)':<12} | {'E2E (P50)':<12} | {'Peak VRAM':<12} | {'GPU Util':<8}"
    )
    print(header)
    print("-" * 95)

    for pt in sorted(points, key=lambda x: x["param_value"]):
        p_val = pt["param_value"]
        bench = pt.get("benchmark", {})
        ttft_s = bench.get("ttft_stats")
        tpot_s = bench.get("tpot_stats")
        e2e_s = bench.get("e2e_latency_stats")
        gpu_s = pt.get("gpu")

        ttft_p50_str = format_ms(ttft_s["p50"]) if ttft_s else "N/A"
        ttft_mean_str = format_ms(ttft_s["mean"]) if ttft_s else "N/A"
        tpot_p50_str = format_ms(tpot_s["p50"]) if tpot_s else "N/A"
        e2e_p50_str = format_ms(e2e_s["p50"]) if e2e_s else "N/A"

        vram_val = (
            f"{gpu_s['peak_memory_used_mb']:.0f} MiB"
            if gpu_s and "peak_memory_used_mb" in gpu_s
            else "N/A"
        )
        gpu_util_str = (
            f"{gpu_s['avg_utilization_gpu_pct']:.1f}%"
            if gpu_s and "avg_utilization_gpu_pct" in gpu_s
            else "N/A"
        )

        row = (
            f"{p_val:<14} | {ttft_p50_str:<12} | {ttft_mean_str:<12} | "
            f"{tpot_p50_str:<12} | {e2e_p50_str:<12} | "
            f"{vram_val:<12} | {gpu_util_str:<8}"
        )
        print(row)

    print("=" * 95)


async def main_async(args: argparse.Namespace) -> int:
    """Async main entrypoint for E001-A benchmark."""
    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}")
        return 1

    loaded = load_config(config_path)
    if not isinstance(loaded, SweepConfig):
        print(f"Error: Config file {config_path} does not specify a SweepConfig")
        return 1

    sweep_config = loaded
    if args.base_url or args.output_dir:
        from dataclasses import replace
        overrides = {}
        if args.base_url:
            overrides["base_url"] = args.base_url
        if args.output_dir:
            overrides["output_dir"] = args.output_dir
        base_cfg = replace(sweep_config.base_config, **overrides)
        sweep_config = replace(sweep_config, base_config=base_cfg)

    print("=" * 75)
    print(" inference_os — E001-A Input Length Scaling Experiment")
    print("=" * 75)
    print(f" Model:               {sweep_config.base_config.model}")
    print(f" Endpoint:            {sweep_config.base_config.base_url}")
    print(f" Sweep Parameter:     {sweep_config.sweep_param}")
    print(f" Sweep Values:        {list(sweep_config.sweep_values)}")
    print(f" Fixed Output Tokens: {sweep_config.base_config.max_output_tokens}")
    print(" Concurrency:         1 (sequential)")
    print(f" Prefix Caching:      {sweep_config.base_config.enable_prefix_caching}")
    print(f" Chunked Prefill:     {sweep_config.base_config.chunked_prefill}")
    print(f" Measured Requests:   {sweep_config.base_config.num_requests}")
    print(f" Warm-up Requests:    {sweep_config.base_config.warmup_requests}")

    print("-" * 75)
    print("Executing parameter sweep pipeline...")

    sweep_dir, point_results = await execute_sweep(sweep_config)

    print("-" * 75)
    print(" Sweep Execution Completed Successfully!")
    print()
    print_sweep_table(point_results)
    print()
    print(f" Artifacts & Raw Measurements: {Path(sweep_dir).resolve()}")
    print(f" Rendered Plots Directory:     {Path(sweep_dir / 'plots').resolve()}")
    print("=" * 75)
    return 0


def main() -> int:
    """CLI entry point for running E001-A benchmark."""
    parser = create_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
