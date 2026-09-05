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
    print("=" * 105)
    print(" E001-A: INPUT LENGTH SCALING RESULTS SUMMARY")
    print("=" * 105)
    header = (
        f"{'Prompt Tokens':<14} | {'Status':<8} | {'TTFT (P50)':<12} | "
        f"{'TTFT (Mean)':<12} | {'TPOT (P50)':<12} | {'E2E (P50)':<12} | "
        f"{'Peak VRAM':<12} | {'GPU Util':<8}"
    )
    print(header)
    print("-" * 105)

    for pt in sorted(points, key=lambda x: x["param_value"]):
        p_val = pt["param_value"]
        bench = pt.get("benchmark", {})
        succ_reqs = bench.get("successful_requests", 0)
        tot_reqs = bench.get("total_requests", 0)
        is_point_ok = pt.get("success", succ_reqs > 0)

        status_str = "OK" if is_point_ok else f"FAIL ({succ_reqs}/{tot_reqs})"
        ttft_s = bench.get("ttft_stats")
        tpot_s = bench.get("tpot_stats")
        e2e_s = bench.get("e2e_latency_stats")
        gpu_s = pt.get("gpu")

        ttft_p50_str = (
            format_ms(ttft_s["p50"])
            if ttft_s
            else ("FAILED" if not is_point_ok else "N/A")
        )
        ttft_mean_str = (
            format_ms(ttft_s["mean"])
            if ttft_s
            else ("FAILED" if not is_point_ok else "N/A")
        )
        tpot_p50_str = (
            format_ms(tpot_s["p50"])
            if tpot_s
            else ("FAILED" if not is_point_ok else "N/A")
        )
        e2e_p50_str = (
            format_ms(e2e_s["p50"])
            if e2e_s
            else ("FAILED" if not is_point_ok else "N/A")
        )

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
            f"{p_val:<14} | {status_str:<8} | {ttft_p50_str:<12} | "
            f"{ttft_mean_str:<12} | {tpot_p50_str:<12} | {e2e_p50_str:<12} | "
            f"{vram_val:<12} | {gpu_util_str:<8}"
        )
        print(row)

    print("=" * 105)


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

    prefix_cache_status = (
        "disabled" if not sweep_config.base_config.enable_prefix_caching else "enabled"
    )
    chunked_prefill_status = (
        "disabled"
        if not sweep_config.base_config.enable_chunked_prefill
        and sweep_config.base_config.chunked_prefill is None
        else f"{sweep_config.base_config.chunked_prefill}"
    )

    print("=" * 75)
    print(" inference_os — E001-A Input Length Scaling Experiment")
    print("=" * 75)
    print(f" Model:               {sweep_config.base_config.model}")
    print(f" Endpoint:            {sweep_config.base_config.base_url}")
    print(f" Sweep Parameter:     {sweep_config.sweep_param}")
    print(f" Sweep Values:        {list(sweep_config.sweep_values)}")
    print(f" Fixed Output Tokens: {sweep_config.base_config.max_output_tokens}")
    print(" Concurrency:         1 (sequential)")
    prefix_val = sweep_config.base_config.enable_prefix_caching
    chunked_val = sweep_config.base_config.enable_chunked_prefill
    print(f" Prefix Caching:      {prefix_val} ({prefix_cache_status})")
    print(f" Chunked Prefill:     {chunked_val} ({chunked_prefill_status})")
    print(f" Measured Requests:   {sweep_config.base_config.num_requests}")
    print(f" Warm-up Requests:    {sweep_config.base_config.warmup_requests}")

    print("-" * 75)
    print("Executing parameter sweep pipeline...")

    sweep_dir, point_results = await execute_sweep(sweep_config)

    successful_points = sum(
        1
        for p in point_results
        if p.get("benchmark", {}).get("successful_requests", 0) > 0
    )
    total_points = len(point_results)

    print("-" * 75)
    if successful_points == 0:
        print(
            " [ERROR] Sweep Execution Failed: All sweep points encountered 0 "
            "successful requests!"
        )
        # Collect and print sample error
        sample_errors = []
        for p in point_results:
            errs = p.get("benchmark", {}).get("errors") or []
            sample_errors.extend(errs)
        if sample_errors:
            print(f" Sample Error: {sample_errors[0]}")
        print()
        print_sweep_table(point_results)
        print()
        print(f" Artifacts & Error Logs:       {Path(sweep_dir).resolve()}")
        print("=" * 75)
        return 1

    if successful_points < total_points:
        failed_count = total_points - successful_points
        print(
            f" [WARNING] Sweep Execution Completed with Failures: "
            f"{failed_count}/{total_points} points failed."
        )
    else:
        print(" Sweep Execution Completed Successfully!")

    print()
    print_sweep_table(point_results)
    print()
    print(f" Artifacts & Raw Measurements: {Path(sweep_dir).resolve()}")
    if (Path(sweep_dir) / "plots").exists():
        print(f" Rendered Plots Directory:     {Path(sweep_dir / 'plots').resolve()}")
    print("=" * 75)
    return 0 if successful_points == total_points else 1


def main() -> int:
    """CLI entry point for running E001-A benchmark."""
    parser = create_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
