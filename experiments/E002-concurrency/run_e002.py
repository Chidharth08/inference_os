"""E002 — Concurrency Scaling Benchmark Execution Script."""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from inference_os.config import SweepConfig, load_config
from inference_os.runner.sweep import execute_sweep


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for E002 benchmark."""
    parser = argparse.ArgumentParser(
        description="E002 — Concurrency Scaling Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON configuration file for E002",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run cheap pilot verification sweep ([1, 4, 8] concurrency)",
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
    print("=" * 120)
    print(" E002: CONCURRENCY SCALING RESULTS SUMMARY")
    print("=" * 120)
    header = (
        f"{'Concurrency':<12} | {'Status':<8} | {'Req Throughput':<14} | "
        f"{'Tok Throughput':<14} | {'TTFT (P50)':<12} | {'TPOT (P50)':<12} | "
        f"{'E2E (P50)':<12} | {'Peak VRAM':<11} | {'GPU Util':<8} | {'Err %':<6}"
    )
    print(header)
    print("-" * 120)

    for pt in sorted(points, key=lambda x: x["param_value"]):
        c_val = pt["param_value"]
        bench = pt.get("benchmark", {})
        succ_reqs = bench.get("successful_requests", 0)
        tot_reqs = bench.get("total_requests", 0)
        is_point_ok = pt.get("success", succ_reqs > 0)

        status_str = "OK" if is_point_ok else f"FAIL ({succ_reqs}/{tot_reqs})"
        req_tp = bench.get("request_throughput", 0.0)
        tok_tp = bench.get("output_token_throughput", 0.0)
        err_rate = bench.get("error_rate", 0.0)

        req_tp_str = f"{req_tp:6.2f} req/s" if is_point_ok else "FAILED"
        tok_tp_str = f"{tok_tp:6.2f} tok/s" if is_point_ok else "FAILED"

        ttft_s = bench.get("ttft_stats")
        tpot_s = bench.get("tpot_stats")
        e2e_s = bench.get("e2e_latency_stats")
        gpu_s = pt.get("gpu")

        ttft_p50_str = (
            format_ms(ttft_s["p50"])
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
        err_str = f"{err_rate * 100.0:4.1f}%"

        row = (
            f"{c_val:<12} | {status_str:<8} | {req_tp_str:<14} | "
            f"{tok_tp_str:<14} | {ttft_p50_str:<12} | {tpot_p50_str:<12} | "
            f"{e2e_p50_str:<12} | {vram_val:<11} | {gpu_util_str:<8} | {err_str:<6}"
        )
        print(row)

    print("=" * 120)


async def main_async(args: argparse.Namespace) -> int:
    """Async main entrypoint for E002 benchmark."""
    if args.config:
        config_path = Path(args.config)
    elif args.pilot:
        config_path = Path("configs/e002_pilot_concurrency.yaml")
    else:
        config_path = Path("configs/e002_concurrency.yaml")

    if not config_path.is_file():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    print(f"Loading configuration from {config_path}...")
    loaded_cfg = load_config(config_path)
    if not isinstance(loaded_cfg, SweepConfig):
        print(
            f"Error: Config must be a SweepConfig, got {type(loaded_cfg).__name__}",
            file=sys.stderr,
        )
        return 1

    # Apply CLI overrides if provided
    base_overrides: dict[str, Any] = {}
    if args.base_url:
        base_overrides["base_url"] = args.base_url
    if args.output_dir:
        base_overrides["output_dir"] = args.output_dir

    if base_overrides:
        new_base = replace(loaded_cfg.base_config, **base_overrides)
        loaded_cfg = replace(loaded_cfg, base_config=new_base)

    vals = list(loaded_cfg.sweep_values)
    print(f"Executing E002 Sweep: param={loaded_cfg.sweep_param}, values={vals}")
    print(
        f"Base model: {loaded_cfg.base_config.model}, "
        f"Server: {loaded_cfg.base_config.base_url}"
    )
    print(
        f"Input tokens: {loaded_cfg.base_config.prompt_tokens}, "
        f"Output tokens: {loaded_cfg.base_config.max_output_tokens}"
    )
    print(f"Requests per point: {loaded_cfg.base_config.num_requests}")
    print("-" * 60)

    sweep_dir, point_results = await execute_sweep(
        sweep_config=loaded_cfg,
        generate_plots=True,
    )

    print("\nBenchmark sweep completed!")
    print(f"Output directory: {sweep_dir}\n")

    print_sweep_table(point_results)

    # Check for failures
    successful_points = sum(1 for p in point_results if p.get("success", False))
    total_points = len(point_results)

    if successful_points == 0:
        print(f"\nAll {total_points} sweep points failed!", file=sys.stderr)
        return 1
    if successful_points < total_points:
        failed_count = total_points - successful_points
        print(
            f"\nWarning: {failed_count}/{total_points} sweep points "
            "encountered failures."
        )
        return 2

    print(f"\nAll {total_points} sweep points completed successfully.")
    return 0


def main() -> None:
    """CLI entrypoint."""
    parser = create_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
