"""E004 — fixed versus variable workload comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_os.config import BenchmarkConfig, load_config
from inference_os.metrics import summarize_length_buckets
from inference_os.reports.plots import generate_e004_plots
from inference_os.results.persistence import load_benchmark_run
from inference_os.runner.engine import execute_benchmark

DEFAULT_CONFIGS = (
    "configs/e004_fixed.yaml",
    "configs/e004_variable.yaml",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E004 — Fixed versus Variable Workloads",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Fixed and variable workload configuration files",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 12 measured requests per profile as a smoke test",
    )
    parser.add_argument("--base-url", default=None, help="Override vLLM base URL")
    parser.add_argument("--output-dir", default=None, help="Override output root")
    return parser


def load_profile_configs(paths: list[str]) -> list[BenchmarkConfig]:
    configs: list[BenchmarkConfig] = []
    for raw_path in paths:
        loaded = load_config(Path(raw_path))
        if not isinstance(loaded, BenchmarkConfig):
            raise ValueError(f"E004 profile must be a BenchmarkConfig: {raw_path}")
        if loaded.workload is None:
            raise ValueError(f"E004 profile is missing a workload section: {raw_path}")
        configs.append(loaded)

    if len(configs) != 2:
        raise ValueError("E004 requires exactly two workload profiles")
    profile_names = [config.workload.name for config in configs if config.workload]
    if len(profile_names) != len(set(profile_names)):
        raise ValueError("E004 workload profile names must be unique")
    _validate_controlled_fields(configs)
    _validate_workload_comparison(configs)
    return configs


def _validate_controlled_fields(configs: list[BenchmarkConfig]) -> None:
    """Reject comparisons that vary controls other than token-length variance."""
    fields = (
        "model",
        "base_url",
        "num_requests",
        "warmup_requests",
        "seed",
        "temperature",
        "request_timeout_seconds",
        "concurrency",
        "telemetry_interval_seconds",
        "device_index",
        "enable_prefix_caching",
        "enable_chunked_prefill",
    )
    baseline = configs[0]
    for config in configs[1:]:
        mismatches = [
            field
            for field in fields
            if getattr(config, field) != getattr(baseline, field)
        ]
        if mismatches:
            raise ValueError(
                "E004 controlled fields differ across profiles: "
                + ", ".join(mismatches)
            )


def _validate_workload_comparison(configs: list[BenchmarkConfig]) -> None:
    workloads = [config.workload for config in configs]
    assert all(workload is not None for workload in workloads)
    concrete = [workload for workload in workloads if workload is not None]

    fixed_count = sum(
        len(workload.input_tokens.values) == 1
        and len(workload.max_output_tokens.values) == 1
        for workload in concrete
    )
    if fixed_count != 1:
        raise ValueError("E004 requires one fixed and one variable workload")
    if not all(workload.sampling_mode == "stratified" for workload in concrete):
        raise ValueError("E004 profiles must use stratified sampling")

    input_means = [workload.input_tokens.mean for workload in concrete]
    output_means = [workload.max_output_tokens.mean for workload in concrete]
    if not math.isclose(input_means[0], input_means[1], rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("E004 configured input-token means must match")
    if not math.isclose(output_means[0], output_means[1], rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("E004 configured output-token means must match")


async def execute_e004(
    configs: list[BenchmarkConfig],
    *,
    output_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_dir = output_root / f"E004_{timestamp}_{uuid.uuid4().hex[:8]}"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    from inference_os.workloads.hf_tokenizer import HFTokenizer

    tokenizer = HFTokenizer.from_pretrained(configs[0].model)
    profile_results: list[dict[str, Any]] = []

    for config in configs:
        assert config.workload is not None
        profile_name = config.workload.name
        profile_config = replace(
            config,
            output_dir=str(comparison_dir / profile_name),
        )
        run_dir, result, gpu_summary = await execute_benchmark(
            profile_config,
            tokenizer=tokenizer,
        )
        loaded_run = load_benchmark_run(run_dir)
        profile_results.append(
            {
                "profile_name": profile_name,
                "success": (
                    result.summary.total_requests == profile_config.num_requests
                    and result.summary.failed_requests == 0
                ),
                "run_dir": str(run_dir),
                "benchmark": asdict(result.summary),
                "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
                "workload": loaded_run["summary"]["workload"],
                "input_length_buckets": summarize_length_buckets(
                    loaded_run["requests"],
                    loaded_run["workload"],
                    bucket_field="target_input_tokens",
                ),
                "output_length_buckets": summarize_length_buckets(
                    loaded_run["requests"],
                    loaded_run["workload"],
                    bucket_field="max_output_tokens",
                ),
            }
        )

    successful_profiles = sum(item["success"] for item in profile_results)
    status = (
        "SUCCESS"
        if successful_profiles == len(profile_results)
        else ("PARTIAL" if successful_profiles else "FAILED")
    )
    summary = {
        "experiment_id": "E004",
        "status": status,
        "profile_count": len(profile_results),
        "successful_profiles": successful_profiles,
        "controlled_configured_means": {
            "target_input_tokens": configs[0].workload.input_tokens.mean,
            "max_output_tokens": configs[0].workload.max_output_tokens.mean,
        },
        "profiles": profile_results,
    }
    (comparison_dir / "e004_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if successful_profiles:
        generate_e004_plots(profile_results, comparison_dir / "plots")
    return comparison_dir, profile_results


def print_results(results: list[dict[str, Any]]) -> None:
    print("=" * 133)
    print(" E004: FIXED VERSUS VARIABLE WORKLOADS")
    print("=" * 133)
    print(
        f"{'Profile':<12} | {'In CV':>7} | {'Out CV':>7} | {'Req/s':>8} | "
        f"{'TTFT P50':>10} | {'TTFT P95':>10} | {'TTFT P99':>10} | "
        f"{'E2E P50':>10} | {'E2E P95':>10} | {'E2E P99':>10} | {'Errors':>7}"
    )
    print("-" * 133)
    for item in results:
        benchmark = item["benchmark"]
        workload = item["workload"]
        configured = workload.get("configured_distributions", {})
        input_cv = configured.get("target_input_tokens", {}).get(
            "coefficient_of_variation", 0.0
        )
        output_cv = configured.get("max_output_tokens", {}).get(
            "coefficient_of_variation", 0.0
        )
        ttft = benchmark.get("ttft_stats") or {}
        e2e = benchmark.get("e2e_latency_stats") or {}
        print(
            f"{item['profile_name']:<12} | {input_cv:>7.3f} | {output_cv:>7.3f} | "
            f"{benchmark['request_throughput']:>8.2f} | "
            f"{ttft.get('p50', 0.0) * 1000:>8.2f} ms | "
            f"{ttft.get('p95', 0.0) * 1000:>8.2f} ms | "
            f"{ttft.get('p99', 0.0) * 1000:>8.2f} ms | "
            f"{e2e.get('p50', 0.0):>8.2f} s | "
            f"{e2e.get('p95', 0.0):>8.2f} s | "
            f"{e2e.get('p99', 0.0):>8.2f} s | "
            f"{benchmark['error_rate'] * 100:>6.1f}%"
        )
    print("=" * 133)


async def main_async(args: argparse.Namespace) -> int:
    try:
        configs = load_profile_configs(args.configs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.base_url:
        configs = [replace(config, base_url=args.base_url) for config in configs]
    if args.pilot:
        configs = [
            replace(config, num_requests=12, warmup_requests=2) for config in configs
        ]

    output_root = Path(args.output_dir or configs[0].output_dir)
    comparison_dir, results = await execute_e004(configs, output_root=output_root)
    print_results(results)
    print(f"Artifacts: {comparison_dir}")

    successful = sum(item["success"] for item in results)
    if successful == len(results):
        return 0
    return 2 if successful else 1


def main() -> None:
    args = create_parser().parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
