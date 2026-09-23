"""E003 — synthetic application-shaped workload comparison."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference_os.config import BenchmarkConfig, load_config
from inference_os.reports.plots import generate_e003_plots
from inference_os.results.persistence import load_benchmark_run
from inference_os.runner.engine import execute_benchmark

DEFAULT_CONFIGS = (
    "configs/e003_chat_like.yaml",
    "configs/e003_rag_like.yaml",
    "configs/e003_summarization_like.yaml",
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E003 — Application-Shaped Workload Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help="Profile configuration files to compare",
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
            raise ValueError(f"E003 profile must be a BenchmarkConfig: {raw_path}")
        if loaded.workload is None:
            raise ValueError(f"E003 profile is missing a workload section: {raw_path}")
        configs.append(loaded)
    if len(configs) < 2:
        raise ValueError("E003 requires at least two workload profiles")
    profile_names = [config.workload.name for config in configs if config.workload]
    if len(profile_names) != len(set(profile_names)):
        raise ValueError("E003 workload profile names must be unique")
    _validate_controlled_fields(configs)
    return configs


def _validate_controlled_fields(configs: list[BenchmarkConfig]) -> None:
    """Reject comparisons that vary controls other than workload shape."""
    fields = (
        "model",
        "base_url",
        "num_requests",
        "warmup_requests",
        "seed",
        "temperature",
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
                "E003 controlled fields differ across profiles: "
                + ", ".join(mismatches)
            )


async def execute_e003(
    configs: list[BenchmarkConfig],
    *,
    output_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_dir = output_root / f"E003_{timestamp}_{uuid.uuid4().hex[:8]}"
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
                "success": result.summary.successful_requests > 0,
                "run_dir": str(run_dir),
                "benchmark": asdict(result.summary),
                "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
                "workload": loaded_run["summary"]["workload"],
            }
        )

    successful_profiles = sum(item["success"] for item in profile_results)
    status = (
        "SUCCESS"
        if successful_profiles == len(profile_results)
        else ("PARTIAL" if successful_profiles else "FAILED")
    )
    summary = {
        "experiment_id": "E003",
        "status": status,
        "profile_count": len(profile_results),
        "successful_profiles": successful_profiles,
        "profiles": profile_results,
    }
    (comparison_dir / "e003_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if successful_profiles:
        generate_e003_plots(profile_results, comparison_dir / "plots")
    return comparison_dir, profile_results


def print_results(results: list[dict[str, Any]]) -> None:
    print("=" * 113)
    print(" E003: SYNTHETIC APPLICATION-SHAPED WORKLOADS")
    print("=" * 113)
    print(
        f"{'Profile':<20} | {'Req/s':>8} | {'Input tok/s':>12} | "
        f"{'Output tok/s':>13} | {'TTFT P95':>10} | {'E2E P95':>10} | "
        f"{'Errors':>7}"
    )
    print("-" * 113)
    for item in results:
        benchmark = item["benchmark"]
        ttft = benchmark.get("ttft_stats") or {}
        e2e = benchmark.get("e2e_latency_stats") or {}
        print(
            f"{item['profile_name']:<20} | "
            f"{benchmark['request_throughput']:>8.2f} | "
            f"{benchmark['input_token_throughput']:>12.2f} | "
            f"{benchmark['output_token_throughput']:>13.2f} | "
            f"{ttft.get('p95', 0.0) * 1000:>8.2f} ms | "
            f"{e2e.get('p95', 0.0) * 1000:>8.2f} ms | "
            f"{benchmark['error_rate'] * 100:>6.1f}%"
        )
    print("=" * 113)


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
    comparison_dir, results = await execute_e003(configs, output_root=output_root)
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
