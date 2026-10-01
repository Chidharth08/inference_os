"""E008 — fixed-length shared-prefix fraction sensitivity experiment."""

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

import httpx

from inference_os.config import BenchmarkConfig, load_config
from inference_os.metrics.summary import calculate_metric_stats
from inference_os.reports.plots import generate_e008_plots
from inference_os.results.persistence import load_benchmark_run
from inference_os.runner.engine import execute_benchmark
from inference_os.telemetry.vllm_metrics import (
    VLLMMetricsSnapshot,
    compute_prefix_cache_delta,
    fetch_vllm_metrics,
    persist_cache_metric_artifacts,
    validate_cache_server_config,
    validate_pristine_cache_state,
)
from inference_os.workloads.base import Tokenizer
from inference_os.workloads.spec import PromptReuseConfig, WorkloadConfig

REQUESTED_FRACTIONS = (0, 25, 50, 75, 90)
DEFAULT_OFF_CONFIG = "configs/e008_cache_off.yaml"
DEFAULT_ON_CONFIG = "configs/e008_cache_on.yaml"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E008 — Prefix Reuse Sensitivity",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--config",
        help="Run one isolated fraction/cache condition",
    )
    mode.add_argument(
        "--summarize",
        nargs="+",
        metavar="RUN_DIR",
        help="Combine the ten canonical condition directories",
    )
    parser.add_argument(
        "--shared-fraction",
        type=int,
        choices=REQUESTED_FRACTIONS,
        help="Requested reusable prompt percentage for condition mode",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 6 measured requests and 2 disjoint warm-ups",
    )
    parser.add_argument("--base-url", default=None, help="Override vLLM base URL")
    parser.add_argument("--output-dir", default=None, help="Override output root")
    return parser


def load_e008_config(path: Path | str) -> BenchmarkConfig:
    loaded = load_config(path)
    if not isinstance(loaded, BenchmarkConfig):
        raise ValueError("E008 requires a single BenchmarkConfig")
    if loaded.experiment_id != "E008":
        raise ValueError("E008 config must use experiment_id E008")
    if loaded.workload is None:
        raise ValueError("E008 config requires a workload")
    if len(loaded.workload.input_tokens.values) != 1:
        raise ValueError("E008 requires a fixed input length")
    if len(loaded.workload.max_output_tokens.values) != 1:
        raise ValueError("E008 requires a fixed output limit")
    if loaded.concurrency != 1:
        raise ValueError("E008 sensitivity validation requires concurrency 1")
    if loaded.enable_chunked_prefill:
        raise ValueError("E008 requires chunked prefill disabled")
    return loaded


def build_fraction_config(
    base: BenchmarkConfig,
    requested_fraction_percent: int,
) -> tuple[BenchmarkConfig, dict[str, float | int]]:
    """Resolve a requested fraction to a cache-block-aligned prompt boundary."""
    if requested_fraction_percent not in REQUESTED_FRACTIONS:
        raise ValueError(f"shared fraction must be one of {list(REQUESTED_FRACTIONS)}")
    assert base.workload is not None
    total_tokens = base.workload.input_tokens.values[0]
    block_size = base.workload.prompt_reuse.cache_block_size_tokens
    requested_tokens = total_tokens * requested_fraction_percent / 100.0
    resolved_tokens = int(math.floor(requested_tokens / block_size) * block_size)
    if requested_fraction_percent > 0 and resolved_tokens <= 0:
        raise ValueError("requested fraction is smaller than one cache block")
    if resolved_tokens >= total_tokens:
        raise ValueError("shared fraction must leave a positive unique suffix")

    cache_name = "on" if base.enable_prefix_caching else "off"
    mode = "unique_prefix" if resolved_tokens == 0 else "shared_prefix"
    reuse = PromptReuseConfig(
        mode=mode,
        shared_prefix_tokens=resolved_tokens,
        reuse_group_id=f"e008-prefix-{resolved_tokens}",
        cache_block_size_tokens=block_size,
    )
    workload = WorkloadConfig(
        name=f"e008_fraction_{requested_fraction_percent:03d}_cache_{cache_name}",
        input_tokens=base.workload.input_tokens,
        max_output_tokens=base.workload.max_output_tokens,
        prompt_reuse=reuse,
        sampling_mode=base.workload.sampling_mode,
    )
    metadata: dict[str, float | int] = {
        "requested_shared_fraction_percent": requested_fraction_percent,
        "requested_shared_prefix_tokens": requested_tokens,
        "resolved_shared_prefix_tokens": resolved_tokens,
        "resolved_shared_fraction": resolved_tokens / total_tokens,
        "total_prompt_tokens": total_tokens,
        "cache_block_size_tokens": block_size,
    }
    return replace(base, workload=workload), metadata


def condition_name(config: BenchmarkConfig, requested_fraction_percent: int) -> str:
    cache = "on" if config.enable_prefix_caching else "off"
    return f"fraction_{requested_fraction_percent:03d}_cache_{cache}"


async def execute_e008_condition(
    config: BenchmarkConfig,
    fraction_metadata: dict[str, float | int],
    *,
    tokenizer: Tokenizer | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run and persist one isolated E008 fraction/cache condition."""
    assert config.workload is not None
    reuse = config.workload.prompt_reuse
    requested_fraction = int(fraction_metadata["requested_shared_fraction_percent"])
    preflight = await fetch_vllm_metrics(config.base_url, client=client)
    live_config = validate_cache_server_config(
        preflight,
        expected_enabled=config.enable_prefix_caching,
        expected_block_size=reuse.cache_block_size_tokens,
    )
    if config.enable_prefix_caching:
        validate_pristine_cache_state(preflight)

    before_box: list[VLLMMetricsSnapshot] = []

    async def capture_measured_window_start() -> None:
        before_box.append(await fetch_vllm_metrics(config.base_url, client=client))

    run_dir, result, gpu_summary = await execute_benchmark(
        config,
        tokenizer=tokenizer,
        client=client,
        after_warmup_hook=capture_measured_window_start,
    )
    if len(before_box) != 1:
        raise RuntimeError("E008 did not capture the measured-window start")
    after = await fetch_vllm_metrics(config.base_url, client=client)
    delta = compute_prefix_cache_delta(before_box[0], after)
    persist_cache_metric_artifacts(
        run_dir,
        preflight=preflight,
        before=before_box[0],
        after=after,
        delta=delta,
    )

    loaded = load_benchmark_run(run_dir)
    phase_latency = _summarize_cache_phases(loaded["requests"], loaded["workload"])
    reuse_observation = _summarize_reuse_plan(
        loaded["workload"],
        block_size=reuse.cache_block_size_tokens,
        total_prompt_tokens=int(fraction_metadata["total_prompt_tokens"]),
    )
    mechanism_checks = _mechanism_checks(
        config,
        delta.to_dict(),
        expected_hit_tokens=int(reuse_observation["expected_cache_hit_tokens"]),
    )
    condition = {
        "experiment_id": "E008",
        "condition": condition_name(config, requested_fraction),
        "success": (
            result.summary.total_requests == config.num_requests
            and result.summary.failed_requests == 0
        ),
        "mechanism_valid": all(mechanism_checks.values()),
        "mechanism_checks": mechanism_checks,
        "run_dir": str(run_dir),
        "fraction": fraction_metadata,
        "reuse_plan": reuse_observation,
        "server_cache_config": asdict(live_config),
        "benchmark": asdict(result.summary),
        "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
        "cache": delta.to_dict(),
        "cache_phase_latency": phase_latency,
        "workload": loaded["summary"]["workload"],
    }
    (run_dir / "e008_condition.json").write_text(
        json.dumps(condition, indent=2), encoding="utf-8"
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["e008"] = {
        key: condition[key]
        for key in (
            "condition",
            "fraction",
            "reuse_plan",
            "mechanism_valid",
            "mechanism_checks",
            "cache_phase_latency",
        )
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, condition


def summarize_e008_runs(
    run_dirs: list[Path | str],
    *,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate and combine all five paired cache-OFF/cache-ON conditions."""
    conditions: list[dict[str, Any]] = []
    for raw_dir in run_dirs:
        path = Path(raw_dir)
        condition_path = path / "e008_condition.json"
        if not condition_path.is_file():
            raise ValueError(f"missing E008 condition artifact: {condition_path}")
        condition = json.loads(condition_path.read_text(encoding="utf-8"))
        if condition.get("experiment_id") != "E008":
            raise ValueError(f"not an E008 condition: {condition_path}")
        conditions.append(condition)

    expected = {
        f"fraction_{fraction:03d}_cache_{cache}"
        for fraction in REQUESTED_FRACTIONS
        for cache in ("off", "on")
    }
    observed = {str(item["condition"]) for item in conditions}
    if observed != expected or len(conditions) != len(expected):
        raise ValueError(
            "E008 requires exactly ten paired conditions; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )
    if len(observed) != len(conditions):
        raise ValueError("E008 condition directories contain a duplicate")

    by_name = {str(item["condition"]): item for item in conditions}
    points: list[dict[str, Any]] = []
    for fraction in REQUESTED_FRACTIONS:
        off = by_name[f"fraction_{fraction:03d}_cache_off"]
        on = by_name[f"fraction_{fraction:03d}_cache_on"]
        off_ttft = _ttft_p50(off)
        on_ttft = _ttft_p50(on)
        warm_state = "no_cacheable_prefix" if fraction == 0 else "warm"
        points.append(
            {
                "requested_shared_fraction_percent": fraction,
                "resolved_shared_prefix_tokens": on["fraction"][
                    "resolved_shared_prefix_tokens"
                ],
                "resolved_shared_fraction": on["fraction"]["resolved_shared_fraction"],
                "observed_cache_hit_fraction": on["cache"].get(
                    "observed_prefix_cache_hit_fraction"
                ),
                "steady_state_observed_hit_fraction": _steady_state_hit_fraction(on),
                "ttft_p50_cache_off_seconds": off_ttft,
                "ttft_p50_cache_on_seconds": on_ttft,
                "ttft_p50_change_seconds": on_ttft - off_ttft,
                "ttft_p50_change_percent": _relative_change_percent(off_ttft, on_ttft),
                "steady_state_ttft_p50_change_percent": _relative_change_percent(
                    _phase_ttft_p50(off, "disabled"),
                    _phase_ttft_p50(on, warm_state),
                ),
                "cache_off": off,
                "cache_on": on,
            }
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_dir = output_root / f"E008_{timestamp}_{uuid.uuid4().hex[:8]}"
    comparison_dir.mkdir(parents=True, exist_ok=False)
    all_success = all(bool(item["success"]) for item in conditions)
    all_mechanisms_valid = all(bool(item["mechanism_valid"]) for item in conditions)
    monotonic_hits = _nondecreasing(
        [float(point["observed_cache_hit_fraction"] or 0.0) for point in points]
    )
    summary = {
        "experiment_id": "E008",
        "status": "SUCCESS" if all_success and all_mechanisms_valid else "INVALID",
        "all_requests_successful": all_success,
        "all_mechanism_checks_passed": all_mechanisms_valid,
        "observed_hit_fraction_nondecreasing": monotonic_hits,
        "points": points,
    }
    (comparison_dir / "e008_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    generate_e008_plots(points, comparison_dir / "plots")
    return comparison_dir, summary


def _summarize_reuse_plan(
    workload: list[dict[str, Any]],
    *,
    block_size: int,
    total_prompt_tokens: int,
) -> dict[str, Any]:
    measured = [item for item in workload if not bool(item.get("is_warmup"))]
    reusable = [int(item.get("actual_reusable_prefix_tokens", 0)) for item in measured]
    cacheable = [(value // block_size) * block_size for value in reusable]
    steady = cacheable[1:]
    return {
        "measured_requests": len(measured),
        "actual_reusable_prefix_tokens_min": min(reusable, default=0),
        "actual_reusable_prefix_tokens_max": max(reusable, default=0),
        "steady_state_cacheable_prefix_tokens_min": min(steady, default=0),
        "steady_state_cacheable_prefix_tokens_max": max(steady, default=0),
        "expected_cache_hit_tokens": sum(cacheable),
        "expected_measured_hit_fraction": (
            sum(cacheable) / (len(measured) * total_prompt_tokens) if measured else None
        ),
    }


def _mechanism_checks(
    config: BenchmarkConfig,
    cache: dict[str, Any],
    *,
    expected_hit_tokens: int,
) -> dict[str, bool]:
    queries = cache["prefix_cache_query_tokens"]
    hits = cache["prefix_cache_hit_tokens"]
    checks = {"request_plan_valid": True, "server_config_verified": True}
    if config.enable_prefix_caching:
        checks["cache_queries_observed"] = queries is not None and queries > 0
        checks["cache_hits_match_plan"] = hits == expected_hit_tokens
    else:
        checks["no_cache_hits_when_disabled"] = hits in {None, 0}
    return checks


def _summarize_cache_phases(
    requests: list[dict[str, Any]], workload: list[dict[str, Any]]
) -> dict[str, Any]:
    request_by_id = {
        str(item["request_id"]): item
        for item in requests
        if not bool(item.get("is_warmup"))
    }
    values: dict[str, list[float]] = {}
    for item in workload:
        if bool(item.get("is_warmup")):
            continue
        request = request_by_id.get(str(item["request_id"]))
        if request is None or request.get("ttft_seconds") is None:
            continue
        state = str(item.get("expected_cache_state", "unknown"))
        values.setdefault(state, []).append(float(request["ttft_seconds"]))
    return {
        state: asdict(stats)
        for state, samples in values.items()
        if (stats := calculate_metric_stats(samples)) is not None
    }


def _ttft_p50(condition: dict[str, Any]) -> float:
    return float((condition["benchmark"].get("ttft_stats") or {}).get("p50", 0.0))


def _phase_ttft_p50(condition: dict[str, Any], state: str) -> float:
    return float(
        (condition.get("cache_phase_latency", {}).get(state) or {}).get("p50", 0.0)
    )


def _steady_state_hit_fraction(condition: dict[str, Any]) -> float | None:
    requests = int(condition["reuse_plan"]["measured_requests"])
    total = int(condition["fraction"]["total_prompt_tokens"])
    hits = condition["cache"].get("prefix_cache_hit_tokens")
    if hits is None or requests <= 1:
        return None
    return float(hits) / ((requests - 1) * total)


def _relative_change_percent(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def _nondecreasing(values: list[float]) -> bool:
    return all(left <= right for left, right in zip(values, values[1:]))


def print_condition(condition: dict[str, Any]) -> None:
    benchmark = condition["benchmark"]
    ttft = benchmark.get("ttft_stats") or {}
    cache = condition["cache"]
    fraction = condition["fraction"]
    print("=" * 78)
    print(f" E008 CONDITION: {condition['condition']}")
    print("=" * 78)
    print(f" Requested fraction: {fraction['requested_shared_fraction_percent']}%")
    print(f" Resolved prefix:    {fraction['resolved_shared_prefix_tokens']} tokens")
    print(f" Requests successful: {benchmark['successful_requests']}")
    print(f" TTFT P50:            {ttft.get('p50', 0.0) * 1000:.2f} ms")
    print(f" TTFT P95:            {ttft.get('p95', 0.0) * 1000:.2f} ms")
    print(f" Cache query tokens:  {cache['prefix_cache_query_tokens']}")
    print(f" Cache hit tokens:    {cache['prefix_cache_hit_tokens']}")
    print(f" Hit fraction:        {cache['observed_prefix_cache_hit_fraction']}")
    print(f" Mechanism valid:     {condition['mechanism_valid']}")
    print("=" * 78)


async def main_async(args: argparse.Namespace) -> int:
    if args.summarize:
        try:
            comparison_dir, summary = summarize_e008_runs(
                args.summarize,
                output_root=Path(args.output_dir or "runs"),
            )
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"E008 comparison status: {summary['status']}")
        print(f"Artifacts: {comparison_dir}")
        return 0 if summary["status"] == "SUCCESS" else 2

    if args.shared_fraction is None:
        print("Error: --shared-fraction is required with --config", file=sys.stderr)
        return 1
    try:
        config = load_e008_config(args.config)
        config, fraction_metadata = build_fraction_config(config, args.shared_fraction)
        if args.base_url:
            config = replace(config, base_url=args.base_url)
        if args.output_dir:
            config = replace(config, output_dir=args.output_dir)
        if args.pilot:
            config = replace(config, num_requests=6, warmup_requests=2)
        run_dir, condition = await execute_e008_condition(config, fraction_metadata)
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_condition(condition)
    print(f"Artifacts: {run_dir}")
    if not condition["success"]:
        return 1
    return 0 if condition["mechanism_valid"] else 2


def main() -> None:
    args = create_parser().parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
