"""E007 — prefix-caching mechanism validation and four-condition comparison."""

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

import httpx

from inference_os.config import BenchmarkConfig, load_config
from inference_os.metrics.summary import calculate_metric_stats
from inference_os.reports.plots import generate_e007_plots
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

DEFAULT_CONFIG = "configs/e007_cache_off_unique.yaml"
EXPECTED_CONDITIONS = {
    "unique_prefix_cache_off",
    "unique_prefix_cache_on",
    "shared_prefix_cache_off",
    "shared_prefix_cache_on",
}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E007 — Prefix-Caching Mechanism Validation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--config",
        default=None,
        help="Run one condition against the currently running vLLM server",
    )
    mode.add_argument(
        "--summarize",
        nargs=4,
        metavar="RUN_DIR",
        help="Combine the four isolated condition run directories",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run 6 measured requests and 2 disjoint warm-up requests",
    )
    parser.add_argument("--base-url", default=None, help="Override vLLM base URL")
    parser.add_argument("--output-dir", default=None, help="Override output root")
    return parser


def load_e007_config(path: Path | str) -> BenchmarkConfig:
    loaded = load_config(path)
    if not isinstance(loaded, BenchmarkConfig):
        raise ValueError("E007 requires a single BenchmarkConfig")
    if loaded.experiment_id != "E007":
        raise ValueError("E007 config must use experiment_id E007")
    if loaded.workload is None:
        raise ValueError("E007 config requires a workload")
    reuse = loaded.workload.prompt_reuse
    if reuse.mode not in {"unique_prefix", "shared_prefix"}:
        raise ValueError("E007 requires unique_prefix or shared_prefix mode")
    if len(loaded.workload.input_tokens.values) != 1:
        raise ValueError("E007 requires a fixed input length")
    if len(loaded.workload.max_output_tokens.values) != 1:
        raise ValueError("E007 requires a fixed output limit")
    if loaded.concurrency != 1:
        raise ValueError("E007 mechanism validation requires concurrency 1")
    return loaded


def condition_name(config: BenchmarkConfig) -> str:
    assert config.workload is not None
    cache = "on" if config.enable_prefix_caching else "off"
    return f"{config.workload.prompt_reuse.mode}_cache_{cache}"


async def execute_e007_condition(
    config: BenchmarkConfig,
    *,
    tokenizer: Tokenizer | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Run and persist one isolated E007 condition."""
    assert config.workload is not None
    reuse = config.workload.prompt_reuse
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
        raise RuntimeError("E007 did not capture the measured-window start")
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
    mechanism_checks = _mechanism_checks(config, delta.to_dict())
    condition = {
        "experiment_id": "E007",
        "condition": condition_name(config),
        "success": (
            result.summary.total_requests == config.num_requests
            and result.summary.failed_requests == 0
        ),
        "mechanism_valid": all(mechanism_checks.values()),
        "mechanism_checks": mechanism_checks,
        "run_dir": str(run_dir),
        "server_cache_config": asdict(live_config),
        "benchmark": asdict(result.summary),
        "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
        "cache": delta.to_dict(),
        "cache_phase_latency": phase_latency,
        "workload": loaded["summary"]["workload"],
    }
    (run_dir / "e007_condition.json").write_text(
        json.dumps(condition, indent=2), encoding="utf-8"
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["e007"] = {
        key: condition[key]
        for key in (
            "condition",
            "mechanism_valid",
            "mechanism_checks",
            "cache_phase_latency",
        )
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, condition


def summarize_e007_runs(
    run_dirs: list[Path | str],
    *,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate and combine four separately executed E007 conditions."""
    conditions: list[dict[str, Any]] = []
    for raw_dir in run_dirs:
        path = Path(raw_dir)
        condition_path = path / "e007_condition.json"
        if not condition_path.is_file():
            raise ValueError(f"missing E007 condition artifact: {condition_path}")
        condition = json.loads(condition_path.read_text(encoding="utf-8"))
        if condition.get("experiment_id") != "E007":
            raise ValueError(f"not an E007 condition: {condition_path}")
        conditions.append(condition)

    observed = {str(item["condition"]) for item in conditions}
    if observed != EXPECTED_CONDITIONS or len(conditions) != 4:
        missing = sorted(EXPECTED_CONDITIONS - observed)
        unexpected = sorted(observed - EXPECTED_CONDITIONS)
        raise ValueError(
            f"E007 requires exactly four conditions; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if len(observed) != len(conditions):
        raise ValueError("E007 condition directories contain a duplicate")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_dir = output_root / f"E007_{timestamp}_{uuid.uuid4().hex[:8]}"
    comparison_dir.mkdir(parents=True, exist_ok=False)
    by_name = {str(item["condition"]): item for item in conditions}
    comparisons = {
        "unique_prefix_ttft_p50_change_percent": _relative_change_percent(
            _ttft_p50(by_name["unique_prefix_cache_off"]),
            _ttft_p50(by_name["unique_prefix_cache_on"]),
        ),
        "shared_prefix_ttft_p50_change_percent": _relative_change_percent(
            _ttft_p50(by_name["shared_prefix_cache_off"]),
            _ttft_p50(by_name["shared_prefix_cache_on"]),
        ),
        "shared_prefix_steady_state_ttft_p50_change_percent": (
            _relative_change_percent(
                _phase_ttft_p50(by_name["shared_prefix_cache_off"], "disabled"),
                _phase_ttft_p50(by_name["shared_prefix_cache_on"], "warm"),
            )
        ),
    }
    all_success = all(bool(item["success"]) for item in conditions)
    all_mechanisms_valid = all(bool(item["mechanism_valid"]) for item in conditions)
    summary = {
        "experiment_id": "E007",
        "status": "SUCCESS" if all_success and all_mechanisms_valid else "INVALID",
        "all_requests_successful": all_success,
        "all_mechanism_checks_passed": all_mechanisms_valid,
        "comparisons": comparisons,
        "conditions": conditions,
    }
    (comparison_dir / "e007_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    generate_e007_plots(conditions, comparison_dir / "plots")
    return comparison_dir, summary


def _mechanism_checks(
    config: BenchmarkConfig, cache: dict[str, Any]
) -> dict[str, bool]:
    assert config.workload is not None
    mode = config.workload.prompt_reuse.mode
    queries = cache["prefix_cache_query_tokens"]
    hits = cache["prefix_cache_hit_tokens"]
    checks = {"request_plan_valid": True, "server_config_verified": True}
    if config.enable_prefix_caching:
        checks["cache_queries_observed"] = queries is not None and queries > 0
        if mode == "shared_prefix":
            checks["cache_hits_observed"] = hits is not None and hits > 0
        else:
            checks["no_unique_prefix_hits"] = hits == 0
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
    stats = condition["benchmark"].get("ttft_stats") or {}
    return float(stats.get("p50", 0.0))


def _phase_ttft_p50(condition: dict[str, Any], state: str) -> float:
    stats = condition.get("cache_phase_latency", {}).get(state) or {}
    return float(stats.get("p50", 0.0))


def _relative_change_percent(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def print_condition(condition: dict[str, Any]) -> None:
    benchmark = condition["benchmark"]
    ttft = benchmark.get("ttft_stats") or {}
    cache = condition["cache"]
    print("=" * 78)
    print(f" E007 CONDITION: {condition['condition']}")
    print("=" * 78)
    print(f" Requests successful: {benchmark['successful_requests']}")
    print(f" TTFT P50:           {ttft.get('p50', 0.0) * 1000:.2f} ms")
    print(f" TTFT P95:           {ttft.get('p95', 0.0) * 1000:.2f} ms")
    print(f" Cache query tokens: {cache['prefix_cache_query_tokens']}")
    print(f" Cache hit tokens:   {cache['prefix_cache_hit_tokens']}")
    print(f" Hit fraction:       {cache['observed_prefix_cache_hit_fraction']}")
    print(f" Mechanism valid:    {condition['mechanism_valid']}")
    print("=" * 78)


async def main_async(args: argparse.Namespace) -> int:
    if args.summarize:
        try:
            comparison_dir, summary = summarize_e007_runs(
                args.summarize,
                output_root=Path(args.output_dir or "runs"),
            )
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"E007 comparison status: {summary['status']}")
        print(f"Artifacts: {comparison_dir}")
        return 0 if summary["status"] == "SUCCESS" else 2

    try:
        config = load_e007_config(args.config or DEFAULT_CONFIG)
        if args.base_url:
            config = replace(config, base_url=args.base_url)
        if args.output_dir:
            config = replace(config, output_dir=args.output_dir)
        if args.pilot:
            config = replace(config, num_requests=6, warmup_requests=2)
        run_dir, condition = await execute_e007_condition(config)
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
