"""E009 — application-shaped prefix reuse and selected load comparison."""

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

from inference_os.config import BenchmarkConfig, SLOConfig, load_config
from inference_os.metrics.summary import calculate_metric_stats
from inference_os.reports.plots import generate_e009_load_plots, generate_e009_plots
from inference_os.results.persistence import load_benchmark_run
from inference_os.runner.engine import execute_benchmark
from inference_os.runner.load_engine import execute_open_loop_benchmark
from inference_os.telemetry.vllm_metrics import (
    VLLMMetricsSnapshot,
    compute_prefix_cache_delta,
    fetch_vllm_metrics,
    persist_cache_metric_artifacts,
    validate_cache_server_config,
    validate_pristine_cache_state,
)
from inference_os.workloads.base import Tokenizer
from inference_os.workloads.spec import (
    PromptReuseConfig,
    TokenLengthDistribution,
    WorkloadConfig,
)

PROFILES = (
    "chat_like",
    "rag_friendly",
    "rag_hostile",
    "summarization_like",
    "agent_like",
)
LOAD_PROFILES = ("rag_friendly", "rag_hostile")
DEFAULT_LOAD_RATE = 1.0
DEFAULT_SLO = SLOConfig(
    percentile=95.0,
    max_ttft_seconds=1.0,
    max_e2e_latency_seconds=3.0,
    max_error_rate=0.01,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E009 — Application-Shaped Prefix Reuse",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="Run one isolated E009 condition")
    mode.add_argument(
        "--summarize", nargs="+", metavar="RUN_DIR", help="Combine 10 controlled runs"
    )
    mode.add_argument(
        "--summarize-load",
        nargs="+",
        metavar="RUN_DIR",
        help="Combine four selected open-loop runs",
    )
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument(
        "--load-rate",
        type=float,
        default=None,
        help="Run selected profile under deterministic open-loop load",
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser


def load_e009_config(path: Path | str) -> BenchmarkConfig:
    loaded = load_config(path)
    if not isinstance(loaded, BenchmarkConfig):
        raise ValueError("E009 requires a single BenchmarkConfig")
    if loaded.experiment_id != "E009" or loaded.workload is None:
        raise ValueError("E009 requires experiment_id E009 and a workload")
    if loaded.concurrency != 1:
        raise ValueError("E009 controlled conditions require concurrency 1")
    if loaded.enable_chunked_prefill:
        raise ValueError("E009 requires chunked prefill disabled")
    return loaded


def build_profile_config(base: BenchmarkConfig, profile: str) -> BenchmarkConfig:
    """Create one deterministic 24-request application relationship plan."""
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {list(PROFILES)}")
    assert base.workload is not None
    cache = "on" if base.enable_prefix_caching else "off"
    block_size = base.workload.prompt_reuse.cache_block_size_tokens
    if profile == "chat_like":
        application_profile = "chat_like"
        layout = "default"
        groups = 4
        stable = 512
        document = 0
        inputs = tuple(
            value
            for value in (1024, 1536, 2048, 2560, 3072, 3584)
            for _ in range(groups)
        )
        outputs = (64,) * 24
    elif profile in {"rag_friendly", "rag_hostile"}:
        application_profile = "rag_like"
        layout = "friendly" if profile == "rag_friendly" else "hostile"
        groups = 1
        stable = 3328 if layout == "friendly" else 256
        document = 3072
        inputs = (4096,)
        outputs = (64,)
    elif profile == "summarization_like":
        application_profile = "summarization_like"
        layout = "default"
        groups = 1
        stable = 256
        document = 0
        inputs = (4096,)
        outputs = (64,)
    else:
        application_profile = "agent_like"
        layout = "default"
        groups = 4
        stable = 4000
        document = 0
        inputs = tuple(
            value
            for value in (5504, 6016, 6528, 7040, 7552, 8064)
            for _ in range(groups)
        )
        outputs = tuple(
            value for value in (64, 96, 128, 160, 192, 256) for _ in range(groups)
        )

    reuse = PromptReuseConfig(
        mode="application",
        application_profile=application_profile,
        prompt_layout=layout,
        relationship_group_count=groups,
        stable_prefix_tokens=stable,
        document_tokens=document,
        cache_block_size_tokens=block_size,
    )
    workload = WorkloadConfig(
        name=f"e009_{profile}_cache_{cache}",
        input_tokens=TokenLengthDistribution(inputs, (1.0,) * len(inputs)),
        max_output_tokens=TokenLengthDistribution(outputs, (1.0,) * len(outputs)),
        prompt_reuse=reuse,
        sampling_mode="sequence",
    )
    return replace(base, workload=workload, num_requests=24)


def condition_name(
    config: BenchmarkConfig, profile: str, load_rate: float | None
) -> str:
    cache = "on" if config.enable_prefix_caching else "off"
    base = f"{profile}_cache_{cache}"
    return base if load_rate is None else f"{base}_rate_{load_rate:g}"


async def execute_e009_condition(
    config: BenchmarkConfig,
    *,
    profile: str,
    load_rate: float | None = None,
    tokenizer: Tokenizer | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Execute one isolated controlled or open-loop E009 condition."""
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

    async def measured_start() -> None:
        before_box.append(await fetch_vllm_metrics(config.base_url, client=client))

    slo_summary = None
    load_summary = None
    if load_rate is None:
        run_dir, result, gpu = await execute_benchmark(
            config,
            tokenizer=tokenizer,
            client=client,
            after_warmup_hook=measured_start,
        )
    else:
        if profile not in LOAD_PROFILES:
            raise ValueError("open-loop E009 is limited to RAG friendly/hostile")
        duration = config.num_requests / load_rate
        run_dir, result, slo_summary, gpu = await execute_open_loop_benchmark(
            config,
            request_rate=load_rate,
            duration_seconds=duration,
            max_in_flight=64,
            max_drain_seconds=300.0,
            slo=DEFAULT_SLO,
            tokenizer=tokenizer,
            client=client,
            after_warmup_hook=measured_start,
        )
        load_summary = {
            "offered_rate": load_rate,
            "offered_requests": len(result.measured_requests),
            "measured_duration_seconds": result.measured_duration_seconds,
            "dispatched_requests": result.dispatched_requests,
            "dropped_requests": result.dropped_requests,
            "timed_out_requests": result.timed_out_requests,
            "max_observed_in_flight": result.max_observed_in_flight,
        }
    if len(before_box) != 1:
        raise RuntimeError("E009 did not capture measured-window start")
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
    reuse_plan = _summarize_reuse_plan(
        loaded["workload"], block_size=reuse.cache_block_size_tokens
    )
    phase_latency = _summarize_cache_phases(loaded["requests"], loaded["workload"])
    mechanism_checks = _mechanism_checks(
        config,
        delta.to_dict(),
        expected_hits=int(reuse_plan["expected_cache_hit_tokens"]),
        exact_hits=load_rate is None,
    )
    success = (
        len(result.measured_requests) == config.num_requests
        and result.summary.failed_requests == 0
    )
    condition = {
        "experiment_id": "E009",
        "stage": "controlled" if load_rate is None else "open_loop",
        "condition": condition_name(config, profile, load_rate),
        "profile": profile,
        "success": success,
        "mechanism_valid": all(mechanism_checks.values()),
        "mechanism_checks": mechanism_checks,
        "run_dir": str(run_dir),
        "server_cache_config": asdict(live_config),
        "benchmark": asdict(result.summary),
        "gpu": asdict(gpu) if gpu is not None else None,
        "cache": delta.to_dict(),
        "reuse_plan": reuse_plan,
        "cache_phase_latency": phase_latency,
        "workload": loaded["summary"]["workload"],
        "load": load_summary,
        "slo": asdict(slo_summary) if slo_summary is not None else None,
    }
    (run_dir / "e009_condition.json").write_text(
        json.dumps(condition, indent=2), encoding="utf-8"
    )
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["e009"] = {
        key: condition[key]
        for key in (
            "stage",
            "condition",
            "profile",
            "mechanism_valid",
            "mechanism_checks",
            "reuse_plan",
            "cache_phase_latency",
        )
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir, condition


def summarize_e009_runs(
    run_dirs: list[Path | str], *, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    conditions = _load_conditions(run_dirs, stage="controlled")
    expected = {
        f"{profile}_cache_{cache}" for profile in PROFILES for cache in ("off", "on")
    }
    _validate_condition_set(conditions, expected, "ten controlled")
    by_name = {item["condition"]: item for item in conditions}
    profiles = []
    for profile in PROFILES:
        off = by_name[f"{profile}_cache_off"]
        on = by_name[f"{profile}_cache_on"]
        profiles.append(_paired_profile_summary(profile, off, on))
    valid = all(item["success"] and item["mechanism_valid"] for item in conditions)
    summary = {
        "experiment_id": "E009",
        "stage": "controlled",
        "status": "SUCCESS" if valid else "INVALID",
        "profiles": profiles,
    }
    out = _comparison_dir(output_root, "E009_controlled")
    (out / "e009_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    generate_e009_plots(profiles, out / "plots")
    return out, summary


def summarize_e009_load_runs(
    run_dirs: list[Path | str], *, output_root: Path
) -> tuple[Path, dict[str, Any]]:
    conditions = _load_conditions(run_dirs, stage="open_loop")
    expected = {
        f"{profile}_cache_{cache}_rate_{DEFAULT_LOAD_RATE:g}"
        for profile in LOAD_PROFILES
        for cache in ("off", "on")
    }
    _validate_condition_set(conditions, expected, "four open-loop")
    by_name = {item["condition"]: item for item in conditions}
    profiles = []
    for profile in LOAD_PROFILES:
        off = by_name[f"{profile}_cache_off_rate_{DEFAULT_LOAD_RATE:g}"]
        on = by_name[f"{profile}_cache_on_rate_{DEFAULT_LOAD_RATE:g}"]
        profiles.append(
            {
                "profile": profile,
                "offered_rate": DEFAULT_LOAD_RATE,
                "cache_off": off,
                "cache_on": on,
                "throughput_change_percent": _relative_change(
                    off["benchmark"]["request_throughput"],
                    on["benchmark"]["request_throughput"],
                ),
                "goodput_change_percent": _relative_change(
                    off["slo"]["goodput"], on["slo"]["goodput"]
                ),
            }
        )
    valid = all(item["success"] and item["mechanism_valid"] for item in conditions)
    summary = {
        "experiment_id": "E009",
        "stage": "open_loop",
        "status": "SUCCESS" if valid else "INVALID",
        "slo_definition": asdict(DEFAULT_SLO),
        "profiles": profiles,
    }
    out = _comparison_dir(output_root, "E009_load")
    (out / "e009_load_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    generate_e009_load_plots(profiles, out / "plots")
    return out, summary


def _paired_profile_summary(
    profile: str, off: dict[str, Any], on: dict[str, Any]
) -> dict[str, Any]:
    off_ttft = float(off["benchmark"]["ttft_stats"]["p50"])
    on_ttft = float(on["benchmark"]["ttft_stats"]["p50"])
    return {
        "profile": profile,
        "cache_off": off,
        "cache_on": on,
        "ttft_p50_change_percent": _relative_change(off_ttft, on_ttft),
        "e2e_p50_change_percent": _relative_change(
            off["benchmark"]["e2e_latency_stats"]["p50"],
            on["benchmark"]["e2e_latency_stats"]["p50"],
        ),
        "throughput_change_percent": _relative_change(
            off["benchmark"]["request_throughput"],
            on["benchmark"]["request_throughput"],
        ),
        "observed_cache_hit_fraction": on["cache"].get(
            "observed_prefix_cache_hit_fraction"
        ),
    }


def _load_conditions(run_dirs: list[Path | str], *, stage: str) -> list[dict[str, Any]]:
    result = []
    for raw in run_dirs:
        path = Path(raw) / "e009_condition.json"
        if not path.is_file():
            raise ValueError(f"missing E009 condition artifact: {path}")
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("experiment_id") != "E009" or item.get("stage") != stage:
            raise ValueError(f"unexpected E009 condition stage: {path}")
        result.append(item)
    return result


def _validate_condition_set(
    conditions: list[dict[str, Any]], expected: set[str], label: str
) -> None:
    observed = {str(item["condition"]) for item in conditions}
    if observed != expected or len(conditions) != len(expected):
        raise ValueError(
            f"E009 requires exactly {label} conditions; "
            f"missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _summarize_reuse_plan(
    workload: list[dict[str, Any]], *, block_size: int
) -> dict[str, Any]:
    measured = [item for item in workload if not item.get("is_warmup")]
    reusable = [int(item.get("actual_reusable_prefix_tokens", 0)) for item in measured]
    cacheable = [(value // block_size) * block_size for value in reusable]
    by_sequence: dict[str, list[int]] = {}
    for item, value in zip(measured, cacheable):
        key = str(item.get("sequence_index"))
        by_sequence.setdefault(key, []).append(value)
    return {
        "measured_requests": len(measured),
        "expected_cache_hit_tokens": sum(cacheable),
        "actual_reusable_prefix_tokens_min": min(reusable, default=0),
        "actual_reusable_prefix_tokens_max": max(reusable, default=0),
        "cacheable_tokens_by_sequence_index": {
            key: sum(values) / len(values) for key, values in by_sequence.items()
        },
        "relationship_ids": sorted(
            {str(item.get("relationship_id")) for item in measured}
        ),
    }


def _mechanism_checks(
    config: BenchmarkConfig,
    cache: dict[str, Any],
    *,
    expected_hits: int,
    exact_hits: bool,
) -> dict[str, bool]:
    hits = cache["prefix_cache_hit_tokens"]
    queries = cache["prefix_cache_query_tokens"]
    checks = {"request_plan_valid": True, "server_config_verified": True}
    if config.enable_prefix_caching:
        checks["cache_queries_observed"] = queries is not None and queries > 0
        checks["cache_hits_observed"] = hits is not None and hits > 0
        if exact_hits:
            checks["cache_hits_match_plan"] = hits == expected_hits
    else:
        checks["no_cache_hits_when_disabled"] = hits in {None, 0}
    return checks


def _summarize_cache_phases(
    requests: list[dict[str, Any]], workload: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {
        str(item["request_id"]): item for item in requests if not item.get("is_warmup")
    }
    values: dict[str, list[float]] = {}
    for item in workload:
        request = by_id.get(str(item["request_id"]))
        if (
            item.get("is_warmup")
            or request is None
            or request.get("ttft_seconds") is None
        ):
            continue
        state = str(item.get("expected_cache_state", "unknown"))
        values.setdefault(state, []).append(float(request["ttft_seconds"]))
    return {
        state: asdict(stats)
        for state, samples in values.items()
        if (stats := calculate_metric_stats(samples)) is not None
    }


def _relative_change(baseline: float, candidate: float) -> float | None:
    return (
        ((float(candidate) - float(baseline)) / float(baseline)) * 100.0
        if baseline
        else None
    )


def _comparison_dir(root: Path, prefix: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / f"{prefix}_{timestamp}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def print_condition(condition: dict[str, Any]) -> None:
    cache = condition["cache"]
    ttft = condition["benchmark"].get("ttft_stats") or {}
    print("=" * 78)
    print(f" E009 CONDITION: {condition['condition']}")
    print("=" * 78)
    print(f" Requests successful: {condition['benchmark']['successful_requests']}")
    print(f" TTFT P50:           {ttft.get('p50', 0.0) * 1000:.2f} ms")
    print(f" Cache query tokens: {cache['prefix_cache_query_tokens']}")
    print(f" Cache hit tokens:   {cache['prefix_cache_hit_tokens']}")
    print(f" Hit fraction:       {cache['observed_prefix_cache_hit_fraction']}")
    print(f" Mechanism valid:    {condition['mechanism_valid']}")
    print("=" * 78)


async def main_async(args: argparse.Namespace) -> int:
    if args.summarize or args.summarize_load:
        try:
            function = (
                summarize_e009_runs if args.summarize else summarize_e009_load_runs
            )
            out, summary = function(
                args.summarize or args.summarize_load,
                output_root=Path(args.output_dir or "runs"),
            )
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"E009 summary status: {summary['status']}")
        print(f"Artifacts: {out}")
        return 0 if summary["status"] == "SUCCESS" else 2

    if args.profile is None:
        print("Error: --profile is required with --config", file=sys.stderr)
        return 1
    try:
        base = load_e009_config(args.config)
        config = build_profile_config(base, args.profile)
        if args.base_url:
            config = replace(config, base_url=args.base_url)
        if args.output_dir:
            config = replace(config, output_dir=args.output_dir)
        if args.pilot:
            config = replace(config, num_requests=8, warmup_requests=2)
        run_dir, condition = await execute_e009_condition(
            config,
            profile=args.profile,
            load_rate=args.load_rate,
        )
    except (OSError, ValueError, httpx.HTTPError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_condition(condition)
    print(f"Artifacts: {run_dir}")
    return 0 if condition["success"] and condition["mechanism_valid"] else 2


def main() -> None:
    sys.exit(asyncio.run(main_async(create_parser().parse_args())))


if __name__ == "__main__":
    main()
