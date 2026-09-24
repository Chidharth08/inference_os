"""Benchmark run result persistence and disk serialization."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from inference_os.config import BenchmarkConfig
from inference_os.metrics.summary import calculate_metric_stats
from inference_os.telemetry.environment import EnvironmentMetadata
from inference_os.telemetry.gpu import GPUSample, GPUTelemetrySummary
from inference_os.workloads.spec import RequestSpec, TokenLengthDistribution

if TYPE_CHECKING:
    from inference_os.runner.benchmark import BenchmarkResult


def save_benchmark_run(
    config: BenchmarkConfig,
    environment: EnvironmentMetadata,
    result: BenchmarkResult,
    gpu_summary: Optional[GPUTelemetrySummary] = None,
    gpu_samples: Optional[Sequence[GPUSample]] = None,
    output_dir: Optional[Path | str] = None,
    run_id: Optional[str] = None,
    warmup_workload_specs: Optional[Sequence[RequestSpec]] = None,
    workload_specs: Optional[Sequence[RequestSpec]] = None,
) -> Path:
    """Save complete benchmark run artifacts to disk.

    Creates a structured run directory containing:
    - config.json: Input configuration
    - environment.json: Hardware and software environment snapshot
    - summary.json: Aggregate benchmark, warmup, and GPU summaries
    - requests.jsonl: Line-by-line raw RequestMeasurement records
    - workload.jsonl: Exact requested token shape for each planned request
    - telemetry.jsonl: Line-by-line raw GPUSample records (if present)

    Returns:
        The Path to the created run directory.
    """
    base_dir = Path(output_dir if output_dir is not None else config.output_dir)

    if run_id is None:
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rand_id = uuid.uuid4().hex[:8]
        run_id = f"{config.experiment_id}_{ts_str}_{rand_id}"

    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    warmup_specs = (
        list(warmup_workload_specs)
        if warmup_workload_specs is not None
        else [
            RequestSpec(config.prompt_tokens, config.max_output_tokens)
            for _ in result.warmup_measurements
        ]
    )
    measured_specs = (
        list(workload_specs)
        if workload_specs is not None
        else [
            RequestSpec(config.prompt_tokens, config.max_output_tokens)
            for _ in result.measured_requests
        ]
    )
    if len(warmup_specs) != len(result.warmup_measurements):
        raise ValueError("warmup workload plan does not match warmup measurements")
    if len(measured_specs) != len(result.measured_requests):
        raise ValueError("workload plan does not match measured requests")

    # 1. Write config.json
    config_path = run_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    # 2. Write environment.json
    env_path = run_dir / "environment.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(environment.to_dict(), f, indent=2)

    # 3. Write summary.json
    summary_data: dict[str, Any] = {
        "run_id": run_id,
        "benchmark": asdict(result.summary),
        "warmup": (
            asdict(result.warmup_summary) if result.warmup_summary is not None else None
        ),
        "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
        "workload": _summarize_workload(config, measured_specs),
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 4. Write the exact realized request plan.
    workload_path = run_dir / "workload.jsonl"
    with open(workload_path, "w", encoding="utf-8") as f:
        for measurement, spec in zip(result.warmup_measurements, warmup_specs):
            f.write(
                json.dumps(_workload_record(measurement.request_id, True, spec)) + "\n"
            )
        for measurement, spec in zip(result.measured_requests, measured_specs):
            f.write(
                json.dumps(_workload_record(measurement.request_id, False, spec)) + "\n"
            )

    # 5. Write requests.jsonl
    requests_path = run_dir / "requests.jsonl"
    with open(requests_path, "w", encoding="utf-8") as f:
        for req in result.warmup_measurements:
            req_dict = asdict(req)
            req_dict["is_warmup"] = True
            req_dict["ttft_seconds"] = req.ttft_seconds
            req_dict["e2e_latency_seconds"] = req.e2e_latency_seconds
            f.write(json.dumps(req_dict) + "\n")

        for req in result.measured_requests:
            req_dict = asdict(req)
            req_dict["is_warmup"] = False
            req_dict["ttft_seconds"] = req.ttft_seconds
            req_dict["e2e_latency_seconds"] = req.e2e_latency_seconds
            f.write(json.dumps(req_dict) + "\n")

    # 6. Write telemetry.jsonl
    if gpu_samples:
        telemetry_path = run_dir / "telemetry.jsonl"
        with open(telemetry_path, "w", encoding="utf-8") as f:
            for s in gpu_samples:
                f.write(json.dumps(asdict(s)) + "\n")

    return run_dir


def load_benchmark_run(run_dir: Path | str) -> dict[str, Any]:
    """Load all saved artifacts from a benchmark run directory."""
    path = Path(run_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {path}")

    with open(path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(path / "environment.json", "r", encoding="utf-8") as f:
        environment = json.load(f)

    with open(path / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    requests: list[dict[str, Any]] = []
    requests_file = path / "requests.jsonl"
    if requests_file.exists():
        with open(requests_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    requests.append(json.loads(line))

    telemetry: list[dict[str, Any]] = []
    telemetry_file = path / "telemetry.jsonl"
    if telemetry_file.exists():
        with open(telemetry_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    telemetry.append(json.loads(line))

    workload: list[dict[str, Any]] = []
    workload_file = path / "workload.jsonl"
    if workload_file.exists():
        with open(workload_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    workload.append(json.loads(line))

    return {
        "run_dir": str(path),
        "config": config,
        "environment": environment,
        "summary": summary,
        "requests": requests,
        "workload": workload,
        "telemetry": telemetry,
    }


def _workload_record(
    request_id: str,
    is_warmup: bool,
    spec: RequestSpec,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "is_warmup": is_warmup,
        "target_input_tokens": spec.target_input_tokens,
        "max_output_tokens": spec.max_output_tokens,
    }


def _summarize_workload(
    config: BenchmarkConfig,
    specs: Sequence[RequestSpec],
) -> dict[str, Any]:
    input_stats = calculate_metric_stats(
        [float(spec.target_input_tokens) for spec in specs]
    )
    output_stats = calculate_metric_stats(
        [float(spec.max_output_tokens) for spec in specs]
    )
    if config.workload is None:
        input_distribution = None
        output_distribution = None
        sampling_mode = "fixed"
    else:
        input_distribution = config.workload.input_tokens
        output_distribution = config.workload.max_output_tokens
        sampling_mode = config.workload.sampling_mode

    summary = {
        "profile_name": config.workload.name if config.workload is not None else None,
        "seed": config.seed,
        "sampling_mode": sampling_mode,
        "request_count": len(specs),
        "target_input_tokens": asdict(input_stats) if input_stats else None,
        "max_output_tokens": asdict(output_stats) if output_stats else None,
    }
    if input_distribution is not None and output_distribution is not None:
        summary["configured_distributions"] = {
            "target_input_tokens": _distribution_summary(input_distribution),
            "max_output_tokens": _distribution_summary(output_distribution),
        }
        summary["sampling_error"] = {
            "target_input_tokens": _mean_sampling_error(
                input_stats.mean if input_stats else 0.0,
                input_distribution.mean,
            ),
            "max_output_tokens": _mean_sampling_error(
                output_stats.mean if output_stats else 0.0,
                output_distribution.mean,
            ),
        }
    return summary


def _distribution_summary(
    distribution: TokenLengthDistribution,
) -> dict[str, Any]:
    mean = distribution.mean
    return {
        "values": list(distribution.values),
        "weights": list(distribution.normalized_weights),
        "mean": mean,
        "std_dev": distribution.std_dev,
        "coefficient_of_variation": distribution.std_dev / mean,
    }


def _mean_sampling_error(
    realized_mean: float, configured_mean: float
) -> dict[str, float]:
    delta = realized_mean - configured_mean
    return {
        "configured_mean": configured_mean,
        "realized_mean": realized_mean,
        "mean_delta_tokens": delta,
        "mean_delta_percent": (delta / configured_mean) * 100.0,
    }
