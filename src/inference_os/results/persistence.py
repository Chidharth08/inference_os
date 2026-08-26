"""Benchmark run result persistence and disk serialization."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Sequence

from inference_os.config import BenchmarkConfig
from inference_os.telemetry.environment import EnvironmentMetadata
from inference_os.telemetry.gpu import GPUSample, GPUTelemetrySummary

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
) -> Path:
    """Save complete benchmark run artifacts to disk.

    Creates a structured run directory containing:
    - config.json: Input configuration
    - environment.json: Hardware and software environment snapshot
    - summary.json: Aggregate benchmark, warmup, and GPU summaries
    - requests.jsonl: Line-by-line raw RequestMeasurement records
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
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # 4. Write requests.jsonl
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

    # 5. Write telemetry.jsonl
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

    return {
        "run_dir": str(path),
        "config": config,
        "environment": environment,
        "summary": summary,
        "requests": requests,
        "telemetry": telemetry,
    }
