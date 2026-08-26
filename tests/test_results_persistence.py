"""Unit tests for benchmark result persistence and loading."""

from pathlib import Path

import pytest

from inference_os import BenchmarkConfig
from inference_os.metrics import (
    BenchmarkSummary,
    MetricStats,
    RequestMeasurement,
)
from inference_os.results import load_benchmark_run, save_benchmark_run
from inference_os.runner import BenchmarkResult
from inference_os.telemetry import (
    EnvironmentMetadata,
    GitMetadata,
    GPUMetadata,
    GPUSample,
    GPUTelemetrySummary,
)


def test_save_and_load_benchmark_run(tmp_path: Path) -> None:
    """Verify saving run artifacts to disk and loading them back accurately."""
    config = BenchmarkConfig(
        model="Qwen/Qwen2.5-7B-Instruct",
        prompt_tokens=128,
        max_output_tokens=64,
        num_requests=2,
        warmup_requests=1,
        experiment_id="E000",
    )

    env = EnvironmentMetadata(
        timestamp_utc="2026-08-25T12:00:00Z",
        hostname="test-host",
        os_name="Linux",
        os_release="6.8.0",
        python_version="3.11.3",
        git=GitMetadata(commit_hash="abc1234", branch="main", is_dirty=False),
        gpu=GPUMetadata(
            name="NVIDIA RTX 3090", driver_version="580", total_memory_mb=24576
        ),
        packages={"transformers": "5.15.1"},
    )

    m_warmup = RequestMeasurement(
        request_id="warmup-1",
        start_time_ns=1_000_000_000,
        first_token_time_ns=1_200_000_000,
        completion_time_ns=2_000_000_000,
        input_tokens=128,
        output_tokens=64,
        success=True,
    )
    m_req1 = RequestMeasurement(
        request_id="req-1",
        start_time_ns=2_100_000_000,
        first_token_time_ns=2_300_000_000,
        completion_time_ns=3_000_000_000,
        input_tokens=128,
        output_tokens=64,
        success=True,
    )
    m_req2 = RequestMeasurement(
        request_id="req-2",
        start_time_ns=3_100_000_000,
        first_token_time_ns=3_300_000_000,
        completion_time_ns=4_000_000_000,
        input_tokens=128,
        output_tokens=64,
        success=True,
    )

    stats = MetricStats(
        count=2,
        mean=0.2,
        std_dev=0.0,
        min=0.2,
        max=0.2,
        p50=0.2,
        p90=0.2,
        p95=0.2,
        p99=0.2,
    )
    summary = BenchmarkSummary(
        total_requests=2,
        successful_requests=2,
        failed_requests=0,
        total_input_tokens=256,
        total_output_tokens=128,
        total_duration_seconds=1.9,
        request_throughput=2 / 1.9,
        output_token_throughput=128 / 1.9,
        ttft_stats=stats,
        e2e_latency_stats=stats,
    )
    warmup_summary = BenchmarkSummary(
        total_requests=1,
        successful_requests=1,
        failed_requests=0,
        total_input_tokens=128,
        total_output_tokens=64,
        total_duration_seconds=1.0,
        request_throughput=1.0,
        output_token_throughput=64.0,
    )

    result = BenchmarkResult(
        warmup_measurements=[m_warmup],
        measured_requests=[m_req1, m_req2],
        summary=summary,
        warmup_summary=warmup_summary,
    )

    gpu_sample = GPUSample(
        timestamp_ns=2_500_000_000,
        memory_used_mb=14000,
        memory_total_mb=24576,
        utilization_gpu_pct=80,
    )
    gpu_summary = GPUTelemetrySummary(
        sample_count=1,
        peak_memory_used_mb=14000,
        avg_memory_used_mb=14000.0,
        peak_utilization_gpu_pct=80,
        avg_utilization_gpu_pct=80.0,
        total_memory_mb=24576,
    )

    run_dir = save_benchmark_run(
        config=config,
        environment=env,
        result=result,
        gpu_summary=gpu_summary,
        gpu_samples=[gpu_sample],
        output_dir=tmp_path,
        run_id="test_run_001",
    )

    # Verify directory and files
    assert run_dir.is_dir()
    assert (run_dir / "config.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "requests.jsonl").exists()
    assert (run_dir / "telemetry.jsonl").exists()

    # Load back
    loaded = load_benchmark_run(run_dir)
    assert loaded["config"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert loaded["environment"]["hostname"] == "test-host"
    assert loaded["summary"]["benchmark"]["total_requests"] == 2
    assert loaded["summary"]["warmup"]["total_requests"] == 1
    assert loaded["summary"]["gpu"]["peak_memory_used_mb"] == 14000

    # Verify requests lines
    assert len(loaded["requests"]) == 3  # 1 warmup + 2 measured
    assert loaded["requests"][0]["is_warmup"] is True
    assert loaded["requests"][0]["request_id"] == "warmup-1"
    assert loaded["requests"][1]["is_warmup"] is False
    assert loaded["requests"][1]["request_id"] == "req-1"

    # Verify telemetry lines
    assert len(loaded["telemetry"]) == 1
    assert loaded["telemetry"][0]["memory_used_mb"] == 14000


def test_load_non_existent_run(tmp_path: Path) -> None:
    """Verify FileNotFoundError when loading non-existent directory."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_run(tmp_path / "non_existent_dir_123")
