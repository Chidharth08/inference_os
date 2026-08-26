"""E000 — Measurement Validation Benchmark Execution Script."""

import argparse
import asyncio
import sys
from pathlib import Path

from inference_os.config import BenchmarkConfig
from inference_os.runner.engine import execute_benchmark


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser for E000 benchmark."""
    parser = argparse.ArgumentParser(
        description="E000 — LLM Inference Measurement Validation Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Serving model identifier on vLLM server",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM OpenAI-compatible server",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Target number of prompt tokens",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=64,
        help="Maximum number of output tokens to generate per request",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of measured benchmark requests to run",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
        help="Number of warm-up requests to run before measurement",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic prompt generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Directory to store benchmark run results",
    )
    return parser


def format_ms(val_seconds: float | None) -> str:
    """Format seconds value as milliseconds string."""
    if val_seconds is None:
        return "N/A"
    return f"{val_seconds * 1000.0:8.2f} ms ({val_seconds:6.4f} s)"


async def main_async(args: argparse.Namespace) -> int:
    """Async main entrypoint for E000 benchmark."""
    config = BenchmarkConfig(
        model=args.model,
        base_url=args.base_url,
        prompt_tokens=args.prompt_tokens,
        max_output_tokens=args.max_output_tokens,
        num_requests=args.num_requests,
        warmup_requests=args.warmup_requests,
        seed=args.seed,
        temperature=args.temperature,
        experiment_id="E000",
        output_dir=args.output_dir,
    )

    print("=" * 70)
    print(" inference_os — E000 Measurement Validation Benchmark")
    print("=" * 70)
    print(f" Model:               {config.model}")
    print(f" Endpoint:            {config.base_url}")
    print(f" Target Prompt Tokens:{config.prompt_tokens}")
    print(f" Max Output Tokens:   {config.max_output_tokens}")
    print(f" Warm-up Requests:    {config.warmup_requests}")
    print(f" Measured Requests:   {config.num_requests}")
    print(f" Seed:                {config.seed}")
    print("-" * 70)
    print("Executing benchmark pipeline...")

    run_dir, result, gpu_summary = await execute_benchmark(config)
    summary = result.summary

    print("-" * 70)
    print(" Benchmark Execution Completed Successfully!")
    print("=" * 70)
    print(" EXECUTION SUMMARY")
    print(f"   Total Measured:     {summary.total_requests}")
    print(f"   Successful:         {summary.successful_requests}")
    print(f"   Failed:             {summary.failed_requests}")
    print(f"   Total Input Tokens: {summary.total_input_tokens}")
    print(f"   Total Output Tokens:{summary.total_output_tokens}")
    print(f"   Benchmark Duration: {summary.total_duration_seconds:.4f} s")
    print()
    print(" THROUGHPUT")
    print(f"   Request Throughput: {summary.request_throughput:.2f} req/s")
    print(f"   Token Throughput:   {summary.output_token_throughput:.2f} tok/s")
    print()

    if summary.ttft_stats is not None:
        print(" TIME TO FIRST TOKEN (TTFT)")
        print(f"   Mean:   {format_ms(summary.ttft_stats.mean)}")
        print(f"   P50:    {format_ms(summary.ttft_stats.p50)}")
        print(f"   P90:    {format_ms(summary.ttft_stats.p90)}")
        print(f"   P95:    {format_ms(summary.ttft_stats.p95)}")
        print(f"   Min:    {format_ms(summary.ttft_stats.min)}")
        print(f"   Max:    {format_ms(summary.ttft_stats.max)}")
        print(f"   StdDev: {format_ms(summary.ttft_stats.std_dev)}")
        print()

    if summary.e2e_latency_stats is not None:
        print(" END-TO-END LATENCY (E2E)")
        print(f"   Mean:   {format_ms(summary.e2e_latency_stats.mean)}")
        print(f"   P50:    {format_ms(summary.e2e_latency_stats.p50)}")
        print(f"   P90:    {format_ms(summary.e2e_latency_stats.p90)}")
        print(f"   P95:    {format_ms(summary.e2e_latency_stats.p95)}")
        print(f"   Min:    {format_ms(summary.e2e_latency_stats.min)}")
        print(f"   Max:    {format_ms(summary.e2e_latency_stats.max)}")
        print(f"   StdDev: {format_ms(summary.e2e_latency_stats.std_dev)}")
        print()

    if gpu_summary is not None:
        print(" GPU TELEMETRY")
        print(
            f"   VRAM Usage:         Peak: {gpu_summary.peak_memory_used_mb} MiB "
            f"/ Avg: {gpu_summary.avg_memory_used_mb:.1f} MiB "
            f"(Total: {gpu_summary.total_memory_mb} MiB)"
        )
        print(
            f"   GPU Compute:        Peak: {gpu_summary.peak_utilization_gpu_pct}% "
            f"/ Avg: {gpu_summary.avg_utilization_gpu_pct:.1f}%"
        )
        print()

    print(f" Results Saved To: {Path(run_dir).resolve()}")
    print("=" * 70)
    return 0


def main() -> int:
    """CLI entry point for running E000 benchmark."""
    parser = create_parser()
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
