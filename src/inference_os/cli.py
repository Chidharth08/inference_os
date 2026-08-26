"""CLI entry point for inference_os."""

import argparse
import asyncio
import sys
from typing import Sequence

from inference_os.config import BenchmarkConfig
from inference_os.runner.engine import execute_benchmark


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="inference-os",
        description="A reproducible LLM inference experimentation framework.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    run_parser = subparsers.add_parser("run", help="Run benchmark experiment")
    run_parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Serving model identifier on vLLM server",
    )
    run_parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the vLLM OpenAI-compatible server",
    )
    run_parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Target number of prompt tokens",
    )
    run_parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=64,
        help="Maximum output tokens to generate",
    )
    run_parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of measured benchmark requests",
    )
    run_parser.add_argument(
        "--warmup-requests",
        type=int,
        default=2,
        help="Number of warm-up requests",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt generation",
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    run_parser.add_argument(
        "--output-dir",
        type=str,
        default="runs",
        help="Output directory for results",
    )
    run_parser.add_argument(
        "--experiment-id",
        type=str,
        default="E000",
        help="Experiment ID tag",
    )

    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "run":
        config = BenchmarkConfig(
            model=parsed_args.model,
            base_url=parsed_args.base_url,
            prompt_tokens=parsed_args.prompt_tokens,
            max_output_tokens=parsed_args.max_output_tokens,
            num_requests=parsed_args.num_requests,
            warmup_requests=parsed_args.warmup_requests,
            seed=parsed_args.seed,
            temperature=parsed_args.temperature,
            experiment_id=parsed_args.experiment_id,
            output_dir=parsed_args.output_dir,
        )
        run_dir, result, _ = asyncio.run(execute_benchmark(config))
        print(f"Benchmark completed successfully! Run saved to: {run_dir}")
        print(f"Total Measured Requests: {result.summary.total_requests}")
        print(f"Request Throughput:      {result.summary.request_throughput:.2f} req/s")
        print(
            f"Token Throughput:        "
            f"{result.summary.output_token_throughput:.2f} tok/s"
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
