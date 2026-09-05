"""CLI entry point for inference_os."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from inference_os.config import BenchmarkConfig, SweepConfig, load_config
from inference_os.runner.engine import execute_benchmark
from inference_os.runner.sweep import execute_sweep


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="inference-os",
        description="A reproducible LLM inference experimentation framework.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Single run command
    run_parser = subparsers.add_parser("run", help="Run single benchmark experiment")
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON benchmark configuration file",
    )
    run_parser.add_argument(
        "--model",
        type=str,
        default=None,
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

    # Sweep command
    sweep_parser = subparsers.add_parser(
        "sweep", help="Run 1D parameter sweep experiment"
    )
    sweep_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML or JSON sweep configuration file",
    )

    return parser


def main(args: Sequence[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command == "run":
        if parsed_args.config:
            loaded = load_config(parsed_args.config)
            if not isinstance(loaded, BenchmarkConfig):
                raise ValueError(
                    f"Expected single-run BenchmarkConfig in {parsed_args.config}, "
                    "found SweepConfig"
                )
            config = loaded
        else:
            if not parsed_args.model:
                parser.error("Must provide either --config or --model for 'run'")
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
        if result.summary.successful_requests == 0:
            tot = result.summary.total_requests
            print(
                f"Benchmark failed: 0/{tot} requests succeeded. Run saved to: {run_dir}"
            )
            return 1

        print(f"Benchmark completed successfully! Run saved to: {run_dir}")
        print(f"Total Measured Requests: {result.summary.total_requests}")
        print(f"Request Throughput:      {result.summary.request_throughput:.2f} req/s")
        print(
            f"Token Throughput:        "
            f"{result.summary.output_token_throughput:.2f} tok/s"
        )
        return 0

    if parsed_args.command == "sweep":
        loaded = load_config(parsed_args.config)
        if not isinstance(loaded, SweepConfig):
            raise ValueError(
                f"Expected SweepConfig in {parsed_args.config}, "
                "found single-run BenchmarkConfig"
            )
        sweep_dir, point_results = asyncio.run(execute_sweep(loaded))
        successful_points = sum(
            1
            for p in point_results
            if p.get("benchmark", {}).get("successful_requests", 0) > 0
        )
        total_points = len(point_results)

        if successful_points == 0:
            print(
                f"Parameter sweep failed: all {total_points} sweep points failed "
                f"(0 successful requests). Saved to: {sweep_dir}"
            )
            return 1

        if successful_points < total_points:
            print(
                f"Parameter sweep partially completed "
                f"({successful_points}/{total_points} passed). Saved to: {sweep_dir}"
            )
            return 1

        print(f"Parameter sweep completed successfully! Saved to: {sweep_dir}")
        print(f"Sweep Points Executed: {len(point_results)}")
        plots_dir = Path(sweep_dir) / "plots"
        if plots_dir.exists():
            print(f"Plots saved to: {plots_dir}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
