"""E005 — open-loop load, saturation, and SLO goodput."""

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

from inference_os.config import OpenLoopSweepConfig, load_config
from inference_os.reports.plots import generate_e005_plots
from inference_os.runner.load import constant_rate_arrival_offsets
from inference_os.runner.load_engine import execute_open_loop_benchmark

DEFAULT_CONFIG = "configs/e005_open_loop.yaml"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E005 — Open-Loop Load, Saturation, and SLO Goodput",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the first two rates for 10 seconds with two warm-ups",
    )
    parser.add_argument("--base-url", default=None, help="Override vLLM base URL")
    parser.add_argument("--output-dir", default=None, help="Override output root")
    return parser


async def execute_e005(
    sweep: OpenLoopSweepConfig,
    *,
    output_root: Path,
) -> tuple[Path, list[dict[str, Any]]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_root / f"E005_{timestamp}_{uuid.uuid4().hex[:8]}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    from inference_os.workloads.hf_tokenizer import HFTokenizer

    tokenizer = HFTokenizer.from_pretrained(sweep.base_config.model)
    points: list[dict[str, Any]] = []
    for rate in sweep.request_rates:
        duration_seconds = sweep.duration_for_rate(rate)
        request_count = len(constant_rate_arrival_offsets(rate, duration_seconds))
        point_config = replace(
            sweep.base_config,
            experiment_id=sweep.experiment_id,
            num_requests=request_count,
            output_dir=str(experiment_dir / f"rate_{rate:g}"),
        )
        run_dir, result, slo_summary, gpu_summary = await execute_open_loop_benchmark(
            point_config,
            request_rate=rate,
            duration_seconds=duration_seconds,
            max_in_flight=sweep.max_in_flight,
            max_drain_seconds=sweep.max_drain_seconds,
            slo=sweep.slo,
            tokenizer=tokenizer,
        )
        load = {
            "offered_rate": rate,
            "realized_offered_rate": len(result.measured_requests) / duration_seconds,
            "offered_duration_seconds": duration_seconds,
            "measured_duration_seconds": result.measured_duration_seconds,
            "offered_requests": len(result.measured_requests),
            "dispatched_requests": result.dispatched_requests,
            "dropped_requests": result.dropped_requests,
            "timed_out_requests": result.timed_out_requests,
            "max_observed_in_flight": result.max_observed_in_flight,
        }
        points.append(
            {
                "offered_rate": rate,
                "run_dir": str(run_dir),
                "accounting_complete": (
                    len(result.measured_requests)
                    == result.dispatched_requests + result.dropped_requests
                ),
                "benchmark": asdict(result.summary),
                "load": load,
                "slo": asdict(slo_summary),
                "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
            }
        )

    passing_rates = [
        point["offered_rate"] for point in points if point["slo"]["all_objectives_met"]
    ]
    summary = {
        "experiment_id": "E005",
        "status": (
            "SUCCESS"
            if all(point["accounting_complete"] for point in points)
            else "FAILED"
        ),
        "arrival_process": "deterministic_constant_rate",
        "rate_point_count": len(points),
        "slo_definition": asdict(sweep.slo),
        "highest_observed_slo_passing_rate": max(passing_rates, default=None),
        "interpretation_note": (
            "The highest passing rate is bounded by the configured sweep and is not "
            "an automatic capacity recommendation."
        ),
        "points": points,
    }
    (experiment_dir / "e005_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    generate_e005_plots(points, experiment_dir / "plots")
    return experiment_dir, points


def print_results(points: list[dict[str, Any]]) -> None:
    print("=" * 125)
    print(" E005: OPEN-LOOP LOAD, SATURATION, AND SLO GOODPUT")
    print("=" * 125)
    print(
        f"{'Offered':>8} | {'Achieved':>8} | {'Goodput':>8} | {'TTFT P95':>10} | "
        f"{'E2E P95':>10} | {'In-flight':>9} | {'Drops':>7} | {'SLO':>5}"
    )
    print("-" * 125)
    for point in points:
        benchmark = point["benchmark"]
        ttft = benchmark.get("ttft_stats") or {}
        e2e = benchmark.get("e2e_latency_stats") or {}
        print(
            f"{point['offered_rate']:>8.2f} | "
            f"{benchmark['request_throughput']:>8.2f} | "
            f"{point['slo']['goodput']:>8.2f} | "
            f"{ttft.get('p95', 0.0) * 1000:>8.2f} ms | "
            f"{e2e.get('p95', 0.0):>8.2f} s | "
            f"{point['load']['max_observed_in_flight']:>9} | "
            f"{point['load']['dropped_requests']:>7} | "
            f"{('PASS' if point['slo']['all_objectives_met'] else 'FAIL'):>5}"
        )
    print("=" * 125)


async def main_async(args: argparse.Namespace) -> int:
    try:
        loaded = load_config(Path(args.config))
        if not isinstance(loaded, OpenLoopSweepConfig):
            raise ValueError("E005 requires an open-loop config with request_rates")
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    sweep = loaded
    base = sweep.base_config
    if args.base_url:
        base = replace(base, base_url=args.base_url)
    if args.pilot:
        sweep = replace(
            sweep,
            request_rates=sweep.request_rates[:2],
            duration_seconds=10.0,
            requests_per_rate=None,
            base_config=replace(base, warmup_requests=2),
        )
    else:
        sweep = replace(sweep, base_config=base)
    output_root = Path(args.output_dir or sweep.base_config.output_dir)
    experiment_dir, points = await execute_e005(sweep, output_root=output_root)
    print_results(points)
    print(f"Artifacts: {experiment_dir}")
    return 0 if all(point["accounting_complete"] for point in points) else 1


def main() -> None:
    args = create_parser().parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
