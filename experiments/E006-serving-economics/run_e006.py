"""E006 — offline serving-economics analysis of canonical E005 results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml

from inference_os.metrics.economics import (
    EconomicsConfig,
    compute_cost_metrics,
)
from inference_os.reports.plots import generate_e006_plots

DEFAULT_CONFIG = "configs/e006_economics.yaml"


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="E006 — Offline Serving Economics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-summary", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu-price-per-hour", type=float, default=None)
    return parser


def load_economics_config(path: Path | str) -> EconomicsConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("E006 configuration must be a mapping")
    return EconomicsConfig.from_dict(data)


def analyze_e005_summary(
    source: dict[str, Any], config: EconomicsConfig
) -> dict[str, Any]:
    """Convert one E005 sweep summary into an auditable economics summary."""
    if source.get("experiment_id") != "E005":
        raise ValueError("E006 source_summary must contain an E005 experiment")
    raw_points = source.get("points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("E005 source summary has no load points")

    points: list[dict[str, Any]] = []
    for point in raw_points:
        benchmark = point.get("benchmark") or {}
        load = point.get("load") or {}
        slo = point.get("slo") or {}
        metrics = compute_cost_metrics(
            measured_duration_seconds=float(
                load.get(
                    "measured_duration_seconds",
                    benchmark.get("total_duration_seconds", 0.0),
                )
            ),
            completed_requests=int(benchmark.get("successful_requests", 0)),
            slo_compliant_requests=int(slo.get("compliant_requests", 0)),
            input_tokens=int(benchmark.get("total_input_tokens", 0)),
            output_tokens=int(benchmark.get("total_output_tokens", 0)),
            assumptions=config.assumptions,
        )
        points.append(
            {
                "offered_rate": float(point["offered_rate"]),
                "achieved_request_throughput": float(
                    benchmark.get("request_throughput", 0.0)
                ),
                "goodput": float(slo.get("goodput", 0.0)),
                "slo_compliance_rate": float(slo.get("compliance_rate", 0.0)),
                "aggregate_slo_met": bool(slo.get("all_objectives_met", False)),
                "cost_metrics": asdict(metrics),
            }
        )

    aggregate_passing = [
        point["offered_rate"] for point in points if point["aggregate_slo_met"]
    ]
    normalized_cost = sum(point["cost_metrics"]["estimated_cost"] for point in points)
    measured_seconds = sum(
        point["cost_metrics"]["measured_duration_seconds"] for point in points
    )
    billed_seconds = sum(
        point["cost_metrics"]["billable_duration_seconds"] for point in points
    )
    actual = config.assumptions.actual_session_cost
    overhead = actual - normalized_cost if actual is not None else None
    ratio = (
        actual / normalized_cost if actual is not None and normalized_cost > 0 else None
    )
    return {
        "experiment_id": config.experiment_id,
        "status": "SUCCESS",
        "analysis_type": "offline_benchmark_normalized_cost_estimate",
        "source_experiment_id": source["experiment_id"],
        "source_slo_definition": source.get("slo_definition"),
        "assumptions": config.assumptions.to_dict(),
        "measurement_scope": config.measurement_scope,
        "aggregate_slo_passing_rates": aggregate_passing,
        "highest_observed_aggregate_slo_passing_rate": (
            max(aggregate_passing) if aggregate_passing else None
        ),
        "sustainable_cost_available": bool(aggregate_passing),
        "sustainable_cost_note": (
            "No E005 point passed the complete aggregate SLO; point costs are "
            "reported without labeling any point sustainable."
            if not aggregate_passing
            else "Passing rates are empirical and bounded by the source sweep."
        ),
        "benchmark_normalized_totals": {
            "measured_duration_seconds": measured_seconds,
            "billable_duration_seconds_sum_of_points": billed_seconds,
            "estimated_cost": normalized_cost,
        },
        "session_reconciliation": {
            "actual_session_cost": actual,
            "actual_session_cost_is_approximate": (
                config.assumptions.actual_session_cost_is_approximate
            ),
            "unattributed_session_cost": overhead,
            "actual_to_benchmark_normalized_ratio": ratio,
            "note": (
                "Unattributed session cost includes setup, model transfer/loading, "
                "warm-up, inter-point time, packaging, storage, bandwidth, and idle "
                "time; the available balance change cannot separate these components."
            ),
        },
        "points": sorted(points, key=lambda point: point["offered_rate"]),
    }


def _money(value: float | None, digits: int = 6) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def render_report(summary: dict[str, Any], source_path: Path) -> str:
    currency = summary["assumptions"]["currency"]
    price = summary["assumptions"]["gpu_price_per_hour"]
    total = summary["benchmark_normalized_totals"]
    session = summary["session_reconciliation"]
    if session["actual_session_cost"] is None:
        session_lines = [
            "No actual session cost was configured, so this report does not compare",
            "the measured-window estimate with total rental spend.",
        ]
    else:
        session_lines = [
            f"The observed Vast.ai credit decrease was approximately {currency} "
            f"{session['actual_session_cost']:.2f}. The difference of {currency} "
            f"{session['unattributed_session_cost']:.6f} is not assigned to serving "
            "requests because the available account observation does not separate "
            "model download, container transfer, storage, bandwidth, setup, "
            "warm-up, idle time, and other session activity.",
        ]
    lines = [
        "# E006 Serving Economics — Report",
        "",
        "## Scope and conclusion",
        "",
        "E006 is an offline cost analysis of the canonical E005 open-loop run. It",
        "does not execute inference or rent another GPU. All estimates use the",
        f"explicit price assumption of **{currency} {price:.3f} per GPU-hour**.",
        "",
        "No E005 point passed the complete aggregate SLO because E2E P95 exceeded",
        "five seconds at every offered rate. Therefore this report does not present",
        "a cost at sustainable aggregate-SLO capacity. It reports measured point",
        "costs and per-request SLO-compliant cost without changing the original SLO.",
        "",
        "## Assumptions",
        "",
        f"- Source: `{source_path.as_posix()}`",
        f"- Currency: {currency}",
        f"- GPU price: {currency} {price:.3f}/hour",
        f"- GPU count: {summary['assumptions']['gpu_count']}",
        "- Billing granularity: "
        f"{summary['assumptions']['billing_increment_seconds']:.3f} second(s)",
        "- Idle time inside each measured point: "
        f"{'included' if summary['assumptions']['idle_time_included'] else 'excluded'}",
        f"- Measurement scope: {summary['measurement_scope']}",
        "",
        "## Point estimates",
        "",
        "| Offered req/s | Completed | Individually compliant | Measured s | "
        f"Estimated {currency} | {currency}/1k completed | "
        f"{currency}/1k compliant | {currency}/1M output tokens | Aggregate SLO |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for point in summary["points"]:
        item = point["cost_metrics"]
        lines.append(
            f"| {point['offered_rate']:.1f} | {item['completed_requests']} | "
            f"{item['slo_compliant_requests']} | "
            f"{item['measured_duration_seconds']:.3f} | "
            f"{_money(item['estimated_cost'])} | "
            f"{_money(item['cost_per_1000_completed_requests'], 4)} | "
            f"{_money(item['cost_per_1000_slo_compliant_requests'], 4)} | "
            f"{_money(item['cost_per_million_output_tokens'], 4)} | "
            f"{'Pass' if point['aggregate_slo_met'] else 'Fail'} |"
        )
    lines.extend(
        [
            "",
            "Higher offered load reduced normalized cost per completion through",
            "batching and higher utilization, but individual SLO compliance fell.",
            "A cheaper completion is not automatically a better service outcome.",
            "",
            "## Benchmark cost versus session spend",
            "",
            f"The five measured windows total "
            f"{total['measured_duration_seconds']:.3f} seconds. Applying per-point "
            f"billing granularity produces "
            f"{total['billable_duration_seconds_sum_of_points']:.3f} billable "
            f"seconds and an estimated benchmark-normalized cost of {currency} "
            f"{total['estimated_cost']:.6f}.",
            "",
            *session_lines,
            "",
            "## Metric definitions",
            "",
            "```text",
            "measured GPU-seconds = measured wall seconds × GPU count",
            "estimated point cost = billable GPU-seconds × GPU price per second",
            "cost per completed request = estimated point cost / completions",
            "cost per compliant request = estimated point cost / individually "
            "compliant completions",
            "cost per million output tokens = estimated point cost / output "
            "tokens × 1,000,000",
            "```",
            "",
            "Failed requests would remain in billed time but not in the completed or",
            "compliant denominator. A zero denominator is reported as null/`N/A`, not",
            "zero cost.",
            "",
            "## Limitations",
            "",
            "1. The GPU-hour price is a user-supplied estimate and varies by host "
            "and time.",
            "2. Storage and bandwidth prices were not captured, so session "
            "overhead cannot be decomposed.",
            "3. Point estimates exclude warm-up and all time outside measured "
            "E005 windows.",
            "4. The actual session cost is an approximate credit-balance change, "
            "not an invoice line-item total.",
            "5. No tested point passed the aggregate SLO; E006 makes no "
            "sustainable-capacity claim.",
            "6. Results apply only to the recorded model, workload, hardware, "
            "backend, and SLO.",
            "7. E005 measured one workload shape, so E006 compares load-point "
            "economics, not cost differences between workload profiles.",
            "",
        ]
    )
    return "\n".join(lines)


def execute_e006(config: EconomicsConfig) -> tuple[Path, dict[str, Any]]:
    source_path = Path(config.source_summary)
    if not source_path.is_file():
        raise FileNotFoundError(f"E005 summary not found: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    summary = analyze_e005_summary(source, config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )
    (output_dir / "e006_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "e006_report.md").write_text(
        render_report(summary, source_path), encoding="utf-8"
    )
    generate_e006_plots(
        summary["points"],
        output_dir / "plots",
        currency=config.assumptions.currency,
    )
    return output_dir, summary


def print_results(summary: dict[str, Any]) -> None:
    currency = summary["assumptions"]["currency"]
    print("=" * 112)
    print(" E006: BENCHMARK-NORMALIZED SERVING ECONOMICS")
    print("=" * 112)
    print(
        f"{'Offered':>8} | {'Cost':>10} | {'/1k done':>12} | "
        f"{'/1k SLO':>12} | {'/1M out tok':>13} | {'SLO':>5}"
    )
    print("-" * 112)
    for point in summary["points"]:
        item = point["cost_metrics"]
        print(
            f"{point['offered_rate']:>8.2f} | "
            f"{item['estimated_cost']:>10.6f} | "
            f"{item['cost_per_1000_completed_requests']:>12.4f} | "
            f"{item['cost_per_1000_slo_compliant_requests']:>12.4f} | "
            f"{item['cost_per_million_output_tokens']:>13.4f} | "
            f"{('PASS' if point['aggregate_slo_met'] else 'FAIL'):>5}"
        )
    print("=" * 112)
    print(f"Currency: {currency}")
    print(summary["sustainable_cost_note"])


def main() -> None:
    args = create_parser().parse_args()
    try:
        config = load_economics_config(args.config)
        if args.source_summary:
            config = replace(config, source_summary=args.source_summary)
        if args.output_dir:
            config = replace(config, output_dir=args.output_dir)
        if args.gpu_price_per_hour is not None:
            config = replace(
                config,
                assumptions=replace(
                    config.assumptions,
                    gpu_price_per_hour=args.gpu_price_per_hour,
                ),
            )
        output_dir, summary = execute_e006(config)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print_results(summary)
    print(f"Artifacts: {output_dir}")


if __name__ == "__main__":
    main()
