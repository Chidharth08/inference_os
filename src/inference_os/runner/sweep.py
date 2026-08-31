"""1D Parameter Sweep runner for controlled scaling benchmarks."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx

from inference_os.config import SweepConfig
from inference_os.reports.plots import generate_e001a_plots
from inference_os.runner.engine import execute_benchmark
from inference_os.workloads.base import Tokenizer
from inference_os.workloads.hf_tokenizer import HFTokenizer


async def execute_sweep(
    sweep_config: SweepConfig,
    tokenizer: Optional[Tokenizer] = None,
    client: Optional[httpx.AsyncClient] = None,
    generate_plots: bool = True,
) -> tuple[Path, list[dict[str, Any]]]:
    """Execute a 1D sequential parameter sweep across controlled configurations.

    Steps:
    1. Initialize parent sweep directory.
    2. Initialize tokenizer once for the target model.
    3. Iterate through sweep points sequentially, running execute_benchmark() for each.
    4. Persist raw measurements and telemetry for each point in point-specific
       subdirectories.
    5. Aggregate and persist top-level `sweep_summary.json` and `sweep_config.json`.
    6. Automatically render sweep plots into `plots/` subdirectory.

    Args:
        sweep_config: 1D parameter sweep configuration specification.
        tokenizer: Optional pre-initialized Tokenizer instance.
        client: Optional httpx.AsyncClient instance (for testing/reuse).
        generate_plots: Whether to generate Matplotlib plot images upon completion.

    Returns:
        Tuple of (parent_sweep_dir_path, list_of_point_summary_dicts).
    """
    # 1. Create parent sweep directory
    base_dir = Path(sweep_config.base_config.output_dir)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand_id = uuid.uuid4().hex[:8]
    sweep_run_id = f"{sweep_config.experiment_id}_{ts_str}_{rand_id}"
    sweep_dir = base_dir / sweep_run_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # 2. Save sweep_config.json
    config_file = sweep_dir / "sweep_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(sweep_config.to_dict(), f, indent=2)

    # 3. Initialize tokenizer once
    if tokenizer is None:
        tokenizer = HFTokenizer.from_pretrained(sweep_config.base_config.model)

    point_results: list[dict[str, Any]] = []

    # 4. Sequentially execute each point in the sweep
    point_configs = sweep_config.generate_point_configs()
    for param_val, point_cfg in point_configs:
        point_subdir_name = f"{sweep_config.sweep_param}_{param_val}"
        point_output_dir = sweep_dir / point_subdir_name

        # Ensure output directory for point is set
        from dataclasses import replace
        point_cfg_with_dir = replace(point_cfg, output_dir=str(point_output_dir))

        run_dir, result, gpu_summary = await execute_benchmark(
            config=point_cfg_with_dir,
            tokenizer=tokenizer,
            client=client,
        )

        point_summary_entry: dict[str, Any] = {
            "param_name": sweep_config.sweep_param,
            "param_value": param_val,
            "run_dir": str(run_dir),
            "benchmark": asdict(result.summary),
            "gpu": asdict(gpu_summary) if gpu_summary is not None else None,
        }
        point_results.append(point_summary_entry)

    # 5. Persist sweep_summary.json
    sweep_summary_payload: dict[str, Any] = {
        "sweep_run_id": sweep_run_id,
        "experiment_id": sweep_config.experiment_id,
        "sweep_param": sweep_config.sweep_param,
        "sweep_values": list(sweep_config.sweep_values),
        "base_config": sweep_config.base_config.to_dict(),
        "points": point_results,
    }
    summary_file = sweep_dir / "sweep_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(sweep_summary_payload, f, indent=2)

    # 6. Render plots
    if generate_plots:
        plots_dir = sweep_dir / "plots"
        generate_e001a_plots(point_results, plots_dir)

    return sweep_dir, point_results
