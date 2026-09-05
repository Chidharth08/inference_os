"""Integration unit tests for 1D parameter sweep runner."""

import asyncio
import json
from pathlib import Path

import httpx

from inference_os.config import BenchmarkConfig, SweepConfig
from inference_os.runner.sweep import execute_sweep
from tests.test_benchmark_engine import MockWordTokenizer


def test_execute_sweep_mock(tmp_path: Path) -> None:
    """Verify complete 1D parameter sweep flow with mock transport and tokenizer."""

    def mock_sse_handler(request: httpx.Request) -> httpx.Response:
        content = (
            'data: {"choices": [{"text": "Alpha"}]}\n\n'
            'data: {"choices": [{"text": " Beta"}]}\n\n'
            'data: {"choices": [{"text": " Gamma"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=content)

    transport = httpx.MockTransport(mock_sse_handler)
    client = httpx.AsyncClient(transport=transport)
    tokenizer = MockWordTokenizer()

    base_config = BenchmarkConfig(
        model="test-model",
        max_output_tokens=10,
        num_requests=2,
        warmup_requests=1,
        seed=42,
        output_dir=str(tmp_path),
    )

    sweep_config = SweepConfig(
        sweep_param="prompt_tokens",
        sweep_values=(16, 32),
        base_config=base_config,
        experiment_id="E001A_TEST",
    )

    async def _run() -> None:
        sweep_dir, point_results = await execute_sweep(
            sweep_config=sweep_config,
            tokenizer=tokenizer,
            client=client,
            generate_plots=True,
        )

        assert sweep_dir.is_dir()
        assert (sweep_dir / "sweep_config.json").exists()
        assert (sweep_dir / "sweep_summary.json").exists()

        # Points verification
        assert len(point_results) == 2
        assert point_results[0]["param_value"] == 16
        assert point_results[1]["param_value"] == 32

        # Subdirectories verification
        assert (sweep_dir / "prompt_tokens_16").is_dir()
        assert (sweep_dir / "prompt_tokens_32").is_dir()

        # Plots verification
        plots_dir = sweep_dir / "plots"
        assert plots_dir.is_dir()
        assert (plots_dir / "ttft_vs_input_tokens.png").exists()
        assert (plots_dir / "e2e_vs_input_tokens.png").exists()
        assert (plots_dir / "tpot_vs_input_tokens.png").exists()

        # Check summary file contents
        with open(sweep_dir / "sweep_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["experiment_id"] == "E001A_TEST"
            assert summary["status"] == "SUCCESS"
            assert summary["total_points"] == 2
            assert summary["successful_points"] == 2
            assert summary["failed_points"] == 0
            assert summary["sweep_param"] == "prompt_tokens"
            assert summary["sweep_values"] == [16, 32]
            assert len(summary["points"]) == 2
            assert summary["points"][0]["success"] is True
            assert summary["points"][0]["benchmark"]["total_requests"] == 2
            assert summary["points"][0]["benchmark"]["ttft_stats"] is not None
            assert summary["points"][0]["benchmark"]["tpot_stats"] is not None

    asyncio.run(_run())


def test_execute_sweep_all_failed(tmp_path: Path) -> None:
    """Verify that a failing sweep records failure status and preserves errors."""

    def mock_sse_error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error: Out of Memory")

    transport = httpx.MockTransport(mock_sse_error_handler)
    client = httpx.AsyncClient(transport=transport)
    tokenizer = MockWordTokenizer()

    base_config = BenchmarkConfig(
        model="test-model",
        max_output_tokens=10,
        num_requests=2,
        warmup_requests=1,
        seed=42,
        output_dir=str(tmp_path),
    )

    sweep_config = SweepConfig(
        sweep_param="prompt_tokens",
        sweep_values=(16, 32),
        base_config=base_config,
        experiment_id="E001A_FAIL_TEST",
    )

    async def _run() -> None:
        sweep_dir, point_results = await execute_sweep(
            sweep_config=sweep_config,
            tokenizer=tokenizer,
            client=client,
            generate_plots=True,
        )

        assert sweep_dir.is_dir()
        assert len(point_results) == 2

        # Both points must be marked as failed
        assert point_results[0]["success"] is False
        assert point_results[0]["benchmark"]["successful_requests"] == 0
        assert point_results[0]["benchmark"]["failed_requests"] == 2
        assert point_results[0]["benchmark"]["errors"] is not None
        assert len(point_results[0]["benchmark"]["errors"]) > 0

        assert point_results[1]["success"] is False
        assert point_results[1]["benchmark"]["successful_requests"] == 0

        # Summary JSON validation
        with open(sweep_dir / "sweep_summary.json", "r", encoding="utf-8") as f:
            summary = json.load(f)
            assert summary["status"] == "FAILED"
            assert summary["total_points"] == 2
            assert summary["successful_points"] == 0
            assert summary["failed_points"] == 2

        # Raw requests.jsonl in point directory must contain the error messages
        pt16_reqs = Path(point_results[0]["run_dir"]) / "requests.jsonl"
        assert pt16_reqs.exists()
        with open(pt16_reqs, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) == 3  # 1 warmup + 2 measured
            for line in lines:
                assert line["success"] is False
                assert line["error_message"] is not None
                assert (
                    "500" in line["error_message"]
                    or "Internal Server Error" in line["error_message"]
                )

        # No plots should be generated when all points fail
        plots_dir = sweep_dir / "plots"
        assert not plots_dir.exists()

    asyncio.run(_run())
