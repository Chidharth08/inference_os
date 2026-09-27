"""End-to-end orchestration for one open-loop offered-load point."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import AsyncGenerator, Callable

import httpx

from inference_os.backends.vllm import vllm_stream_completion
from inference_os.config import BenchmarkConfig, SLOConfig
from inference_os.metrics.slo import SLOSummary, evaluate_slo
from inference_os.results.persistence import save_benchmark_run
from inference_os.runner.load import (
    OpenLoopResult,
    constant_rate_arrival_offsets,
    run_open_loop,
)
from inference_os.telemetry.environment import capture_environment
from inference_os.telemetry.gpu import GPUTelemetrySampler, GPUTelemetrySummary
from inference_os.workloads.base import Tokenizer
from inference_os.workloads.spec import TokenLengthDistribution, generate_request_specs
from inference_os.workloads.synthetic import generate_synthetic_prompt


async def execute_open_loop_benchmark(
    config: BenchmarkConfig,
    *,
    request_rate: float,
    duration_seconds: float,
    max_in_flight: int,
    max_drain_seconds: float,
    slo: SLOConfig,
    tokenizer: Tokenizer | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Path, OpenLoopResult, SLOSummary, GPUTelemetrySummary | None]:
    """Realize, run, evaluate, and persist one constant-rate load point."""
    if tokenizer is None:
        from inference_os.workloads.hf_tokenizer import HFTokenizer

        tokenizer = HFTokenizer.from_pretrained(config.model)

    request_count = len(constant_rate_arrival_offsets(request_rate, duration_seconds))
    if config.workload is None:
        input_distribution = TokenLengthDistribution.fixed(config.prompt_tokens)
        output_distribution = TokenLengthDistribution.fixed(config.max_output_tokens)
        sampling_mode = "iid"
    else:
        input_distribution = config.workload.input_tokens
        output_distribution = config.workload.max_output_tokens
        sampling_mode = config.workload.sampling_mode

    if sampling_mode == "stratified":
        warmup_specs = generate_request_specs(
            num_requests=config.warmup_requests,
            seed=config.seed + 1,
            input_tokens=input_distribution,
            max_output_tokens=output_distribution,
            sampling_mode=sampling_mode,
        )
        measured_specs = generate_request_specs(
            num_requests=request_count,
            seed=config.seed,
            input_tokens=input_distribution,
            max_output_tokens=output_distribution,
            sampling_mode=sampling_mode,
        )
        all_specs = [*warmup_specs, *measured_specs]
    else:
        all_specs = generate_request_specs(
            num_requests=config.warmup_requests + request_count,
            seed=config.seed,
            input_tokens=input_distribution,
            max_output_tokens=output_distribution,
            sampling_mode=sampling_mode,
        )
        warmup_specs = all_specs[: config.warmup_requests]
        measured_specs = all_specs[config.warmup_requests :]

    prompts = [
        generate_synthetic_prompt(
            tokenizer,
            spec.target_input_tokens,
            config.seed + index,
        )
        for index, spec in enumerate(all_specs)
    ]
    input_counts = [tokenizer.count_tokens(prompt) for prompt in prompts]

    async def request_factory(
        request_id: str, index: int, is_warmup: bool
    ) -> tuple[AsyncGenerator[str, None], int, int | Callable[[], int]]:
        plan_index = index if is_warmup else config.warmup_requests + index
        spec = all_specs[plan_index]
        prompt = prompts[plan_index]
        chunks: list[str] = []

        async def stream_wrapper() -> AsyncGenerator[str, None]:
            async for chunk in vllm_stream_completion(
                model=config.model,
                prompt=prompt,
                max_tokens=spec.max_output_tokens,
                temperature=config.temperature,
                request_timeout_seconds=config.request_timeout_seconds,
                base_url=config.base_url,
                client=client,
            ):
                chunks.append(chunk)
                yield chunk

        def output_tokens() -> int:
            return tokenizer.count_tokens("".join(chunks)) if chunks else 0

        return stream_wrapper(), input_counts[plan_index], output_tokens

    sampler = GPUTelemetrySampler(
        interval_seconds=config.telemetry_interval_seconds,
        device_index=config.device_index,
    )
    async with sampler:
        result = await run_open_loop(
            request_factory,
            request_rate,
            duration_seconds,
            max_in_flight=max_in_flight,
            max_drain_seconds=max_drain_seconds,
            warmup_requests=config.warmup_requests,
        )
    gpu_summary = sampler.get_summary()
    slo_summary = evaluate_slo(
        result.measured_requests,
        slo,
        result.measured_duration_seconds,
    )
    load_summary = {
        "arrival_process": "deterministic_constant_rate",
        "offered_rate": request_rate,
        "realized_offered_rate": len(result.measured_requests) / duration_seconds,
        "offered_duration_seconds": duration_seconds,
        "measured_duration_seconds": result.measured_duration_seconds,
        "offered_requests": len(result.measured_requests),
        "dispatched_requests": result.dispatched_requests,
        "dropped_requests": result.dropped_requests,
        "timed_out_requests": result.timed_out_requests,
        "max_observed_in_flight": result.max_observed_in_flight,
    }
    run_dir = save_benchmark_run(
        config=config,
        environment=capture_environment(),
        result=result,  # OpenLoopResult intentionally matches persistence protocol.
        gpu_summary=gpu_summary,
        gpu_samples=sampler.get_samples(),
        warmup_workload_specs=warmup_specs,
        workload_specs=measured_specs,
        extra_config={
            "request_rate": request_rate,
            "duration_seconds": duration_seconds,
            "max_in_flight": max_in_flight,
            "max_drain_seconds": max_drain_seconds,
            "slo": asdict(slo),
        },
        extra_summary={"load": load_summary, "slo": asdict(slo_summary)},
        in_flight_samples=result.in_flight_samples,
    )
    return run_dir, result, slo_summary, gpu_summary
