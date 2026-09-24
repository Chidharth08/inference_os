"""High-level orchestration engine for end-to-end benchmark execution."""

from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import httpx

from inference_os.backends.vllm import vllm_stream_completion
from inference_os.config import BenchmarkConfig
from inference_os.results.persistence import save_benchmark_run
from inference_os.runner.benchmark import (
    BenchmarkResult,
    run_benchmark,
)
from inference_os.telemetry.environment import capture_environment
from inference_os.telemetry.gpu import GPUTelemetrySampler, GPUTelemetrySummary
from inference_os.workloads.base import Tokenizer
from inference_os.workloads.spec import (
    TokenLengthDistribution,
    generate_request_specs,
)
from inference_os.workloads.synthetic import generate_synthetic_prompt


async def execute_benchmark(
    config: BenchmarkConfig,
    tokenizer: Optional[Tokenizer] = None,
    client: Optional[httpx.AsyncClient] = None,
) -> tuple[Path, BenchmarkResult, Optional[GPUTelemetrySummary]]:
    """Execute a complete end-to-end benchmark run and persist results to disk.

    Steps:
    1. Load tokenizer (or use provided instance).
    2. Generate synthetic prompt of exact token length `config.prompt_tokens`.
    3. Start background GPU telemetry sampling.
    4. Execute benchmark with closed-loop concurrency (W warmup + N measured requests).
    5. Aggregate GPU telemetry and request measurements.
    6. Capture hardware and software environment metadata.
    7. Persist run directory (`config.json`, `environment.json`, `summary.json`,
       `requests.jsonl`, `telemetry.jsonl`).

    Args:
        config: Benchmark configuration parameters.
        tokenizer: Optional Tokenizer instance (defaults to HFTokenizer for model).
        client: Optional httpx.AsyncClient (for connection reuse or mock transport).

    Returns:
        Tuple of (run_dir_path, benchmark_result, gpu_summary).
    """
    # 1. Initialize tokenizer
    if tokenizer is None:
        from inference_os.workloads.hf_tokenizer import HFTokenizer

        tokenizer = HFTokenizer.from_pretrained(config.model)

    # 2. Realize the complete request plan before measurement begins.
    if config.workload is None:
        input_distribution = TokenLengthDistribution.fixed(config.prompt_tokens)
        output_distribution = TokenLengthDistribution.fixed(config.max_output_tokens)
    else:
        input_distribution = config.workload.input_tokens
        output_distribution = config.workload.max_output_tokens

    sampling_mode = (
        config.workload.sampling_mode if config.workload is not None else "iid"
    )
    if sampling_mode == "stratified":
        # Stratify each phase independently so the measured distribution is not
        # distorted by removing the warm-up prefix from one combined plan.
        warmup_specs = generate_request_specs(
            num_requests=config.warmup_requests,
            seed=config.seed + 1,
            input_tokens=input_distribution,
            max_output_tokens=output_distribution,
            sampling_mode=sampling_mode,
        )
        measured_specs = generate_request_specs(
            num_requests=config.num_requests,
            seed=config.seed,
            input_tokens=input_distribution,
            max_output_tokens=output_distribution,
            sampling_mode=sampling_mode,
        )
        all_specs = [*warmup_specs, *measured_specs]
    else:
        all_specs = generate_request_specs(
            num_requests=config.warmup_requests + config.num_requests,
            seed=config.seed,
            input_tokens=input_distribution,
            max_output_tokens=output_distribution,
            sampling_mode=sampling_mode,
        )
        warmup_specs = all_specs[: config.warmup_requests]
        measured_specs = all_specs[config.warmup_requests :]

    # Keep legacy fixed-size runs behaviorally identical: V1 reuses one prompt.
    # Named V2 profiles use distinct, deterministic content for every request.
    if config.workload is None:
        fixed_prompt = generate_synthetic_prompt(
            tokenizer=tokenizer,
            num_tokens=config.prompt_tokens,
            seed=config.seed,
        )
        prompts = [fixed_prompt] * len(all_specs)
    else:
        prompts = [
            generate_synthetic_prompt(
                tokenizer=tokenizer,
                num_tokens=spec.target_input_tokens,
                seed=config.seed + index,
            )
            for index, spec in enumerate(all_specs)
        ]
    actual_input_counts = [tokenizer.count_tokens(prompt) for prompt in prompts]

    # 3. Define request factory
    async def request_factory(
        request_id: str,
        index: int,
        is_warmup: bool,
    ) -> tuple[AsyncGenerator[str, None], int, Optional[int | Callable[[], int]]]:
        plan_index = index if is_warmup else config.warmup_requests + index
        spec = all_specs[plan_index]
        prompt = prompts[plan_index]
        actual_input_tokens = actual_input_counts[plan_index]
        collected_chunks: list[str] = []

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
                collected_chunks.append(chunk)
                yield chunk

        def get_actual_output_tokens() -> int:
            if not collected_chunks:
                return 0
            full_text = "".join(collected_chunks)
            return tokenizer.count_tokens(full_text)

        return stream_wrapper(), actual_input_tokens, get_actual_output_tokens

    # 4. Execute benchmark with background GPU telemetry
    sampler = GPUTelemetrySampler(
        interval_seconds=config.telemetry_interval_seconds,
        device_index=config.device_index,
    )

    async with sampler:
        result = await run_benchmark(
            request_factory=request_factory,
            num_requests=config.num_requests,
            concurrency=config.concurrency,
            warmup_requests=config.warmup_requests,
        )

    gpu_summary = sampler.get_summary()
    gpu_samples = sampler.get_samples()

    # 5. Capture environment metadata
    environment = capture_environment()

    # 6. Save benchmark run to disk
    run_dir = save_benchmark_run(
        config=config,
        environment=environment,
        result=result,
        gpu_summary=gpu_summary,
        gpu_samples=gpu_samples,
        warmup_workload_specs=warmup_specs,
        workload_specs=measured_specs,
    )

    return run_dir, result, gpu_summary
