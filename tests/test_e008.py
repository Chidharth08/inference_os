"""Offline integration tests for E008 prefix-fraction sensitivity."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from inference_os.config import BenchmarkConfig


class WordTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        for word in text.strip().split():
            if word not in self._vocab:
                token_id = len(self._vocab) + 1
                self._vocab[word] = token_id
                self._words[token_id] = word
            result.append(self._vocab[word])
        return result

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(
            self._words.get(token_id, f"unk_{token_id}") for token_id in token_ids
        )

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


def _load_runner() -> ModuleType:
    path = Path("experiments/E008-prefix-reuse-sensitivity/run_e008.py")
    spec = importlib.util.spec_from_file_location("run_e008", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("path", "cache_enabled"),
    [
        ("configs/e008_cache_off.yaml", False),
        ("configs/e008_cache_on.yaml", True),
    ],
)
def test_e008_base_configs_load(path: str, cache_enabled: bool) -> None:
    runner = _load_runner()
    config = runner.load_e008_config(path)
    assert isinstance(config, BenchmarkConfig)
    assert config.enable_prefix_caching is cache_enabled
    assert config.concurrency == 1
    assert config.workload is not None
    assert config.workload.input_tokens.values == (4096,)
    assert config.workload.max_output_tokens.values == (64,)


@pytest.mark.parametrize(
    ("fraction", "tokens", "mode"),
    [
        (0, 0, "unique_prefix"),
        (25, 1024, "shared_prefix"),
        (50, 2048, "shared_prefix"),
        (75, 3072, "shared_prefix"),
        (90, 3680, "shared_prefix"),
    ],
)
def test_e008_fraction_resolution(fraction: int, tokens: int, mode: str) -> None:
    runner = _load_runner()
    base = runner.load_e008_config("configs/e008_cache_on.yaml")
    config, metadata = runner.build_fraction_config(base, fraction)
    assert config.workload is not None
    assert config.workload.prompt_reuse.mode == mode
    assert config.workload.prompt_reuse.shared_prefix_tokens == tokens
    assert metadata["resolved_shared_prefix_tokens"] == tokens
    assert metadata["resolved_shared_fraction"] == tokens / 4096
    assert runner.condition_name(config, fraction) == (
        f"fraction_{fraction:03d}_cache_on"
    )


def test_e008_condition_validates_hits_against_realized_plan(tmp_path: Path) -> None:
    runner = _load_runner()
    base = runner.load_e008_config("configs/e008_cache_on.yaml")
    base = BenchmarkConfig.from_dict(
        {
            **base.to_dict(),
            "num_requests": 4,
            "warmup_requests": 2,
            "output_dir": str(tmp_path),
            "workload": {
                **base.to_dict()["workload"],
                "input_tokens": {"values": [64], "weights": [1.0]},
                "max_output_tokens": {"values": [4], "weights": [1.0]},
                "prompt_reuse": {
                    "mode": "unique_prefix",
                    "shared_prefix_tokens": 0,
                    "cache_block_size_tokens": 8,
                },
            },
        }
    )
    config, fraction_metadata = runner.build_fraction_config(base, 50)
    tokenizer = WordTokenizer()
    query_tokens = 0
    hit_tokens = 0
    prior_prompts: list[list[str]] = []

    def common_prefix(left: list[str], right: list[str]) -> int:
        count = 0
        for left_token, right_token in zip(left, right):
            if left_token != right_token:
                break
            count += 1
        return count

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal query_tokens, hit_tokens
        if request.method == "GET" and request.url.path == "/metrics":
            body = (
                'vllm:cache_config_info{block_size="8",cache_dtype="auto",'
                'enable_prefix_caching="True",gpu_memory_utilization="0.9"} 1\n'
                f"vllm:prefix_cache_queries {query_tokens}\n"
                f"vllm:prefix_cache_hits {hit_tokens}\n"
                f"vllm:prompt_tokens {query_tokens}\n"
                f"vllm:prompt_tokens_cached {hit_tokens}\n"
                "vllm:num_preemptions 0\n"
                "vllm:kv_cache_usage_perc 0.1\n"
            )
            return httpx.Response(200, text=body)
        if request.method == "POST" and request.url.path == "/v1/completions":
            payload = json.loads(request.content)
            prompt = str(payload["prompt"]).split()
            reusable = max(
                (common_prefix(prompt, previous) for previous in prior_prompts),
                default=0,
            )
            query_tokens += len(prompt)
            hit_tokens += (reusable // 8) * 8
            prior_prompts.append(prompt)
            return httpx.Response(
                200,
                text='data: {"choices": [{"text": "done now"}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(404)

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            run_dir, condition = await runner.execute_e008_condition(
                config,
                fraction_metadata,
                tokenizer=tokenizer,
                client=client,
            )
        assert condition["success"] is True
        assert condition["mechanism_valid"] is True
        assert condition["fraction"]["resolved_shared_prefix_tokens"] == 32
        assert condition["cache"]["prefix_cache_query_tokens"] == 256
        assert condition["cache"]["prefix_cache_hit_tokens"] == 32 * 3
        assert condition["reuse_plan"]["expected_cache_hit_tokens"] == 32 * 3
        assert condition["cache_phase_latency"]["cold"]["count"] == 1
        assert condition["cache_phase_latency"]["warm"]["count"] == 3
        assert (run_dir / "e008_condition.json").is_file()

    asyncio.run(run())


def test_e008_summarizer_requires_ten_paired_conditions(tmp_path: Path) -> None:
    runner = _load_runner()
    run_dirs: list[Path] = []
    for fraction in runner.REQUESTED_FRACTIONS:
        for cache in ("off", "on"):
            run_dir = tmp_path / f"fraction-{fraction}-{cache}"
            run_dir.mkdir()
            cache_on = cache == "on"
            hit_fraction = (fraction / 100.0) * (9 / 10) if cache_on else None
            ttft = 0.1 * (1.0 - (fraction / 200.0)) if cache_on else 0.1
            resolved_tokens = int(4096 * fraction / 100) // 16 * 16
            condition = {
                "experiment_id": "E008",
                "condition": f"fraction_{fraction:03d}_cache_{cache}",
                "success": True,
                "mechanism_valid": True,
                "fraction": {
                    "requested_shared_fraction_percent": fraction,
                    "resolved_shared_prefix_tokens": resolved_tokens,
                    "resolved_shared_fraction": resolved_tokens / 4096,
                    "total_prompt_tokens": 4096,
                },
                "reuse_plan": {"measured_requests": 10},
                "benchmark": {
                    "ttft_stats": {"p50": ttft, "p95": ttft * 1.1},
                    "e2e_latency_stats": {"p50": 0.2, "p95": 0.3},
                    "tpot_stats": {"p50": 0.01, "p95": 0.02},
                },
                "cache": {
                    "observed_prefix_cache_hit_fraction": hit_fraction,
                    "prefix_cache_hit_tokens": (
                        hit_fraction * 40960 if hit_fraction is not None else 0
                    ),
                },
                "cache_phase_latency": {
                    "disabled" if not cache_on else (
                        "no_cacheable_prefix" if fraction == 0 else "warm"
                    ): {"p50": ttft}
                },
            }
            (run_dir / "e008_condition.json").write_text(
                json.dumps(condition), encoding="utf-8"
            )
            run_dirs.append(run_dir)

    comparison_dir, summary = runner.summarize_e008_runs(
        run_dirs, output_root=tmp_path
    )
    assert summary["status"] == "SUCCESS"
    assert summary["observed_hit_fraction_nondecreasing"] is True
    assert len(summary["points"]) == 5
    assert summary["points"][3]["ttft_p50_change_percent"] == pytest.approx(-37.5)
    assert (comparison_dir / "e008_summary.json").is_file()
    assert (comparison_dir / "plots/ttft_vs_shared_fraction.png").is_file()
    assert (comparison_dir / "plots/reuse_vs_ttft_improvement.png").is_file()
    assert (comparison_dir / "plots/e2e_tpot_vs_shared_fraction.png").is_file()

    with pytest.raises(ValueError, match="exactly ten"):
        runner.summarize_e008_runs(run_dirs[:-1], output_root=tmp_path)
