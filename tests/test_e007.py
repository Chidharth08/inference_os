"""Offline integration tests for E007 prefix-caching validation."""

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
    path = Path("experiments/E007-prefix-caching-baseline/run_e007.py")
    spec = importlib.util.spec_from_file_location("run_e007", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("path", "condition"),
    [
        ("configs/e007_cache_off_unique.yaml", "unique_prefix_cache_off"),
        ("configs/e007_cache_on_unique.yaml", "unique_prefix_cache_on"),
        ("configs/e007_cache_off_repeated.yaml", "shared_prefix_cache_off"),
        ("configs/e007_cache_on_repeated.yaml", "shared_prefix_cache_on"),
    ],
)
def test_e007_configs_load(path: str, condition: str) -> None:
    runner = _load_runner()
    config = runner.load_e007_config(path)
    assert isinstance(config, BenchmarkConfig)
    assert runner.condition_name(config) == condition
    assert config.concurrency == 1
    assert config.workload is not None
    assert config.workload.input_tokens.values == (4096,)
    assert config.workload.max_output_tokens.values == (64,)


def test_e007_condition_captures_measured_only_cache_delta(tmp_path: Path) -> None:
    runner = _load_runner()
    config = runner.load_e007_config("configs/e007_cache_on_repeated.yaml")
    config = BenchmarkConfig.from_dict(
        {
            **config.to_dict(),
            "num_requests": 4,
            "warmup_requests": 2,
            "output_dir": str(tmp_path),
            "workload": {
                **config.to_dict()["workload"],
                "input_tokens": {"values": [64], "weights": [1.0]},
                "max_output_tokens": {"values": [4], "weights": [1.0]},
                "prompt_reuse": {
                    "mode": "shared_prefix",
                    "shared_prefix_tokens": 48,
                    "reuse_group_id": "test-prefix",
                    "cache_block_size_tokens": 8,
                },
            },
        }
    )
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
            run_dir, condition = await runner.execute_e007_condition(
                config, tokenizer=tokenizer, client=client
            )
        assert condition["success"] is True
        assert condition["mechanism_valid"] is True
        # The two disjoint warmups occurred before the baseline snapshot, so only
        # four measured 64-token prompts appear in the delta.
        assert condition["cache"]["prefix_cache_query_tokens"] == 256
        assert condition["cache"]["prefix_cache_hit_tokens"] >= 48 * 3
        assert condition["cache_phase_latency"]["cold"]["count"] == 1
        assert condition["cache_phase_latency"]["warm"]["count"] == 3
        workload = [
            json.loads(line)
            for line in (run_dir / "workload.jsonl").read_text().splitlines()
        ]
        measured = [item for item in workload if not item["is_warmup"]]
        assert measured[0]["expected_cache_state"] == "cold"
        assert measured[1]["expected_cache_state"] == "warm"
        assert measured[1]["actual_reusable_prefix_tokens"] >= 48
        assert (run_dir / "cache_metrics_preflight.prom").is_file()
        assert (run_dir / "e007_condition.json").is_file()

    asyncio.run(run())


def test_e007_summarizer_requires_and_combines_four_conditions(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    run_dirs: list[Path] = []
    for index, condition in enumerate(sorted(runner.EXPECTED_CONDITIONS)):
        run_dir = tmp_path / f"condition-{index}"
        run_dir.mkdir()
        cache_on = condition.endswith("_on")
        shared = condition.startswith("shared")
        record = {
            "experiment_id": "E007",
            "condition": condition,
            "success": True,
            "mechanism_valid": True,
            "benchmark": {
                "ttft_stats": {"p50": 0.05 if cache_on and shared else 0.1},
                "e2e_latency_stats": {"p50": 0.2, "p95": 0.3},
                "tpot_stats": {"p50": 0.01, "p95": 0.02},
            },
            "cache": {
                "observed_prefix_cache_hit_fraction": 0.75
                if shared and cache_on
                else 0,
                "prefix_cache_hit_tokens": 100 if shared and cache_on else 0,
            },
        }
        (run_dir / "e007_condition.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
        run_dirs.append(run_dir)

    comparison_dir, summary = runner.summarize_e007_runs(run_dirs, output_root=tmp_path)
    assert summary["status"] == "SUCCESS"
    assert summary["comparisons"]["shared_prefix_ttft_p50_change_percent"] == -50
    assert (comparison_dir / "e007_summary.json").is_file()
    assert (comparison_dir / "plots/latency_by_condition.png").is_file()

    with pytest.raises(ValueError, match="exactly four"):
        runner.summarize_e007_runs(run_dirs[:3], output_root=tmp_path)
