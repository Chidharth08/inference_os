"""Offline integration tests for E009 application-shaped reuse."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from inference_os.config import BenchmarkConfig
from inference_os.workloads.spec import (
    PromptReuseConfig,
    TokenLengthDistribution,
    WorkloadConfig,
)


class WordTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._words: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        result = []
        for word in text.strip().split():
            if word not in self._vocab:
                token_id = len(self._vocab) + 1
                self._vocab[word] = token_id
                self._words[token_id] = word
            result.append(self._vocab[word])
        return result

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._words.get(item, f"unk_{item}") for item in token_ids)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


def _load_runner() -> ModuleType:
    path = Path("experiments/E009-application-prefix-reuse/run_e009.py")
    spec = importlib.util.spec_from_file_location("run_e009", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "path", ["configs/e009_cache_off.yaml", "configs/e009_cache_on.yaml"]
)
def test_e009_base_configs_and_profiles(path: str) -> None:
    runner = _load_runner()
    base = runner.load_e009_config(path)
    for profile in runner.PROFILES:
        config = runner.build_profile_config(base, profile)
        assert config.num_requests == 24
        assert config.workload is not None
        assert config.workload.prompt_reuse.mode == "application"
        assert (
            len(
                config.workload.input_tokens.sample_many(
                    24, __import__("random").Random(), mode="sequence"
                )
            )
            == 24
        )
    agent = runner.build_profile_config(base, "agent_like")
    assert agent.workload is not None
    assert agent.workload.input_tokens.values[-1] == 8064
    hostile = runner.build_profile_config(base, "rag_hostile")
    assert hostile.workload is not None
    assert hostile.workload.prompt_reuse.stable_prefix_tokens == 256
    assert hostile.workload.prompt_reuse.document_tokens == 3072


def test_e009_controlled_condition_validates_exact_hits(tmp_path: Path) -> None:
    runner = _load_runner()
    base = runner.load_e009_config("configs/e009_cache_on.yaml")
    config = BenchmarkConfig.from_dict(
        {
            **base.to_dict(),
            "num_requests": 4,
            "warmup_requests": 1,
            "output_dir": str(tmp_path),
        }
    )
    config = replace_workload_for_test(config)
    tokenizer = WordTokenizer()
    query_tokens = 0
    hit_tokens = 0
    prior: list[list[str]] = []

    def prefix(left: list[str], right: list[str]) -> int:
        count = 0
        for left_item, right_item in zip(left, right):
            if left_item != right_item:
                break
            count += 1
        return count

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal query_tokens, hit_tokens
        if request.method == "GET" and request.url.path == "/metrics":
            return httpx.Response(
                200,
                text=(
                    'vllm:cache_config_info{block_size="8",cache_dtype="auto",'
                    'enable_prefix_caching="True",gpu_memory_utilization="0.9"} 1\n'
                    f"vllm:prefix_cache_queries {query_tokens}\n"
                    f"vllm:prefix_cache_hits {hit_tokens}\n"
                    f"vllm:prompt_tokens {query_tokens}\n"
                    f"vllm:prompt_tokens_cached {hit_tokens}\n"
                    "vllm:num_preemptions 0\n"
                    "vllm:kv_cache_usage_perc 0.1\n"
                ),
            )
        if request.method == "POST" and request.url.path == "/v1/completions":
            words = str(json.loads(request.content)["prompt"]).split()
            reusable = max((prefix(words, item) for item in prior), default=0)
            query_tokens += len(words)
            hit_tokens += reusable // 8 * 8
            prior.append(words)
            return httpx.Response(
                200,
                text='data: {"choices": [{"text": "done now"}]}\n\ndata: [DONE]\n\n',
            )
        return httpx.Response(404)

    async def run() -> None:
        nonlocal query_tokens, hit_tokens, prior
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            run_dir, condition = await runner.execute_e009_condition(
                config,
                profile="rag_friendly",
                tokenizer=tokenizer,
                client=client,
            )
        assert condition["success"] is True
        assert condition["mechanism_valid"] is True
        assert condition["cache"]["prefix_cache_hit_tokens"] == 48 * 3
        assert condition["reuse_plan"]["expected_cache_hit_tokens"] == 48 * 3
        assert (run_dir / "e009_condition.json").is_file()

        query_tokens = 0
        hit_tokens = 0
        prior = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            load_dir, load_condition = await runner.execute_e009_condition(
                config,
                profile="rag_friendly",
                load_rate=100.0,
                tokenizer=tokenizer,
                client=client,
            )
        assert load_condition["stage"] == "open_loop"
        assert load_condition["success"] is True
        assert load_condition["mechanism_valid"] is True
        assert load_condition["load"]["offered_requests"] == 4
        assert load_condition["slo"] is not None
        assert (load_dir / "in_flight.jsonl").is_file()

    asyncio.run(run())


def replace_workload_for_test(config: BenchmarkConfig) -> BenchmarkConfig:
    workload = WorkloadConfig(
        name="test-rag-friendly",
        input_tokens=TokenLengthDistribution.fixed(64),
        max_output_tokens=TokenLengthDistribution.fixed(4),
        prompt_reuse=PromptReuseConfig(
            mode="application",
            application_profile="rag_like",
            prompt_layout="friendly",
            stable_prefix_tokens=48,
            document_tokens=40,
            cache_block_size_tokens=8,
        ),
        sampling_mode="sequence",
    )
    return BenchmarkConfig.from_dict({**config.to_dict(), "workload": workload})


def _fake_condition(profile: str, cache: str, stage: str = "controlled") -> dict:
    cache_on = cache == "on"
    name = f"{profile}_cache_{cache}"
    if stage == "open_loop":
        name += "_rate_1"
    return {
        "experiment_id": "E009",
        "stage": stage,
        "condition": name,
        "profile": profile,
        "success": True,
        "mechanism_valid": True,
        "benchmark": {
            "request_throughput": 2.0 if cache_on else 1.0,
            "ttft_stats": {"p50": 0.05 if cache_on else 0.1, "p95": 0.12},
            "e2e_latency_stats": {"p50": 0.2, "p95": 0.3},
            "tpot_stats": {"p50": 0.01, "p95": 0.02},
        },
        "cache": {"observed_prefix_cache_hit_fraction": 0.5 if cache_on else None},
        "reuse_plan": {"cacheable_tokens_by_sequence_index": {"1": 100, "2": 200}},
        "slo": {"goodput": 1.5 if cache_on else 0.75},
    }


def _persist_conditions(tmp_path: Path, conditions: list[dict]) -> list[Path]:
    tmp_path.mkdir(parents=True)
    paths = []
    for index, condition in enumerate(conditions):
        path = tmp_path / f"condition-{index}"
        path.mkdir()
        (path / "e009_condition.json").write_text(
            json.dumps(condition), encoding="utf-8"
        )
        paths.append(path)
    return paths


def test_e009_controlled_and_load_summarizers(tmp_path: Path) -> None:
    runner = _load_runner()
    controlled = [
        _fake_condition(profile, cache)
        for profile in runner.PROFILES
        for cache in ("off", "on")
    ]
    controlled_paths = _persist_conditions(tmp_path / "controlled", controlled)
    out, summary = runner.summarize_e009_runs(controlled_paths, output_root=tmp_path)
    assert summary["status"] == "SUCCESS"
    assert len(summary["profiles"]) == 5
    assert (out / "plots/latency_by_application_profile.png").is_file()
    assert (out / "plots/cache_reuse_and_ttft_benefit.png").is_file()
    assert (out / "plots/cacheable_prefix_by_sequence_index.png").is_file()

    load = [
        _fake_condition(profile, cache, "open_loop")
        for profile in runner.LOAD_PROFILES
        for cache in ("off", "on")
    ]
    load_paths = _persist_conditions(tmp_path / "load", load)
    load_out, load_summary = runner.summarize_e009_load_runs(
        load_paths, output_root=tmp_path
    )
    assert load_summary["status"] == "SUCCESS"
    assert (load_out / "plots/throughput_and_goodput_by_rag_layout.png").is_file()
    assert (load_out / "plots/open_loop_latency_by_rag_layout.png").is_file()

    with pytest.raises(ValueError, match="ten controlled"):
        runner.summarize_e009_runs(controlled_paths[:-1], output_root=tmp_path)
