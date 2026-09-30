"""Offline tests for V4 prefix plans and vLLM cache metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference_os.telemetry.vllm_metrics import (
    VLLMMetricsSnapshot,
    compute_prefix_cache_delta,
    parse_prometheus_text,
    persist_cache_metric_artifacts,
    validate_cache_server_config,
    validate_pristine_cache_state,
)
from inference_os.workloads.prefix import (
    common_prefix_length,
    prepare_reuse_prompt_plan,
)
from inference_os.workloads.spec import PromptReuseConfig, RequestSpec


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


def test_shared_prefix_plan_is_exact_and_warmup_is_disjoint() -> None:
    tokenizer = WordTokenizer()
    specs = [RequestSpec(64, 8) for _ in range(4)]
    warmups, measured = prepare_reuse_prompt_plan(
        tokenizer,
        warmup_specs=[RequestSpec(64, 8), RequestSpec(64, 8)],
        measured_specs=specs,
        prompt_reuse=PromptReuseConfig(
            mode="shared_prefix",
            shared_prefix_tokens=48,
            cache_block_size_tokens=8,
        ),
        seed=42,
        prefix_caching_enabled=True,
    )

    assert len(warmups) == 2
    assert len(measured) == 4
    assert all(len(item.token_ids) == 64 for item in [*warmups, *measured])
    assert measured[0].expected_cache_state == "cold"
    assert all(item.expected_cache_state == "warm" for item in measured[1:])
    assert all(item.actual_reusable_prefix_tokens >= 48 for item in measured[1:])
    assert all(
        common_prefix_length(warmup.token_ids, measured_item.token_ids) < 8
        for warmup in warmups
        for measured_item in measured
    )
    assert len({item.prompt_token_sha256 for item in measured}) == 4


def test_unique_prefix_plan_never_shares_a_cacheable_block() -> None:
    tokenizer = WordTokenizer()
    _, measured = prepare_reuse_prompt_plan(
        tokenizer,
        warmup_specs=[],
        measured_specs=[RequestSpec(48, 8) for _ in range(20)],
        prompt_reuse=PromptReuseConfig(mode="unique_prefix", cache_block_size_tokens=8),
        seed=7,
        prefix_caching_enabled=True,
    )

    assert measured[0].expected_cache_state == "cold"
    assert all(
        item.expected_cache_state == "no_cacheable_prefix" for item in measured[1:]
    )
    for index, item in enumerate(measured):
        assert all(
            common_prefix_length(item.token_ids, previous.token_ids) < 8
            for previous in measured[:index]
        )


@pytest.mark.parametrize(
    "config",
    [
        PromptReuseConfig(),
        PromptReuseConfig(mode="unique_prefix", cache_block_size_tokens=16),
        PromptReuseConfig(
            mode="shared_prefix",
            shared_prefix_tokens=32,
            reuse_group_id="group-a",
        ),
    ],
)
def test_prompt_reuse_config_valid_modes(config: PromptReuseConfig) -> None:
    assert config.mode in {"none", "unique_prefix", "shared_prefix"}


def test_prompt_reuse_config_rejects_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="requires positive"):
        PromptReuseConfig(mode="shared_prefix")
    with pytest.raises(ValueError, match="only valid"):
        PromptReuseConfig(mode="unique_prefix", shared_prefix_tokens=16)
    with pytest.raises(ValueError, match="positive integer"):
        PromptReuseConfig(mode="unique_prefix", cache_block_size_tokens=0)


def _snapshot(
    raw: str, captured_at: str = "2026-09-30T00:00:00+00:00"
) -> VLLMMetricsSnapshot:
    return VLLMMetricsSnapshot(
        captured_at=captured_at,
        raw_text=raw,
        samples=parse_prometheus_text(raw),
    )


def test_vllm_metrics_parse_config_and_counter_delta() -> None:
    config_line = (
        'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
        'enable_prefix_caching="True",gpu_memory_utilization="0.9",'
        'model_name="test"} 1\n'
    )
    before = _snapshot(
        config_line
        + """# HELP vllm:prefix_cache_queries Prefix queries
vllm:prefix_cache_queries{model_name="test"} 100
vllm:prefix_cache_hits{model_name="test"} 20
vllm:prompt_tokens{model_name="test"} 200
vllm:prompt_tokens_cached{model_name="test"} 20
vllm:num_preemptions{model_name="test"} 1
vllm:kv_cache_usage_perc{model_name="test"} 0.25
"""
    )
    after = _snapshot(
        config_line
        + """
vllm:prefix_cache_queries{model_name="test"} 500
vllm:prefix_cache_hits{model_name="test"} 300
vllm:prompt_tokens{model_name="test"} 600
vllm:prompt_tokens_cached{model_name="test"} 300
vllm:num_preemptions{model_name="test"} 1
vllm:kv_cache_usage_perc{model_name="test"} 0.50
""",
        captured_at="2026-09-30T00:01:00+00:00",
    )

    config = validate_cache_server_config(
        before, expected_enabled=True, expected_block_size=16
    )
    assert config.cache_dtype == "auto"
    assert config.gpu_memory_utilization == 0.9
    delta = compute_prefix_cache_delta(before, after)
    assert delta.prefix_cache_query_tokens == 400
    assert delta.prefix_cache_hit_tokens == 280
    assert delta.observed_prefix_cache_hit_fraction == 0.7
    assert delta.prompt_tokens == 400
    assert delta.cached_prompt_tokens == 280
    assert delta.preemptions == 0
    assert delta.kv_cache_usage_before == 0.25
    assert delta.kv_cache_usage_after == 0.50


def test_cache_metric_validation_rejects_wrong_server_state() -> None:
    snapshot = _snapshot(
        'vllm:cache_config_info{block_size="16",enable_prefix_caching="False"} 1\n'
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_cache_server_config(
            snapshot, expected_enabled=True, expected_block_size=16
        )


def test_pristine_cache_validation_rejects_prior_activity() -> None:
    pristine = _snapshot("vllm:prefix_cache_queries 0\nvllm:prefix_cache_hits 0\n")
    validate_pristine_cache_state(pristine)
    used = _snapshot("vllm:prefix_cache_queries 16\nvllm:prefix_cache_hits 0\n")
    with pytest.raises(ValueError, match="fresh vLLM server"):
        validate_pristine_cache_state(used)


def test_cache_delta_rejects_counter_reset() -> None:
    before = _snapshot("vllm:prefix_cache_queries 10\n")
    after = _snapshot("vllm:prefix_cache_queries 2\n")
    with pytest.raises(ValueError, match="reset"):
        compute_prefix_cache_delta(before, after)


def test_cache_artifacts_are_persisted_and_summary_is_extended(tmp_path: Path) -> None:
    raw = (
        'vllm:cache_config_info{block_size="16",enable_prefix_caching="True"} 1\n'
        "vllm:prefix_cache_queries 10\n"
        "vllm:prefix_cache_hits 8\n"
    )
    snapshot = _snapshot(raw)
    delta = compute_prefix_cache_delta(
        _snapshot(raw.replace(" 10", " 0").replace(" 8", " 0")), snapshot
    )
    (tmp_path / "summary.json").write_text('{"benchmark": {}}', encoding="utf-8")

    persist_cache_metric_artifacts(
        tmp_path,
        preflight=snapshot,
        before=_snapshot(raw.replace(" 10", " 0").replace(" 8", " 0")),
        after=snapshot,
        delta=delta,
    )

    assert (tmp_path / "cache_metrics_before.prom").is_file()
    assert (tmp_path / "cache_metrics_after.prom").is_file()
    assert (tmp_path / "cache_metrics_delta.json").is_file()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["cache"]["observed_prefix_cache_hit_fraction"] == 0.8
    assert summary["server_cache_config"]["enable_prefix_caching"] is True
