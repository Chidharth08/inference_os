"""Token-level tests for E009 application-shaped prompt plans."""

from __future__ import annotations

import pytest

from inference_os.workloads.application import prepare_application_prompt_plan
from inference_os.workloads.spec import PromptReuseConfig, RequestSpec


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


def test_chat_plan_preserves_growing_conversation_prefixes() -> None:
    tokenizer = WordTokenizer()
    specs = [
        *(RequestSpec(64, 8) for _ in range(4)),
        *(RequestSpec(80, 8) for _ in range(4)),
    ]
    _, measured = prepare_application_prompt_plan(
        tokenizer,
        warmup_specs=[RequestSpec(64, 8)],
        measured_specs=specs,
        prompt_reuse=PromptReuseConfig(
            mode="application",
            application_profile="chat_like",
            relationship_group_count=4,
            stable_prefix_tokens=32,
            cache_block_size_tokens=8,
        ),
        seed=42,
        prefix_caching_enabled=True,
    )
    assert [item.sequence_index for item in measured] == [1, 1, 1, 1, 2, 2, 2, 2]
    assert measured[0].expected_cache_state == "cold"
    assert measured[1].actual_reusable_prefix_tokens >= 32
    assert measured[4].actual_reusable_prefix_tokens >= 64
    assert measured[4].relationship_id == "chat_like-1"


def test_rag_layout_changes_cacheable_prefix_not_content_shape() -> None:
    tokenizer = WordTokenizer()
    specs = [RequestSpec(96, 8) for _ in range(4)]
    plans = {}
    for layout, stable in (("friendly", 72), ("hostile", 16)):
        _, measured = prepare_application_prompt_plan(
            tokenizer,
            warmup_specs=[],
            measured_specs=specs,
            prompt_reuse=PromptReuseConfig(
                mode="application",
                application_profile="rag_like",
                prompt_layout=layout,
                stable_prefix_tokens=stable,
                document_tokens=56,
                cache_block_size_tokens=8,
            ),
            seed=42,
            prefix_caching_enabled=True,
        )
        plans[layout] = measured
    assert all(len(item.token_ids) == 96 for plan in plans.values() for item in plan)
    assert plans["friendly"][1].actual_reusable_prefix_tokens >= 72
    assert plans["hostile"][1].actual_reusable_prefix_tokens >= 16
    assert plans["hostile"][1].actual_reusable_prefix_tokens < 72
    assert plans["friendly"][0].relationship_id == plans["hostile"][0].relationship_id


def test_summarization_has_short_stable_prefix() -> None:
    tokenizer = WordTokenizer()
    _, measured = prepare_application_prompt_plan(
        tokenizer,
        warmup_specs=[],
        measured_specs=[RequestSpec(96, 8) for _ in range(3)],
        prompt_reuse=PromptReuseConfig(
            mode="application",
            application_profile="summarization_like",
            stable_prefix_tokens=16,
            cache_block_size_tokens=8,
        ),
        seed=42,
        prefix_caching_enabled=True,
    )
    assert measured[1].actual_reusable_prefix_tokens >= 16
    assert measured[1].actual_reusable_prefix_tokens < 96


def test_agent_plan_grows_prior_trajectory() -> None:
    tokenizer = WordTokenizer()
    specs = [
        *(RequestSpec(80, 8) for _ in range(2)),
        *(RequestSpec(96, 16) for _ in range(2)),
        *(RequestSpec(112, 24) for _ in range(2)),
    ]
    _, measured = prepare_application_prompt_plan(
        tokenizer,
        warmup_specs=[],
        measured_specs=specs,
        prompt_reuse=PromptReuseConfig(
            mode="application",
            application_profile="agent_like",
            relationship_group_count=2,
            stable_prefix_tokens=64,
            cache_block_size_tokens=8,
        ),
        seed=42,
        prefix_caching_enabled=True,
    )
    assert measured[2].actual_reusable_prefix_tokens >= 80
    assert measured[4].actual_reusable_prefix_tokens >= 96
    assert measured[4].sequence_index == 3


def test_application_config_rejects_invalid_combinations() -> None:
    with pytest.raises(ValueError, match="application_profile"):
        PromptReuseConfig(mode="application", stable_prefix_tokens=16)
    with pytest.raises(ValueError, match="friendly or hostile"):
        PromptReuseConfig(
            mode="application",
            application_profile="rag_like",
            stable_prefix_tokens=16,
            document_tokens=32,
        )
