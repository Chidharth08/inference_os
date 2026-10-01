"""Deterministic application-shaped prompt relationships for E009."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence

from inference_os.workloads.base import Tokenizer
from inference_os.workloads.prefix import (
    PreparedPrompt,
    _compose_shared_prompt,
    _generate_block_distinct_prompts,
    common_prefix_length,
)
from inference_os.workloads.spec import PromptReuseConfig, RequestSpec
from inference_os.workloads.synthetic import generate_synthetic_prompt


def prepare_application_prompt_plan(
    tokenizer: Tokenizer,
    *,
    warmup_specs: Sequence[RequestSpec],
    measured_specs: Sequence[RequestSpec],
    prompt_reuse: PromptReuseConfig,
    seed: int,
    prefix_caching_enabled: bool,
) -> tuple[list[PreparedPrompt], list[PreparedPrompt]]:
    """Build disjoint warm-ups and one verified application-shaped plan."""
    if prompt_reuse.mode != "application" or prompt_reuse.application_profile is None:
        raise ValueError("application prompt plans require application mode")
    block_size = prompt_reuse.cache_block_size_tokens
    warmup_materials = _generate_block_distinct_prompts(
        tokenizer,
        [spec.target_input_tokens for spec in warmup_specs],
        seed=seed + 90_000,
        block_size=block_size,
        forbidden=(),
    )
    warmups = [
        _make_prepared(
            text=item.text,
            token_ids=item.token_ids,
            config=prompt_reuse,
            reusable=0,
            expected_state="disjoint_warmup",
            relationship_id=None,
            sequence_index=None,
        )
        for item in warmup_materials
    ]

    profile = prompt_reuse.application_profile
    if profile in {"chat_like", "agent_like"}:
        raw_plan = _growing_plan(
            tokenizer,
            specs=measured_specs,
            config=prompt_reuse,
            seed=seed,
        )
    elif profile == "rag_like":
        raw_plan = _rag_plan(
            tokenizer,
            specs=measured_specs,
            config=prompt_reuse,
            seed=seed,
        )
    else:
        raw_plan = _shared_instruction_plan(
            tokenizer,
            specs=measured_specs,
            config=prompt_reuse,
            seed=seed,
        )

    measured: list[PreparedPrompt] = []
    prior: list[tuple[int, ...]] = []
    for text, token_ids, relationship_id, sequence_index in raw_plan:
        reusable = max(
            (common_prefix_length(token_ids, previous) for previous in prior),
            default=0,
        )
        if prior and reusable < prompt_reuse.stable_prefix_tokens:
            raise ValueError("application plan lost its configured stable prefix")
        if not prefix_caching_enabled:
            state = "disabled"
        elif not prior:
            state = "cold"
        elif reusable >= block_size:
            state = "warm"
        else:
            state = "no_cacheable_prefix"
        measured.append(
            _make_prepared(
                text=text,
                token_ids=token_ids,
                config=prompt_reuse,
                reusable=reusable,
                expected_state=state,
                relationship_id=relationship_id,
                sequence_index=sequence_index,
            )
        )
        prior.append(token_ids)
    return warmups, measured


def _growing_plan(
    tokenizer: Tokenizer,
    *,
    specs: Sequence[RequestSpec],
    config: PromptReuseConfig,
    seed: int,
) -> list[tuple[str, tuple[int, ...], str, int]]:
    stable_text = generate_synthetic_prompt(
        tokenizer, config.stable_prefix_tokens, seed=seed + 10_000
    )
    stable_ids = tuple(tokenizer.encode(stable_text))
    prior_by_group: dict[int, tuple[str, tuple[int, ...]]] = {}
    result: list[tuple[str, tuple[int, ...], str, int]] = []
    for index, spec in enumerate(specs):
        group = index % config.relationship_group_count
        sequence_index = index // config.relationship_group_count
        prefix_text, prefix_ids = prior_by_group.get(group, (stable_text, stable_ids))
        if spec.target_input_tokens <= len(prefix_ids):
            raise ValueError("growing application schedule must increase prompt length")
        material = _compose_shared_prompt(
            tokenizer,
            prefix_text=prefix_text,
            shared_ids=prefix_ids,
            total_tokens=spec.target_input_tokens,
            suffix_seed=seed + 20_000 + index,
        )
        relationship_id = f"{config.application_profile}-{group + 1}"
        result.append(
            (material.text, material.token_ids, relationship_id, sequence_index + 1)
        )
        prior_by_group[group] = (material.text, material.token_ids)
    return result


def _shared_instruction_plan(
    tokenizer: Tokenizer,
    *,
    specs: Sequence[RequestSpec],
    config: PromptReuseConfig,
    seed: int,
) -> list[tuple[str, tuple[int, ...], str, int]]:
    prefix_text = generate_synthetic_prompt(
        tokenizer, config.stable_prefix_tokens, seed=seed + 10_000
    )
    prefix_ids = tuple(tokenizer.encode(prefix_text))
    result: list[tuple[str, tuple[int, ...], str, int]] = []
    for index, spec in enumerate(specs):
        material = _compose_shared_prompt(
            tokenizer,
            prefix_text=prefix_text,
            shared_ids=prefix_ids,
            total_tokens=spec.target_input_tokens,
            suffix_seed=seed + 20_000 + index,
        )
        result.append(
            (material.text, material.token_ids, "summarization-instruction", index + 1)
        )
    return result


def _rag_plan(
    tokenizer: Tokenizer,
    *,
    specs: Sequence[RequestSpec],
    config: PromptReuseConfig,
    seed: int,
) -> list[tuple[str, tuple[int, ...], str, int]]:
    minimum = min((spec.target_input_tokens for spec in specs), default=0)
    if config.stable_prefix_tokens >= minimum:
        raise ValueError("RAG stable content must leave room for a unique question")
    system_tokens = (
        config.stable_prefix_tokens - config.document_tokens
        if config.prompt_layout == "friendly"
        else config.stable_prefix_tokens
    )
    if system_tokens <= 0:
        raise ValueError("RAG system prefix must be positive")
    system_text = generate_synthetic_prompt(
        tokenizer, system_tokens, seed=seed + 10_000
    )
    system_ids = tuple(tokenizer.encode(system_text))
    prefix_text, prefix_ids, document_text = _compose_system_document(
        tokenizer,
        system_text=system_text,
        system_ids=system_ids,
        document_tokens=config.document_tokens,
        document_seed=seed + 11_000,
    )
    document_hash = _token_hash(tokenizer.encode(document_text))
    result: list[tuple[str, tuple[int, ...], str, int]] = []
    for index, spec in enumerate(specs):
        if config.prompt_layout == "friendly":
            composed = _compose_shared_prompt(
                tokenizer,
                prefix_text=prefix_text,
                shared_ids=prefix_ids,
                total_tokens=spec.target_input_tokens,
                suffix_seed=seed + 20_000 + index,
            )
            text, token_ids = composed.text, composed.token_ids
        else:
            text, token_ids = _compose_hostile_rag_prompt(
                tokenizer,
                system_text=system_text,
                system_ids=system_ids,
                document_text=document_text,
                total_tokens=spec.target_input_tokens,
                question_seed=seed + 20_000 + index,
            )
        result.append(
            (
                text,
                token_ids,
                f"rag-document-{document_hash[:12]}",
                index + 1,
            )
        )
    return result


def _compose_system_document(
    tokenizer: Tokenizer,
    *,
    system_text: str,
    system_ids: tuple[int, ...],
    document_tokens: int,
    document_seed: int,
) -> tuple[str, tuple[int, ...], str]:
    target = len(system_ids) + document_tokens
    candidate_length = document_tokens
    for _ in range(64):
        document_text = generate_synthetic_prompt(
            tokenizer, candidate_length, seed=document_seed
        )
        text = f"{system_text}\n\n{document_text}"
        token_ids = tuple(tokenizer.encode(text))
        if len(token_ids) == target and common_prefix_length(
            token_ids, system_ids
        ) >= len(system_ids):
            return text, token_ids, document_text
        candidate_length += target - len(token_ids)
    raise ValueError("could not compose exact shared RAG system/document prefix")


def _compose_hostile_rag_prompt(
    tokenizer: Tokenizer,
    *,
    system_text: str,
    system_ids: tuple[int, ...],
    document_text: str,
    total_tokens: int,
    question_seed: int,
) -> tuple[str, tuple[int, ...]]:
    document_count = len(tokenizer.encode(document_text))
    question_length = total_tokens - len(system_ids) - document_count
    for _ in range(64):
        if question_length <= 0:
            break
        question = generate_synthetic_prompt(
            tokenizer, question_length, seed=question_seed
        )
        text = f"{system_text}\n\n{question}\n\n{document_text}"
        token_ids = tuple(tokenizer.encode(text))
        if len(token_ids) == total_tokens and common_prefix_length(
            token_ids, system_ids
        ) >= len(system_ids):
            return text, token_ids
        question_length += total_tokens - len(token_ids)
    raise ValueError("could not compose exact-length cache-hostile RAG prompt")


def _make_prepared(
    *,
    text: str,
    token_ids: tuple[int, ...],
    config: PromptReuseConfig,
    reusable: int,
    expected_state: str,
    relationship_id: str | None,
    sequence_index: int | None,
) -> PreparedPrompt:
    reusable_ids = token_ids[:reusable]
    return PreparedPrompt(
        text=text,
        token_ids=token_ids,
        prompt_reuse_mode="application",
        reuse_group_id=relationship_id,
        configured_shared_prefix_tokens=config.stable_prefix_tokens,
        actual_reusable_prefix_tokens=reusable,
        expected_cache_state=expected_state,
        prompt_token_sha256=_token_hash(token_ids),
        reusable_prefix_sha256=(_token_hash(reusable_ids) if reusable_ids else None),
        application_profile=config.application_profile,
        prompt_layout=config.prompt_layout,
        relationship_id=relationship_id,
        sequence_index=sequence_index,
    )


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
