"""Deterministic, token-verified prompt plans for prefix-cache experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

from inference_os.workloads.base import Tokenizer
from inference_os.workloads.spec import PromptReuseConfig, RequestSpec
from inference_os.workloads.synthetic import generate_synthetic_prompt


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """One realized prompt plus auditable prefix-reuse metadata."""

    text: str
    token_ids: tuple[int, ...]
    prompt_reuse_mode: str
    reuse_group_id: str | None
    configured_shared_prefix_tokens: int
    actual_reusable_prefix_tokens: int
    expected_cache_state: str
    prompt_token_sha256: str
    reusable_prefix_sha256: str | None

    def metadata(self) -> dict[str, object]:
        """Return JSON-serializable metadata for ``workload.jsonl``."""
        return {
            "prompt_reuse_mode": self.prompt_reuse_mode,
            "reuse_group_id": self.reuse_group_id,
            "configured_shared_prefix_tokens": self.configured_shared_prefix_tokens,
            "actual_reusable_prefix_tokens": self.actual_reusable_prefix_tokens,
            "expected_cache_state": self.expected_cache_state,
            "prompt_token_sha256": self.prompt_token_sha256,
            "reusable_prefix_sha256": self.reusable_prefix_sha256,
        }


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the exact common-prefix length of two token sequences."""
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def prepare_reuse_prompt_plan(
    tokenizer: Tokenizer,
    *,
    warmup_specs: Sequence[RequestSpec],
    measured_specs: Sequence[RequestSpec],
    prompt_reuse: PromptReuseConfig,
    seed: int,
    prefix_caching_enabled: bool,
) -> tuple[list[PreparedPrompt], list[PreparedPrompt]]:
    """Build disjoint warm-up prompts and a verified measured prompt plan.

    This path is used only by explicit V4 reuse modes. Existing ``mode: none``
    workloads retain their original V2 prompt generation behavior.
    """
    if prompt_reuse.mode not in {"unique_prefix", "shared_prefix"}:
        raise ValueError("reuse prompt plans require an explicit V4 reuse mode")

    block_size = prompt_reuse.cache_block_size_tokens
    measured_tokens: list[tuple[int, ...]] = []
    if prompt_reuse.mode == "shared_prefix":
        minimum_length = min(
            (spec.target_input_tokens for spec in measured_specs), default=0
        )
        if prompt_reuse.shared_prefix_tokens >= minimum_length:
            raise ValueError("shared_prefix_tokens must leave a positive unique suffix")
        prefix_text = generate_synthetic_prompt(
            tokenizer,
            prompt_reuse.shared_prefix_tokens,
            seed=seed + 10_000,
        )
        shared_ids = tuple(tokenizer.encode(prefix_text))
        if len(shared_ids) != prompt_reuse.shared_prefix_tokens:
            raise ValueError("shared-prefix tokenization did not match its target")
        for index, spec in enumerate(measured_specs):
            suffix_length = spec.target_input_tokens - len(shared_ids)
            suffix_text = generate_synthetic_prompt(
                tokenizer,
                suffix_length,
                seed=seed + 20_000 + index,
            )
            suffix_ids = tuple(tokenizer.encode(suffix_text))
            measured_tokens.append(
                _roundtrip_exact(tokenizer, (*shared_ids, *suffix_ids))
            )
    else:
        measured_tokens = _generate_block_distinct_sequences(
            tokenizer,
            [spec.target_input_tokens for spec in measured_specs],
            seed=seed + 20_000,
            block_size=block_size,
            forbidden=(),
        )

    warmup_tokens = _generate_block_distinct_sequences(
        tokenizer,
        [spec.target_input_tokens for spec in warmup_specs],
        seed=seed + 30_000,
        block_size=block_size,
        forbidden=measured_tokens,
    )
    warmups = [
        _prepared(
            tokenizer,
            token_ids,
            mode="unique_prefix",
            group_id=None,
            configured_shared=0,
            reusable=0,
            expected_state="disjoint_warmup",
        )
        for token_ids in warmup_tokens
    ]

    measured: list[PreparedPrompt] = []
    prior: list[tuple[int, ...]] = []
    for token_ids in measured_tokens:
        reusable = max(
            (common_prefix_length(token_ids, previous) for previous in prior),
            default=0,
        )
        if prompt_reuse.mode == "unique_prefix" and reusable >= block_size:
            raise ValueError("unique-prefix plan accidentally shares a cacheable block")
        if prompt_reuse.mode == "shared_prefix" and prior:
            if reusable < prompt_reuse.shared_prefix_tokens:
                raise ValueError(
                    "shared-prefix plan does not preserve the target prefix"
                )
        if not prefix_caching_enabled:
            state = "disabled"
        elif not prior:
            state = "cold"
        elif reusable >= block_size:
            state = "warm"
        else:
            state = "no_cacheable_prefix"
        measured.append(
            _prepared(
                tokenizer,
                token_ids,
                mode=prompt_reuse.mode,
                group_id=(
                    prompt_reuse.reuse_group_id
                    if prompt_reuse.mode == "shared_prefix"
                    else None
                ),
                configured_shared=prompt_reuse.shared_prefix_tokens,
                reusable=reusable,
                expected_state=state,
            )
        )
        prior.append(token_ids)
    return warmups, measured


def _generate_block_distinct_sequences(
    tokenizer: Tokenizer,
    lengths: Sequence[int],
    *,
    seed: int,
    block_size: int,
    forbidden: Sequence[Sequence[int]],
) -> list[tuple[int, ...]]:
    generated: list[tuple[int, ...]] = []
    for index, length in enumerate(lengths):
        for attempt in range(1_000):
            text = generate_synthetic_prompt(
                tokenizer,
                length,
                seed=seed + index * 1_000 + attempt,
            )
            token_ids = tuple(tokenizer.encode(text))
            if len(token_ids) != length:
                continue
            comparisons = [*forbidden, *generated]
            if all(
                common_prefix_length(token_ids, previous) < block_size
                for previous in comparisons
            ):
                generated.append(token_ids)
                break
        else:
            raise ValueError(
                "could not construct prompts unique within the first cacheable block"
            )
    return generated


def _roundtrip_exact(tokenizer: Tokenizer, token_ids: Sequence[int]) -> tuple[int, ...]:
    expected = tuple(token_ids)
    text = tokenizer.decode(list(expected))
    actual = tuple(tokenizer.encode(text))
    if actual != expected:
        raise ValueError(
            "tokenizer decode/encode round trip changed the planned prefix tokens"
        )
    return actual


def _prepared(
    tokenizer: Tokenizer,
    token_ids: tuple[int, ...],
    *,
    mode: str,
    group_id: str | None,
    configured_shared: int,
    reusable: int,
    expected_state: str,
) -> PreparedPrompt:
    text = tokenizer.decode(list(token_ids))
    if tuple(tokenizer.encode(text)) != token_ids:
        raise ValueError("prepared prompt is not token-stable")
    reusable_ids = token_ids[:reusable]
    return PreparedPrompt(
        text=text,
        token_ids=token_ids,
        prompt_reuse_mode=mode,
        reuse_group_id=group_id,
        configured_shared_prefix_tokens=configured_shared,
        actual_reusable_prefix_tokens=reusable,
        expected_cache_state=expected_state,
        prompt_token_sha256=_token_hash(token_ids),
        reusable_prefix_sha256=(_token_hash(reusable_ids) if reusable_ids else None),
    )


def _token_hash(token_ids: Sequence[int]) -> str:
    payload = json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
