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


@dataclass(frozen=True, slots=True)
class _PromptMaterial:
    """Original text and the exact token sequence it produces."""

    text: str
    token_ids: tuple[int, ...]


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
    measured_materials: list[_PromptMaterial] = []
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
            measured_materials.append(
                _compose_shared_prompt(
                    tokenizer,
                    prefix_text=prefix_text,
                    shared_ids=shared_ids,
                    total_tokens=spec.target_input_tokens,
                    suffix_seed=seed + 20_000 + index,
                )
            )
    else:
        measured_materials = _generate_block_distinct_prompts(
            tokenizer,
            [spec.target_input_tokens for spec in measured_specs],
            seed=seed + 20_000,
            block_size=block_size,
            forbidden=(),
        )

    warmup_materials = _generate_block_distinct_prompts(
        tokenizer,
        [spec.target_input_tokens for spec in warmup_specs],
        seed=seed + 30_000,
        block_size=block_size,
        forbidden=[item.token_ids for item in measured_materials],
    )
    warmups = [
        _prepared(
            material,
            mode="unique_prefix",
            group_id=None,
            configured_shared=0,
            reusable=0,
            expected_state="disjoint_warmup",
        )
        for material in warmup_materials
    ]

    measured: list[PreparedPrompt] = []
    prior: list[tuple[int, ...]] = []
    for material in measured_materials:
        token_ids = material.token_ids
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
                material,
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


def _compose_shared_prompt(
    tokenizer: Tokenizer,
    *,
    prefix_text: str,
    shared_ids: tuple[int, ...],
    total_tokens: int,
    suffix_seed: int,
) -> _PromptMaterial:
    """Compose text whose final tokenization preserves an exact shared prefix.

    Token IDs from independently encoded fragments cannot safely be concatenated
    and decoded with every BPE tokenizer. Instead, compose text at a stable
    boundary, tokenize the final string, and adjust the suffix length until the
    complete prompt has the requested size.
    """
    for separator in ("\n\n", "\n", " "):
        suffix_length = total_tokens - len(shared_ids)
        for _ in range(32):
            if suffix_length <= 0:
                break
            suffix_text = generate_synthetic_prompt(
                tokenizer,
                suffix_length,
                seed=suffix_seed,
            )
            text = f"{prefix_text}{separator}{suffix_text}"
            token_ids = tuple(tokenizer.encode(text))
            prefix_length = common_prefix_length(token_ids, shared_ids)
            if len(token_ids) == total_tokens and prefix_length >= len(shared_ids):
                return _PromptMaterial(text=text, token_ids=token_ids)
            suffix_length += total_tokens - len(token_ids)
    raise ValueError(
        "could not compose an exact-length prompt while preserving the shared prefix"
    )


def _generate_block_distinct_prompts(
    tokenizer: Tokenizer,
    lengths: Sequence[int],
    *,
    seed: int,
    block_size: int,
    forbidden: Sequence[Sequence[int]],
) -> list[_PromptMaterial]:
    generated: list[_PromptMaterial] = []
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
            comparisons = [*forbidden, *(item.token_ids for item in generated)]
            if all(
                common_prefix_length(token_ids, previous) < block_size
                for previous in comparisons
            ):
                generated.append(_PromptMaterial(text=text, token_ids=token_ids))
                break
        else:
            raise ValueError(
                "could not construct prompts unique within the first cacheable block"
            )
    return generated


def _prepared(
    material: _PromptMaterial,
    *,
    mode: str,
    group_id: str | None,
    configured_shared: int,
    reusable: int,
    expected_state: str,
) -> PreparedPrompt:
    text = material.text
    token_ids = material.token_ids
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
