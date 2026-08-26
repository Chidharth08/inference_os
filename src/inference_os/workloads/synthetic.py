"""Synthetic prompt generation for controlled workload benchmarking."""

import random
from typing import Sequence

from inference_os.workloads.base import Tokenizer

# Controlled vocabulary of common domain terms
_BASE_VOCABULARY: Sequence[str] = (
    "system architecture performance measurement latency throughput benchmark "
    "inference memory execution evaluation workload parameter concurrency token "
    "prefill decode hardware network telemetry cluster cache optimization context "
    "distribution framework validation analysis baseline capacity timeline request"
).split()


def generate_synthetic_prompt(
    tokenizer: Tokenizer,
    num_tokens: int,
    seed: int = 42,
) -> str:
    """Generate a synthetic text prompt containing exactly `num_tokens` tokens.

    Args:
        tokenizer: Tokenizer instance implementing the Tokenizer protocol.
        num_tokens: Exact number of tokens desired in the prompt.
        seed: Random seed for deterministic text generation.

    Returns:
        A synthetic prompt string where `tokenizer.count_tokens(prompt)`
        equals `num_tokens`.

    Raises:
        ValueError: If num_tokens <= 0.
    """
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}")

    rng = random.Random(seed)

    # Generate an initial pool of words with extra buffer
    words_needed = max(num_tokens * 2, 20)
    generated_words = [rng.choice(_BASE_VOCABULARY) for _ in range(words_needed)]
    raw_text = " ".join(generated_words)

    tokens = tokenizer.encode(raw_text)

    # If initial text wasn't long enough, keep expanding
    while len(tokens) < num_tokens + 10:
        generated_words.extend(
            [rng.choice(_BASE_VOCABULARY) for _ in range(num_tokens)]
        )
        raw_text = " ".join(generated_words)
        tokens = tokenizer.encode(raw_text)

    # Slice token IDs to target length
    target_slice = list(tokens[:num_tokens])
    candidate_text = tokenizer.decode(target_slice)
    actual_count = tokenizer.count_tokens(candidate_text)

    # Fine-tune boundaries if decode-encode roundtrip changes count
    while actual_count < num_tokens:
        target_slice.append(rng.choice(tokens))
        candidate_text = tokenizer.decode(target_slice)
        actual_count = tokenizer.count_tokens(candidate_text)

    while actual_count > num_tokens and len(target_slice) > 1:
        target_slice.pop()
        candidate_text = tokenizer.decode(target_slice)
        actual_count = tokenizer.count_tokens(candidate_text)

    return candidate_text
