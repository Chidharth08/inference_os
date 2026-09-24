"""Unit tests for workload generation and tokenizers."""

from unittest.mock import MagicMock

import pytest

from inference_os.workloads import (
    HFTokenizer,
    RequestSpec,
    Tokenizer,
    TokenLengthDistribution,
    generate_request_specs,
    generate_synthetic_prompt,
)


class MockWordTokenizer:
    """Deterministic whitespace-based mock tokenizer for fast local tests."""

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        words = text.strip().split()
        token_ids: list[int] = []
        for word in words:
            if word not in self._vocab:
                new_id = len(self._vocab) + 1
                self._vocab[word] = new_id
                self._id_to_word[new_id] = word
            token_ids.append(self._vocab[word])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        words = [self._id_to_word.get(tid, f"unk_{tid}") for tid in token_ids]
        return " ".join(words)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))


def test_mock_tokenizer_satisfies_protocol() -> None:
    """Verify MockWordTokenizer conforms to Tokenizer protocol."""
    tok = MockWordTokenizer()
    assert isinstance(tok, Tokenizer)


@pytest.mark.parametrize("target_tokens", [1, 5, 10, 64, 128, 512, 1024])
def test_generate_synthetic_prompt_exact_counts(target_tokens: int) -> None:
    """Verify synthetic prompt generation produces exact token counts."""
    tok = MockWordTokenizer()
    prompt = generate_synthetic_prompt(tok, num_tokens=target_tokens, seed=42)

    assert tok.count_tokens(prompt) == target_tokens
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_generate_synthetic_prompt_determinism() -> None:
    """Verify same seed produces identical prompts and different seeds differ."""
    tok = MockWordTokenizer()
    prompt_a1 = generate_synthetic_prompt(tok, num_tokens=64, seed=123)
    prompt_a2 = generate_synthetic_prompt(tok, num_tokens=64, seed=123)
    prompt_b = generate_synthetic_prompt(tok, num_tokens=64, seed=456)

    assert prompt_a1 == prompt_a2
    assert prompt_a1 != prompt_b


def test_generate_synthetic_prompt_invalid_counts() -> None:
    """Verify ValueError when requesting non-positive token counts."""
    tok = MockWordTokenizer()
    with pytest.raises(ValueError, match="num_tokens must be positive"):
        generate_synthetic_prompt(tok, num_tokens=0)

    with pytest.raises(ValueError, match="num_tokens must be positive"):
        generate_synthetic_prompt(tok, num_tokens=-10)


def test_hf_tokenizer_wrapper() -> None:
    """Verify HFTokenizer methods delegate to underlying tokenizer."""
    mock_hf = MagicMock()
    mock_hf.encode.return_value = [101, 2054, 102]
    mock_hf.decode.return_value = "hello world"

    wrapper = HFTokenizer(mock_hf)
    assert isinstance(wrapper, Tokenizer)

    tokens = wrapper.encode("hello world")
    assert tokens == [101, 2054, 102]
    mock_hf.encode.assert_called_once_with("hello world", add_special_tokens=False)

    text = wrapper.decode([101, 2054, 102])
    assert text == "hello world"
    mock_hf.decode.assert_called_once_with([101, 2054, 102], skip_special_tokens=True)

    count = wrapper.count_tokens("hello world")
    assert count == 3


def test_request_spec_generation_is_deterministic() -> None:
    """The same distributions and seed must realize the identical request plan."""
    input_dist = TokenLengthDistribution(
        values=(128, 512, 2048), weights=(0.2, 0.5, 0.3)
    )
    output_dist = TokenLengthDistribution(
        values=(64, 128, 256), weights=(0.5, 0.3, 0.2)
    )

    first = generate_request_specs(
        num_requests=100,
        seed=42,
        input_tokens=input_dist,
        max_output_tokens=output_dist,
    )
    second = generate_request_specs(
        num_requests=100,
        seed=42,
        input_tokens=input_dist,
        max_output_tokens=output_dist,
    )
    different_seed = generate_request_specs(
        num_requests=100,
        seed=43,
        input_tokens=input_dist,
        max_output_tokens=output_dist,
    )

    assert first == second
    assert first != different_seed
    assert all(isinstance(spec, RequestSpec) for spec in first)
    assert {spec.target_input_tokens for spec in first} <= {128, 512, 2048}
    assert {spec.max_output_tokens for spec in first} <= {64, 128, 256}


def test_fixed_distribution_preserves_v1_request_shape() -> None:
    specs = generate_request_specs(
        num_requests=5,
        seed=42,
        input_tokens=TokenLengthDistribution.fixed(512),
        max_output_tokens=TokenLengthDistribution.fixed(128),
    )
    assert specs == [RequestSpec(512, 128)] * 5


def test_stratified_sampling_matches_weighted_means_and_is_deterministic() -> None:
    input_dist = TokenLengthDistribution(
        values=(1024, 4096, 7168), weights=(0.2, 0.6, 0.2)
    )
    output_dist = TokenLengthDistribution(
        values=(128, 512, 896), weights=(0.2, 0.6, 0.2)
    )

    first = generate_request_specs(
        num_requests=50,
        seed=42,
        input_tokens=input_dist,
        max_output_tokens=output_dist,
        sampling_mode="stratified",
    )
    second = generate_request_specs(
        num_requests=50,
        seed=42,
        input_tokens=input_dist,
        max_output_tokens=output_dist,
        sampling_mode="stratified",
    )

    assert first == second
    assert sum(spec.target_input_tokens for spec in first) / len(first) == 4096
    assert sum(spec.max_output_tokens for spec in first) / len(first) == 512
    assert [spec.target_input_tokens for spec in first].count(1024) == 10
    assert [spec.target_input_tokens for spec in first].count(4096) == 30
    assert [spec.target_input_tokens for spec in first].count(7168) == 10
    assert input_dist.mean == 4096
    assert output_dist.mean == 512
    assert input_dist.std_dev > 0


def test_invalid_sampling_mode_is_rejected() -> None:
    distribution = TokenLengthDistribution.fixed(128)
    with pytest.raises(ValueError, match="sampling mode"):
        generate_request_specs(
            num_requests=2,
            seed=42,
            input_tokens=distribution,
            max_output_tokens=distribution,
            sampling_mode="unknown",
        )


@pytest.mark.parametrize(
    ("values", "weights", "message"),
    [
        ((), (), "values cannot be empty"),
        ((128, 256), (1.0,), "equal lengths"),
        ((0,), (1.0,), "values must be positive"),
        ((128,), (-1.0,), "finite and non-negative"),
        ((128,), (0.0,), "positive sum"),
    ],
)
def test_token_distribution_validation(
    values: tuple[int, ...], weights: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TokenLengthDistribution(values=values, weights=weights)
