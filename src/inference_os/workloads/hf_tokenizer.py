"""HuggingFace AutoTokenizer wrapper."""

from typing import Any

from transformers import AutoTokenizer


class HFTokenizer:
    """Wrapper around HuggingFace AutoTokenizer satisfying the Tokenizer protocol."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> "HFTokenizer":
        """Load a HuggingFace tokenizer by model name or path."""
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs without adding special tokens."""
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to string skipping special tokens."""
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))

    def count_tokens(self, text: str) -> int:
        """Count tokens in text string."""
        return len(self.encode(text))
