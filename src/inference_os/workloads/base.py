"""Base interfaces and protocols for workloads and tokenizers."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol defining the minimal tokenizer interface."""

    def encode(self, text: str) -> list[int]:
        """Convert text string into a list of integer token IDs."""
        ...

    def decode(self, token_ids: list[int]) -> str:
        """Convert a list of integer token IDs back into a string."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        ...
