"""Workload generation and tokenizer modules."""

from typing import TYPE_CHECKING, Any

from inference_os.workloads.base import Tokenizer
from inference_os.workloads.spec import (
    PromptReuseConfig,
    RequestSpec,
    TokenLengthDistribution,
    WorkloadConfig,
    generate_request_specs,
)
from inference_os.workloads.synthetic import generate_synthetic_prompt

if TYPE_CHECKING:
    from inference_os.workloads.hf_tokenizer import HFTokenizer


def __getattr__(name: str) -> Any:
    """Load the optional Transformers-backed tokenizer only when requested."""
    if name == "HFTokenizer":
        from inference_os.workloads.hf_tokenizer import HFTokenizer

        return HFTokenizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Tokenizer",
    "HFTokenizer",
    "PromptReuseConfig",
    "RequestSpec",
    "TokenLengthDistribution",
    "WorkloadConfig",
    "generate_request_specs",
    "generate_synthetic_prompt",
]
