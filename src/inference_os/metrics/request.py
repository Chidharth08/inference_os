"""Request measurement data structures and derived metrics."""

from dataclasses import dataclass
from typing import Optional

NANOSECONDS_PER_SECOND: float = 1_000_000_000.0


@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    """Raw timing and metadata observations for a single inference request.

    Timestamps are integer nanoseconds from a monotonic clock (e.g. perf_counter_ns).
    """

    request_id: str
    start_time_ns: int
    completion_time_ns: int
    input_tokens: int
    output_tokens: int
    success: bool
    first_token_time_ns: Optional[int] = None
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Enforce timestamp and count invariants upon construction."""
        if self.start_time_ns < 0:
            raise ValueError(
                f"start_time_ns must be non-negative, got {self.start_time_ns}"
            )
        if self.completion_time_ns < self.start_time_ns:
            raise ValueError(
                f"completion_time_ns ({self.completion_time_ns}) cannot precede "
                f"start_time_ns ({self.start_time_ns})"
            )
        if self.input_tokens < 0:
            raise ValueError(
                f"input_tokens must be non-negative, got {self.input_tokens}"
            )
        if self.output_tokens < 0:
            raise ValueError(
                f"output_tokens must be non-negative, got {self.output_tokens}"
            )
        if self.first_token_time_ns is not None:
            if self.first_token_time_ns < self.start_time_ns:
                raise ValueError(
                    f"first_token_time_ns ({self.first_token_time_ns}) cannot precede "
                    f"start_time_ns ({self.start_time_ns})"
                )
            if self.first_token_time_ns > self.completion_time_ns:
                raise ValueError(
                    f"first_token_time_ns ({self.first_token_time_ns}) cannot be after "
                    f"completion_time_ns ({self.completion_time_ns})"
                )

    @property
    def ttft_seconds(self) -> Optional[float]:
        """Time to First Token (TTFT) in seconds.

        Returns None if first_token_time_ns was not observed.
        """
        if self.first_token_time_ns is None:
            return None
        return (self.first_token_time_ns - self.start_time_ns) / NANOSECONDS_PER_SECOND

    @property
    def e2e_latency_seconds(self) -> float:
        """End-to-end request latency in seconds."""
        return (self.completion_time_ns - self.start_time_ns) / NANOSECONDS_PER_SECOND

    @property
    def tpot_seconds(self) -> Optional[float]:
        """Time Per Output Token (TPOT) for decode phase in seconds.

        Derived strictly from validated output token counts:
        TPOT = (completion_time - first_token_time) / (output_tokens - 1)
        Returns None if first token was not observed or output_tokens <= 1.
        """
        if self.first_token_time_ns is None or self.output_tokens <= 1:
            return None
        decode_tokens = self.output_tokens - 1
        decode_time_ns = self.completion_time_ns - self.first_token_time_ns
        return (decode_time_ns / decode_tokens) / NANOSECONDS_PER_SECOND

