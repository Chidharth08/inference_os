"""Serving backend abstraction and adapter modules."""

from inference_os.backends.vllm import vllm_stream_completion

__all__ = ["vllm_stream_completion"]
