"""Smoke test script for validating vLLM server with inference_os runner."""

import argparse
import asyncio
import sys
from typing import AsyncGenerator

from inference_os.backends import vllm_stream_completion
from inference_os.runner import run_single_request


async def main() -> int:
    """Run a single-request smoke test against a running vLLM server."""
    parser = argparse.ArgumentParser(
        description="Run a single-request smoke test against a vLLM server."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the running vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model identifier registered on vLLM server",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain why the sky appears blue in two concise sentences.",
        help="Input text prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum output tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--request-id",
        type=str,
        default="smoke-test-001",
        help="Identifier for the test request",
    )

    args = parser.parse_args()

    print(f"Connecting to vLLM server at: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    print("Streaming output: ", end="", flush=True)

    async def stream_wrapper() -> AsyncGenerator[str, None]:
        async for chunk in vllm_stream_completion(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            base_url=args.base_url,
        ):
            print(chunk, end="", flush=True)
            yield chunk
        print()

    measurement = await run_single_request(
        request_id=args.request_id,
        stream=stream_wrapper(),
        input_tokens=0,
    )

    print("-" * 50)
    print("Measurement Results:")
    print(f"  Success: {measurement.success}")
    if measurement.error_message:
        print(f"  Error: {measurement.error_message}")
    if measurement.ttft_seconds is not None:
        ttft_ms = measurement.ttft_seconds * 1000
        print(f"  TTFT: {ttft_ms:.2f} ms ({measurement.ttft_seconds:.4f} s)")
    else:
        print("  TTFT: N/A")
    e2e_ms = measurement.e2e_latency_seconds * 1000
    print(f"  E2E Latency: {e2e_ms:.2f} ms ({measurement.e2e_latency_seconds:.4f} s)")

    return 0 if measurement.success else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
