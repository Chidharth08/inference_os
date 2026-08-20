"""Unit and integration tests for vLLM HTTP backend adapter."""

import asyncio
from typing import Callable, List

import httpx
import pytest

from inference_os.backends import vllm_stream_completion
from inference_os.runner import run_single_request


def make_fake_clock(timestamps: List[int]) -> Callable[[], int]:
    """Helper creating a deterministic nanosecond clock from a sequence."""
    times = iter(timestamps)

    def _clock() -> int:
        return next(times)

    return _clock


def test_vllm_stream_completion_success() -> None:
    """Verify parsing of SSE stream from mock vLLM endpoint."""

    async def _test() -> None:
        sse_lines = [
            'data: {"id":"1","choices":[{"text":"Hello"}]}\n\n',
            'data: {"id":"2","choices":[{"text":" world"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        sse_body = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/completions"
            assert request.method == "POST"
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            chunks = []
            async for chunk in vllm_stream_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt="Say hello",
                max_tokens=10,
                base_url="http://mockserver:8000",
                client=client,
            ):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    asyncio.run(_test())


def test_vllm_adapter_e2e_integration_with_runner() -> None:
    """Verify end-to-end integration with run_single_request and deterministic clock."""

    async def _test() -> None:
        sse_lines = [
            'data: {"choices":[{"text":"The"}]}\n\n',
            'data: {"choices":[{"text":" answer"}]}\n\n',
            'data: {"choices":[{"text":" is 42"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        sse_body = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/completions"
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        clock = make_fake_clock([1_000_000_000, 1_250_000_000, 1_900_000_000])

        async with httpx.AsyncClient(transport=transport) as client:
            stream = vllm_stream_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt="What is 6*7?",
                max_tokens=10,
                base_url="http://mockserver:8000",
                client=client,
            )
            measurement = await run_single_request(
                request_id="req-vllm-e2e",
                stream=stream,
                input_tokens=15,
                clock_fn=clock,
            )

        assert measurement.request_id == "req-vllm-e2e"
        assert measurement.start_time_ns == 1_000_000_000
        assert measurement.first_token_time_ns == 1_250_000_000
        assert measurement.completion_time_ns == 1_900_000_000
        assert measurement.input_tokens == 15
        assert measurement.output_tokens == 3
        assert measurement.success is True
        assert measurement.error_message is None
        assert measurement.ttft_seconds == pytest.approx(0.25)
        assert measurement.e2e_latency_seconds == pytest.approx(0.90)

    asyncio.run(_test())


def test_vllm_adapter_malformed_json_before_first_output() -> None:
    """Verify malformed JSON before first output raises observable error."""

    async def _test() -> None:
        sse_lines = [
            "data: {invalid-json-body\n\n",
        ]
        sse_body = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        clock = make_fake_clock([100, 500])

        async with httpx.AsyncClient(transport=transport) as client:
            stream = vllm_stream_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt="Test prompt",
                max_tokens=10,
                base_url="http://mockserver:8000",
                client=client,
            )
            measurement = await run_single_request(
                request_id="req-malformed-prefill",
                stream=stream,
                input_tokens=10,
                clock_fn=clock,
            )

        assert measurement.success is False
        assert measurement.first_token_time_ns is None
        assert measurement.output_tokens == 0
        assert measurement.error_message is not None
        assert "JSONDecodeError" in measurement.error_message or (
            "Expecting" in measurement.error_message
        )

    asyncio.run(_test())


def test_vllm_adapter_malformed_json_after_first_output() -> None:
    """Verify malformed JSON after first output preserves first token time."""

    async def _test() -> None:
        sse_lines = [
            'data: {"choices":[{"text":"First token"}]}\n\n',
            "data: {corrupted-json\n\n",
        ]
        sse_body = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        clock = make_fake_clock([1_000_000, 2_000_000, 5_000_000])

        async with httpx.AsyncClient(transport=transport) as client:
            stream = vllm_stream_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt="Test prompt",
                max_tokens=10,
                base_url="http://mockserver:8000",
                client=client,
            )
            measurement = await run_single_request(
                request_id="req-malformed-decode",
                stream=stream,
                input_tokens=10,
                clock_fn=clock,
            )

        assert measurement.success is False
        assert measurement.start_time_ns == 1_000_000
        assert measurement.first_token_time_ns == 2_000_000
        assert measurement.completion_time_ns == 5_000_000
        assert measurement.output_tokens == 1
        assert measurement.ttft_seconds == pytest.approx(0.001)
        assert measurement.e2e_latency_seconds == pytest.approx(0.004)
        assert measurement.error_message is not None

    asyncio.run(_test())


def test_vllm_adapter_malformed_schema() -> None:
    """Verify non-dict or missing 'choices' in SSE chunk raises observable error."""

    async def _test() -> None:
        sse_lines = [
            'data: ["unexpected array schema"]\n\n',
        ]
        sse_body = "".join(sse_lines).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=sse_body,
                headers={"content-type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        clock = make_fake_clock([100, 200])

        async with httpx.AsyncClient(transport=transport) as client:
            stream = vllm_stream_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt="Test prompt",
                max_tokens=10,
                base_url="http://mockserver:8000",
                client=client,
            )
            measurement = await run_single_request(
                request_id="req-bad-schema",
                stream=stream,
                input_tokens=10,
                clock_fn=clock,
            )

        assert measurement.success is False
        assert measurement.error_message is not None
        assert "Expected JSON object" in measurement.error_message

    asyncio.run(_test())


def test_vllm_adapter_http_error_handling() -> None:
    """Verify error propagation when vLLM server returns an HTTP error status."""

    async def _test() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "Internal Server Error"})

        transport = httpx.MockTransport(handler)
        clock = make_fake_clock([100, 500])

        async with httpx.AsyncClient(transport=transport) as client:
            stream = vllm_stream_completion(
                model="Qwen/Qwen2.5-7B-Instruct",
                prompt="Fail please",
                max_tokens=10,
                base_url="http://mockserver:8000",
                client=client,
            )
            measurement = await run_single_request(
                request_id="req-vllm-err",
                stream=stream,
                input_tokens=10,
                clock_fn=clock,
            )

        assert measurement.success is False
        assert measurement.first_token_time_ns is None
        assert measurement.output_tokens == 0
        assert measurement.error_message is not None
        assert "500" in measurement.error_message

    asyncio.run(_test())
