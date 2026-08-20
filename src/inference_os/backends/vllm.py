"""vLLM OpenAI-compatible HTTP backend streaming adapter."""

import json
from typing import AsyncGenerator, Optional

import httpx


async def vllm_stream_completion(
    model: str,
    prompt: str,
    max_tokens: int,
    base_url: str = "http://localhost:8000",
    client: Optional[httpx.AsyncClient] = None,
) -> AsyncGenerator[str, None]:
    """Stream text completions from a vLLM OpenAI-compatible server.

    Args:
        model: Name of the model registered on the vLLM server.
        prompt: Input text prompt.
        max_tokens: Maximum tokens to generate.
        base_url: Base URL of the vLLM OpenAI-compatible server.
        client: Optional httpx.AsyncClient instance for testing or connection reuse.

    Yields:
        Non-empty generated text chunks as SSE data events arrive.
    """
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
    }

    close_client = False
    if client is None:
        client = httpx.AsyncClient()
        close_client = True

    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_content = line[6:].strip()
                    if data_content == "[DONE]":
                        break
                    parsed = json.loads(data_content)
                    if not isinstance(parsed, dict):
                        type_name = type(parsed).__name__
                        raise ValueError(
                            f"Expected JSON object in SSE data, got: {type_name}"
                        )
                    choices = parsed.get("choices")
                    if choices is None or not isinstance(choices, list):
                        raise ValueError(
                            "Malformed vLLM response chunk missing 'choices' list: "
                            f"{data_content}"
                        )
                    if choices:
                        first_choice = choices[0]
                        if isinstance(first_choice, dict):
                            text_chunk = first_choice.get("text", "")
                            if text_chunk:
                                yield text_chunk
    finally:
        if close_client:
            await client.aclose()
