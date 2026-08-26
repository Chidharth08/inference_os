"""End-to-end integration tests for benchmark execution engine."""

import asyncio
from pathlib import Path

import httpx

from inference_os import BenchmarkConfig
from inference_os.results import load_benchmark_run
from inference_os.runner.engine import execute_benchmark


class MockWordTokenizer:
    """Deterministic whitespace mock tokenizer for testing."""

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


def test_execute_benchmark_end_to_end(tmp_path: Path) -> None:
    """Verify complete execution flow: synthetic prompt, stream, persistence."""

    def mock_sse_handler(request: httpx.Request) -> httpx.Response:
        content = (
            'data: {"choices": [{"text": "Alpha"}]}\n\n'
            'data: {"choices": [{"text": " Beta"}]}\n\n'
            'data: {"choices": [{"text": " Gamma"}]}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, text=content)

    transport = httpx.MockTransport(mock_sse_handler)
    client = httpx.AsyncClient(transport=transport)
    tokenizer = MockWordTokenizer()

    config = BenchmarkConfig(
        model="test-model",
        prompt_tokens=16,
        max_output_tokens=10,
        num_requests=3,
        warmup_requests=1,
        seed=42,
        experiment_id="E000_TEST",
        output_dir=str(tmp_path),
    )

    async def _run() -> None:
        run_dir, result, gpu_summary = await execute_benchmark(
            config=config,
            tokenizer=tokenizer,
            client=client,
        )

        # Directory verification
        assert run_dir.is_dir()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "environment.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "requests.jsonl").exists()

        # Measurement verification
        assert len(result.warmup_measurements) == 1
        assert len(result.measured_requests) == 3
        assert result.summary.total_requests == 3
        assert result.summary.successful_requests == 3
        assert result.summary.failed_requests == 0
        assert result.summary.total_output_tokens == 9  # 3 chunks * 3 requests
        assert result.summary.ttft_stats is not None
        assert result.summary.e2e_latency_stats is not None

        # Load verification
        loaded = load_benchmark_run(run_dir)
        assert loaded["config"]["model"] == "test-model"
        assert len(loaded["requests"]) == 4  # 1 warmup + 3 measured
        assert loaded["requests"][0]["is_warmup"] is True
        assert loaded["requests"][1]["is_warmup"] is False

    asyncio.run(_run())
