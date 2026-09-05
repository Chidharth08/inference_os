# 📚 inference_os — Complete Study Guide (Through Milestone E000)

This guide covers **everything** we have built, **why** we built it, **how** it works, and **what the numbers mean** at a physical systems level.

---

## Table of Contents

1. [The Big Picture — What Are We Building and Why?](#1-the-big-picture)
2. [GPU Physics — How LLM Inference Actually Works on Hardware](#2-gpu-physics)
3. [Architecture Overview — The Full System Map](#3-architecture-overview)
4. [Layer 1: Workload Generation — Controlling the Input](#4-layer-1-workload-generation)
5. [Layer 2: The vLLM Backend — Talking to the GPU Server](#5-layer-2-the-vllm-backend)
6. [Layer 3: Request Measurement — Capturing Nanosecond Timings](#6-layer-3-request-measurement)
7. [Layer 4: The Benchmark Runner — Warm-up Separation & Sequential Execution](#7-layer-4-the-benchmark-runner)
8. [Layer 5: Statistical Summary — From Raw Data to Percentiles](#8-layer-5-statistical-summary)
9. [Layer 6: GPU Telemetry — Monitoring the Hardware in Real-Time](#9-layer-6-gpu-telemetry)
10. [Layer 7: Environment Capture — Reproducibility Metadata](#10-layer-7-environment-capture)
11. [Layer 8: Results Persistence — Saving Everything to Disk](#11-layer-8-results-persistence)
12. [Layer 9: The Execution Engine — Wiring It All Together](#12-layer-9-the-execution-engine)
13. [Layer 10: Configuration & CLI — User-Facing Interface](#13-layer-10-configuration-and-cli)
14. [The Complete Data Flow — End to End](#14-the-complete-data-flow)
15. [Test Suite — How We Verify Everything Locally](#15-test-suite)
16. [Live Benchmark Results — Reading the Numbers](#16-live-benchmark-results)
17. [Key Design Decisions — Why We Made the Choices We Made](#17-key-design-decisions)
18. [Glossary](#18-glossary)

---

## 1. The Big Picture

### What is `inference_os`?

`inference_os` is a **reproducible framework for designing, running, measuring, and analyzing LLM inference experiments**. Think of it as a scientific instrument for studying how Large Language Models behave when running on real GPU hardware.

### Why does this exist?

When you ask an LLM a question, you send a request to a **serving engine** (like vLLM) running on a GPU. The engine generates tokens one by one and streams them back. But:

- **How fast** does the first token arrive? (TTFT)
- **How fast** does it generate all the tokens? (Throughput)
- **How much GPU memory** does it use? (VRAM)
- **How consistent** are these numbers across repeated runs? (Variance)
- **Are the first few requests slower** than later ones? (Cold vs. warm)

These questions are **hard to answer correctly**. Most benchmarking tools give you a single number, but that number can be misleading if you don't control for warm-up effects, don't verify exact token counts, or don't capture the hardware state.

### The Core Systems Question (E000)

> **"How do you know that an LLM inference benchmark is measuring what you think it is measuring?"**

Milestone E000 answers this by building the measurement instrument itself, validating it on live hardware, and proving that the results are trustworthy, reproducible, and physically meaningful.

---

## 2. GPU Physics

This section explains **what physically happens on the GPU** when you send a prompt to a model. Understanding this is critical to interpreting every metric we collect.

### 2.1 The Two Phases of LLM Inference

Every single request to an LLM goes through two distinct computational phases:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR PROMPT (128 tokens)                     │
│                              │                                       │
│                    ┌─────────▼──────────┐                            │
│                    │    PREFILL PHASE    │   ◄── Processes ALL 128    │
│                    │                    │       tokens in PARALLEL    │
│                    │  Matrix multiplies │       (compute-bound)      │
│                    │  across all tokens │                            │
│                    └─────────┬──────────┘                            │
│                              │                                       │
│                        Token #1 arrives ──► This is TTFT             │
│                              │                                       │
│                    ┌─────────▼──────────┐                            │
│                    │    DECODE PHASE     │   ◄── Generates tokens     │
│                    │                    │       ONE BY ONE            │
│                    │  Sequential token  │       (memory-bandwidth    │
│                    │  generation        │        bound)              │
│                    └─────────┬──────────┘                            │
│                              │                                       │
│                     Token #64 arrives ──► This is E2E Latency        │
└─────────────────────────────────────────────────────────────────────┘
```

#### Prefill Phase (Compute-Bound)

- The GPU receives your entire 128-token prompt at once.
- It runs **matrix multiplications** across all 128 tokens simultaneously. Each of the model's 28 transformer layers performs attention and feed-forward operations on every token.
- This is **compute-bound**: the bottleneck is the raw arithmetic capacity of the GPU's CUDA cores (called Streaming Multiprocessors or SMs).
- On our RTX 3090, this takes **~40 ms** for 128 tokens.
- When prefill finishes, the first output token is ready. The time from "request sent" to "first token arrives" is called **Time to First Token (TTFT)**.

#### Decode Phase (Memory-Bandwidth-Bound)

- After the first token, the model generates tokens **one at a time**, sequentially.
- For each token, it must read the entire model weights (~14.2 GB for a 7B parameter model at BF16) and the KV cache from VRAM.
- This is **memory-bandwidth-bound**: the bottleneck is how fast data can be read from GPU VRAM (936 GB/s on RTX 3090).
- Each token takes **~19.45 ms** to generate (this is called **Time Per Output Token** or TPOT).
- For 64 output tokens: `64 × 19.45 ms ≈ 1,245 ms` plus the 40 ms prefill = ~1,285 ms total.

### 2.2 VRAM (Video RAM) — What Lives in GPU Memory?

The RTX 3090 has **24,576 MiB (24 GB)** of VRAM. During our benchmark, the GPU used **18,914 MiB (18.5 GB)**. Here's what occupied that space:

| Component | Size | What It Is |
|---|---|---|
| **Model Weights** | ~14.2 GB | The 7 billion parameters stored in BF16 format (2 bytes per parameter: 7B × 2 = 14 GB) |
| **KV Cache** | ~3.6 GB | Memory reserved for storing attention keys and values during generation (pre-allocated by vLLM) |
| **Activations & Overhead** | ~0.7 GB | Temporary intermediate computation results, CUDA graph memory, kernel workspace |
| **Total** | **~18.5 GB** | Matches our measured **18,914 MiB** |

### 2.3 GPU Compute Utilization — What Does 94.8% Mean?

The GPU has **10,496 CUDA cores** organized into **82 Streaming Multiprocessors (SMs)**. When we report:

- **94.8% average compute utilization**: This means that during the benchmark, the SMs were executing arithmetic operations 94.8% of the time. The remaining 5.2% is idle cycles waiting for data to arrive from VRAM.
- **100% peak**: During the prefill phase (highly parallel matrix multiplications), compute utilization maxes out.

### 2.4 BF16 (bfloat16) — Why Not Full Precision?

Each model parameter is stored as a **bfloat16** floating-point number (16 bits = 2 bytes) instead of float32 (32 bits = 4 bytes). This:
- **Halves VRAM usage**: 14 GB instead of 28 GB (wouldn't fit in 24 GB VRAM otherwise).
- **Doubles throughput**: The Tensor Cores on the RTX 3090 can process bfloat16 arithmetic at 2× the rate of float32.
- **Negligible quality loss**: BF16 preserves the same exponent range as float32 (just less mantissa precision), so model quality is virtually identical.

### 2.5 Cold Start vs. Warm — Why Warm-up Matters

The **first request** to vLLM after startup is dramatically slower than subsequent ones:

| Request | TTFT | Why |
|---|---|---|
| **Cold (First request)** | ~218 ms | CUDA graph compilation, kernel JIT compilation, KV cache memory page allocation |
| **Warm (After warm-up)** | ~40 ms | Compiled graphs and allocated memory pages are reused |

This is why our benchmark runs **W=2 warm-up requests** before starting the measured N=10 requests. Without this, the cold start would skew all our latency statistics.

---

## 3. Architecture Overview

### 3.1 Directory Structure

```
inference_os/
├── pyproject.toml                           # Project config, dependencies, CLI entry point
├── src/inference_os/
│   ├── __init__.py                          # Package root, exports BenchmarkConfig
│   ├── config.py                            # BenchmarkConfig dataclass + validation
│   ├── cli.py                               # CLI entry point (inference-os run)
│   ├── backends/
│   │   └── vllm.py                          # SSE streaming adapter for vLLM HTTP API
│   ├── workloads/
│   │   ├── base.py                          # Tokenizer Protocol (interface contract)
│   │   ├── hf_tokenizer.py                  # HuggingFace AutoTokenizer adapter
│   │   └── synthetic.py                     # Deterministic prompt generator
│   ├── runner/
│   │   ├── request.py                       # Single-request timing harness
│   │   ├── benchmark.py                     # Sequential benchmark runner (W+N loop)
│   │   └── engine.py                        # Top-level orchestrator (wires everything)
│   ├── metrics/
│   │   ├── request.py                       # RequestMeasurement dataclass
│   │   └── summary.py                       # MetricStats, BenchmarkSummary, percentiles
│   ├── telemetry/
│   │   ├── environment.py                   # Git, OS, GPU, package version capture
│   │   └── gpu.py                           # GPUSample, GPUTelemetrySampler (async)
│   └── results/
│       └── persistence.py                   # save/load benchmark run directories
├── experiments/
│   └── E000-measurement-validation/
│       ├── README.md                        # Experiment documentation + results
│       └── run_e000.py                      # Executable benchmark script
├── tests/                                   # 58 unit + integration tests
└── outputs/
    ├── gpu_smoke_test.md                    # Initial cold smoke test logs
    └── e000_measurement_validation.md       # Full E000 live results + analysis
```

### 3.2 Data Flow Diagram

```
┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐
│ BenchmarkConfig│──►│  HFTokenizer     │──►│ generate_synthetic │
│ (config.py)   │    │ (hf_tokenizer.py)│    │ _prompt()          │
└──────┬───────┘    └──────────────────┘    │ (synthetic.py)     │
       │                                     └──────────┬────────┘
       │                                                │
       │            Prompt (exactly 128 tokens)         │
       │◄───────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│              execute_benchmark() [engine.py]                  │
│                                                              │
│  ┌──────────────────┐    ┌───────────────────────────────┐   │
│  │ GPUTelemetrySampler│    │ run_sequential_benchmark()    │   │
│  │ (gpu.py)          │    │ (benchmark.py)                │   │
│  │                   │    │                               │   │
│  │  ┌──────────────┐│    │  ┌─ Warmup Phase (W=2) ─────┐│   │
│  │  │ nvidia-smi   ││    │  │ for i in 0..1:            ││   │
│  │  │ every 100ms  ││    │  │   request_factory()       ││   │
│  │  │              ││    │  │   ──► vllm_stream()       ││   │
│  │  │ GPUSample[]  ││    │  │   ──► run_single_request()││   │
│  │  └──────────────┘│    │  │   ──► warmup_measurements ││   │
│  │                   │    │  └──────────────────────────┘│   │
│  │  Runs in parallel │    │                               │   │
│  │  as asyncio Task  │    │  ┌─ Measured Phase (N=10) ──┐│   │
│  └──────────────────┘    │  │ for i in 0..9:            ││   │
│                           │  │   request_factory()       ││   │
│                           │  │   ──► vllm_stream()       ││   │
│                           │  │   ──► run_single_request()││   │
│                           │  │   ──► measured_requests   ││   │
│                           │  └──────────────────────────┘│   │
│                           └───────────────────────────────┘   │
│                                                              │
│  ┌──────────────────┐    ┌───────────────────────────────┐   │
│  │ capture_environment│    │ save_benchmark_run()          │   │
│  │ (environment.py)  │    │ (persistence.py)              │   │
│  │                   │    │                               │   │
│  │ Git SHA, OS,     │    │ runs/<run_id>/                │   │
│  │ Python, GPU,     │──►│   config.json                 │   │
│  │ packages         │    │   environment.json            │   │
│  │                   │    │   summary.json               │   │
│  └──────────────────┘    │   requests.jsonl             │   │
│                           │   telemetry.jsonl            │   │
│                           └───────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. Layer 1: Workload Generation

### WHY: Controlling the Input

If you send a real user prompt like "Explain quantum computing", you have **no control** over:
- How many tokens the prompt actually is (could be 3 or 300)
- How the tokenizer splits it (varies by model)
- Whether repeated runs produce the same input

For a **scientific measurement**, you need to control the independent variable precisely. That means generating a prompt that is **exactly** 128 tokens — verified at the tokenizer level — every single time.

### WHAT: Three Components

#### 4.1 Tokenizer Protocol — [`workloads/base.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/workloads/base.py)

```python
@runtime_checkable
class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def count_tokens(self, text: str) -> int: ...
```

**Why a Protocol?** This is a Python structural typing contract. Any class that has `encode`, `decode`, and `count_tokens` methods automatically satisfies this protocol — no inheritance needed. This lets us:
- Use the real `HFTokenizer` in production (hits HuggingFace's library)
- Use a simple fake tokenizer in tests (splits on spaces) without importing the heavy `transformers` library

**Key concept**: `@runtime_checkable` means you can do `isinstance(obj, Tokenizer)` at runtime to verify an object satisfies the contract.

#### 4.2 HFTokenizer — [`workloads/hf_tokenizer.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/workloads/hf_tokenizer.py)

```python
class HFTokenizer:
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, ...) -> "HFTokenizer":
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, ...)
        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self._tokenizer.decode(token_ids, skip_special_tokens=True))
```

**Critical details**:
- `add_special_tokens=False`: We do NOT add `<BOS>` or `<EOS>` tokens during encoding. If we did, our 128-token prompt would secretly become 129 or 130 tokens, and our measurement would be off.
- `skip_special_tokens=True`: When decoding token IDs back to text, we strip any special tokens that might have leaked in.

#### 4.3 Synthetic Prompt Generator — [`workloads/synthetic.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/workloads/synthetic.py)

```python
def generate_synthetic_prompt(tokenizer: Tokenizer, num_tokens: int, seed: int = 42) -> str:
```

**The algorithm**:
1. Use a fixed `random.Random(seed)` so the same seed always produces the same prompt (determinism).
2. Pick random words from a controlled vocabulary: `"system architecture performance measurement latency throughput benchmark inference memory ..."`.
3. Encode the concatenated text and check the token count.
4. If too many tokens: slice the token ID list and decode back.
5. If decode→encode roundtrip changed the count (edge case with multi-byte characters): trim or pad individual tokens until `tokenizer.count_tokens(prompt) == num_tokens` exactly.

**Why is the roundtrip adjustment needed?** Tokenizers are not perfectly invertible. If you take 128 token IDs, decode to text, then re-encode, you might get 127 or 129 tokens because of whitespace normalization or sub-word boundary effects. The fine-tuning loop at the end guarantees exactness.

---

## 5. Layer 2: The vLLM Backend

### WHY: Talking to the Model

The model runs on a GPU server managed by **vLLM**, which exposes an **OpenAI-compatible HTTP API**. Our benchmark needs to send prompts and receive streaming token responses.

### WHAT: SSE Streaming Adapter — [`backends/vllm.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/backends/vllm.py)

```python
async def vllm_stream_completion(
    model: str, prompt: str, max_tokens: int,
    base_url: str, client: Optional[httpx.AsyncClient] = None,
) -> AsyncGenerator[str, None]:
```

### HOW: Server-Sent Events (SSE) Protocol

When you set `"stream": True` in the HTTP POST body, vLLM doesn't wait to generate all 64 tokens and then return them. Instead, it sends each token **as soon as it's generated** using the SSE protocol:

```
HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"choices": [{"text": "The"}]}

data: {"choices": [{"text": " concept"}]}

data: {"choices": [{"text": " of"}]}

...

data: [DONE]
```

Our adapter:
1. Opens an HTTP streaming connection using `httpx.AsyncClient.stream("POST", ...)`.
2. Iterates over lines as they arrive (`response.aiter_lines()`).
3. Skips empty lines and comment lines (starting with `:`).
4. Strips the `data: ` prefix from each line.
5. Stops when it sees `data: [DONE]`.
6. Parses the JSON, extracts `choices[0].text`, and `yield`s the text chunk.

**Why AsyncGenerator?** Using `async for chunk in stream` lets the caller (the request runner) process each token as it arrives. This is essential for measuring TTFT — we need to know the exact nanosecond the first token arrives.

**Why httpx?** `httpx` is a modern async HTTP client that supports streaming, connection pooling, and has a `MockTransport` mechanism that lets us inject fake responses in unit tests without needing a real server.

---

## 6. Layer 3: Request Measurement

### WHY: Capturing Nanosecond-Precision Timings

Every benchmark measurement needs three precise timestamps:
1. **Start time**: When the request is sent
2. **First token time**: When the first generated token arrives (for TTFT)
3. **Completion time**: When the last token arrives (for E2E latency)

### WHAT: Two Data Structures

#### 6.1 RequestMeasurement — [`metrics/request.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/metrics/request.py)

```python
@dataclass(frozen=True, slots=True)
class RequestMeasurement:
    request_id: str
    start_time_ns: int  # Nanosecond monotonic timestamp
    completion_time_ns: int  # Nanosecond monotonic timestamp
    input_tokens: int
    output_tokens: int
    success: bool
    first_token_time_ns: Optional[int] = None
    error_message: Optional[str] = None
```

**Design decisions**:
- **`frozen=True`**: Once a measurement is recorded, it can never be modified. This guarantees data integrity — you cannot accidentally alter a recorded observation.
- **`slots=True`**: Uses Python `__slots__` for memory efficiency. With thousands of measurements, this saves significant memory.
- **Nanosecond integers** (`int`), not floating-point seconds: Floating-point arithmetic introduces rounding errors. By storing raw nanoseconds as 64-bit integers, we preserve perfect precision and convert to seconds only at display time.
- **`__post_init__` validation**: Enforces physical invariants — completion cannot precede start, first token cannot precede start or follow completion, counts cannot be negative.

**Derived metrics** (computed properties, not stored):
```python
@property
def ttft_seconds(self) -> Optional[float]:
    return (self.first_token_time_ns - self.start_time_ns) / 1_000_000_000.0


@property
def e2e_latency_seconds(self) -> float:
    return (self.completion_time_ns - self.start_time_ns) / 1_000_000_000.0
```

#### 6.2 Single Request Runner — [`runner/request.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/runner/request.py)

```python
async def run_single_request(
    request_id: str,
    stream: AsyncIterable[T],
    input_tokens: int,
    output_tokens: Optional[int] = None,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> RequestMeasurement:
```

**The timing logic**:
```python
start_time_ns = clock_fn()  # ← Record start
first_token_time_ns = None
observed_chunks = 0

async for _ in stream:  # ← Consume each token
    if first_token_time_ns is None:
        first_token_time_ns = clock_fn()  # ← First token!
    observed_chunks += 1

completion_time_ns = clock_fn()  # ← All tokens received
```

**Why `clock_fn` as a parameter?** This is **dependency injection** for the clock. In production, we use `time.perf_counter_ns` (the highest-resolution monotonic clock available). In tests, we inject a fake clock that returns predetermined values, so we can verify exact timing calculations without sleeping or dealing with real-time variability.

---

## 7. Layer 4: The Benchmark Runner

### WHY: Warm-up Separation & Sequential Execution

A single request measurement is noisy. We need to run **many requests** and compute statistics. But the first few requests after server startup are artificially slow (CUDA compilation, memory allocation). We need to:
1. Run W warm-up requests and **exclude them from reported metrics**
2. Run N measured requests and compute statistics only from those
3. Still **preserve warm-up data** so we can compare cold vs. warm behavior

### WHAT: Sequential Benchmark Runner — [`runner/benchmark.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/runner/benchmark.py)

```python
async def run_sequential_benchmark(
    request_factory: RequestFactory,
    num_requests: int,
    warmup_requests: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> BenchmarkResult:
```

**The execution flow**:

```
Phase 1: WARMUP (W=2 requests, not timed for summary)
├── warmup-1: send prompt → consume stream → record RequestMeasurement
└── warmup-2: send prompt → consume stream → record RequestMeasurement
    └── warmup_measurements = [warmup-1, warmup-2]
    └── warmup_summary = compute_benchmark_summary(warmup_measurements)

Phase 2: MEASURED (N=10 requests, timed for summary)
├── req-1:  send prompt → consume stream → record RequestMeasurement
├── req-2:  ...
├── ...
└── req-10: send prompt → consume stream → record RequestMeasurement
    └── measured_requests = [req-1, ..., req-10]
    └── summary = compute_benchmark_summary(measured_requests)
```

**Key design: RequestFactory**

```python
RequestFactory = Callable[
    [str, int, bool], Awaitable[tuple[AsyncIterable[Any], int, Optional[int]]]
]
```

This is a callable that takes `(request_id, index, is_warmup)` and returns `(stream, input_tokens, output_tokens)`. The engine creates this factory by closing over the prompt and vLLM connection details. This separation means the benchmark runner has **zero knowledge** of vLLM, HTTP, or tokenizers — it just consumes streams and records timings. This makes it testable with completely fake streams.

**BenchmarkResult** preserves everything:
```python
@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    warmup_measurements: Sequence[RequestMeasurement]  # Cold data (kept!)
    measured_requests: Sequence[RequestMeasurement]  # Reported data
    summary: BenchmarkSummary  # Stats from measured only
    warmup_summary: Optional[BenchmarkSummary]  # Stats from warmup (for comparison)
```

---

## 8. Layer 5: Statistical Summary

### WHY: From Raw Data to Percentiles

10 raw `RequestMeasurement` objects are not actionable. We need to compute:
- **Mean, Min, Max, StdDev**: Basic distribution statistics
- **Percentiles (P50, P90, P95, P99)**: Tail latency characteristics
- **Throughput**: Requests per second and tokens per second

### WHAT: Summary Metrics — [`metrics/summary.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/metrics/summary.py)

#### Percentile Calculation (Linear Interpolation)

```python
def _calculate_percentile(sorted_values, percentile):
    n = len(sorted_values)
    rank = (percentile / 100.0) * (n - 1)  # Continuous rank
    lower_idx = math.floor(rank)  # Integer part
    upper_idx = math.ceil(rank)  # Next integer
    weight = rank - lower_idx  # Fractional part

    return (1.0 - weight) * sorted_values[lower_idx] + weight * sorted_values[upper_idx]
```

**Example**: P90 of 10 sorted values `[1278, 1279, 1280, 1281, 1282, 1283, 1284, 1290, 1300, 1329]`:
- `rank = 0.90 × 9 = 8.1`
- `lower_idx = 8` (value `1300`), `upper_idx = 9` (value `1329`)
- `weight = 0.1`
- `P90 = 0.9 × 1300 + 0.1 × 1329 = 1170 + 132.9 = 1302.9 ms`

This matches NumPy's `np.percentile(values, 90, method='linear')`. We implement it ourselves so we have **zero dependency on NumPy** for the core measurement path.

#### Standard Deviation (Bessel's Correction)

```python
variance = sum((x - mean) ** 2 for x in values) / (n - 1)  # n-1 not n!
std_dev = math.sqrt(variance)
```

We divide by `n-1` (not `n`) because we're computing a **sample standard deviation** from a subset of all possible requests, not the population. This is called **Bessel's correction** and gives an unbiased estimate.

#### Throughput Calculations

```python
request_throughput = successful_count / total_duration_seconds  # req/s
output_token_throughput = total_output_tokens / total_duration_seconds  # tok/s
```

**Important**: Duration is measured as wall-clock time spanning all N measured requests, not per-request. This captures the true sustained throughput including any gaps between requests.

---

## 9. Layer 6: GPU Telemetry

### WHY: Monitoring the Hardware

Latency numbers alone don't tell the full story. You need to know:
- **Is the GPU actually being used?** (Maybe requests are queuing on CPU)
- **How much VRAM is consumed?** (To know how close you are to OOM)
- **Does utilization drop during decode?** (Expected — decode is memory-bound)

### WHAT: GPU Telemetry Sampler — [`telemetry/gpu.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/telemetry/gpu.py)

#### GPUSample — A Single Snapshot

```python
@dataclass(frozen=True, slots=True)
class GPUSample:
    timestamp_ns: int
    memory_used_mb: int  # VRAM currently allocated
    memory_total_mb: int  # Total VRAM on device
    utilization_gpu_pct: int  # SM compute utilization (0-100)
    utilization_memory_pct: Optional[int] = None  # Memory bus utilization
```

Each sample is collected by running `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits` and parsing the CSV output.

#### GPUTelemetrySampler — Async Background Poller

```python
class GPUTelemetrySampler:
    async def _sample_loop(self):
        while self._running:
            sample = self._query_fn(self.device_index)
            if sample is not None:
                self._samples.append(sample)
            await asyncio.sleep(self.interval_seconds)  # 100ms default
```

**How it runs in parallel with the benchmark**: The sampler uses Python's `asyncio.create_task()` to run in the background. While the main benchmark is `await`ing HTTP responses from vLLM, the event loop schedules the telemetry sampler's sleep timer and nvidia-smi calls in between. This is **cooperative multitasking** — no threads, no race conditions.

**Context manager pattern** (`async with sampler:`) ensures the background task is always properly started and stopped, even if the benchmark crashes.

---

## 10. Layer 7: Environment Capture

### WHY: Reproducibility

If you can't reproduce a benchmark result, it's not science. Six months from now, you need to know:
- What exact code version was running?
- What GPU was used?
- What driver and CUDA version?
- What Python and library versions?

### WHAT: Environment Metadata — [`telemetry/environment.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/telemetry/environment.py)

```python
@dataclass(frozen=True, slots=True)
class EnvironmentMetadata:
    timestamp_utc: str  # When the snapshot was taken
    hostname: str  # Machine name
    os_name: str  # "Linux" / "Windows"
    os_release: str  # Kernel version
    python_version: str  # "3.11.15"
    git: GitMetadata  # Commit SHA, branch, dirty status
    gpu: Optional[GPUMetadata]  # GPU name, driver, CUDA, VRAM
    packages: Optional[dict[str, str]]  # Library versions
```

**Fail-safe design**: Every subprocess call (`git`, `nvidia-smi`) first checks `shutil.which()` to verify the binary exists, and wraps the call in try/except. On a developer laptop without a GPU, `capture_environment()` still works — it just returns `gpu=None`. The benchmark never crashes due to missing system tools.

---

## 11. Layer 8: Results Persistence

### WHY: Saving Everything

Every run produces a self-contained directory that can be analyzed, compared, or shared independently.

### WHAT: Disk Serialization — [`results/persistence.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/results/persistence.py)

```
runs/E000_20260826_095535_e6683e48/
├── config.json          ← Input parameters (model, tokens, warmup count, etc.)
├── environment.json     ← Git SHA, OS, GPU, Python, package versions
├── summary.json         ← Aggregate statistics (throughput, percentiles, GPU summary)
├── requests.jsonl       ← One JSON line per request (warmup + measured, tagged)
└── telemetry.jsonl      ← One JSON line per GPU sample (every 100ms)
```

**Why JSONL for requests and telemetry?** JSON Lines (one JSON object per line) is:
- **Streamable**: You can read it line-by-line without loading the entire file
- **Appendable**: You can add records without rewriting the file
- **grep-friendly**: `grep "is_warmup.*true" requests.jsonl` instantly filters warmup records

**Run ID format**: `{experiment_id}_{YYYYMMDD}_{HHMMSS}_{8-char-uuid}` — the combination of timestamp and UUID ensures uniqueness even if two runs start in the same second.

---

## 12. Layer 9: The Execution Engine

### WHY: Wiring Everything Together

All the layers above are independent, testable components. The engine connects them into a single executable pipeline.

### WHAT: Execute Benchmark — [`runner/engine.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/runner/engine.py)

```python
async def execute_benchmark(config, tokenizer=None, client=None):
    # 1. Load tokenizer (or accept injected mock)
    if tokenizer is None:
        tokenizer = HFTokenizer.from_pretrained(config.model)

    # 2. Generate exactly config.prompt_tokens synthetic tokens
    prompt = generate_synthetic_prompt(tokenizer, config.prompt_tokens, config.seed)

    # 3. Create the request factory closure
    async def request_factory(request_id, index, is_warmup):
        stream = vllm_stream_completion(
            config.model, prompt, config.max_output_tokens, ...
        )
        return stream, actual_input_tokens, None

    # 4. Run benchmark with background GPU sampling
    sampler = GPUTelemetrySampler(
        config.telemetry_interval_seconds, config.device_index
    )
    async with sampler:
        result = await run_sequential_benchmark(
            request_factory, config.num_requests, config.warmup_requests
        )

    # 5. Capture environment, save to disk
    environment = capture_environment()
    run_dir = save_benchmark_run(
        config, environment, result, sampler.get_summary(), sampler.get_samples()
    )

    return run_dir, result, gpu_summary
```

**Why accept optional `tokenizer` and `client`?** Dependency injection. In production, they are created automatically. In tests, you inject a mock tokenizer (no HuggingFace download) and a mock HTTP client (no real server needed). This lets us run the full end-to-end pipeline in `pytest` in under 1 second.

---

## 13. Layer 10: Configuration & CLI

### WHAT: BenchmarkConfig — [`config.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/config.py)

```python
@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    model: str  # Required, no default
    base_url: str = "http://localhost:8000"
    prompt_tokens: int = 128
    max_output_tokens: int = 64
    num_requests: int = 10
    warmup_requests: int = 2
    seed: int = 42
    temperature: float = 0.0
    telemetry_interval_seconds: float = 0.1
    device_index: int = 0
    experiment_id: str = "E000"
    output_dir: str = "runs"
```

**`frozen=True`** means once a config is created, it cannot be modified. This prevents accidental mutation during a benchmark run.

**`__post_init__` validation** catches invalid configs immediately:
- `prompt_tokens <= 0` → `ValueError`
- `warmup_requests < 0` → `ValueError` (but `warmup_requests == 0` is valid — for cold-only measurement)
- `model` empty string → `ValueError`

**Serialization**: `to_json()` and `from_json()` enable config to be saved to disk and loaded back, ensuring every run's configuration is fully recoverable.

### WHAT: CLI & Experiment Script

Two entry points exist:

1. **`inference-os run`** ([`cli.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/cli.py)): Generic CLI subcommand registered via `pyproject.toml`'s `[project.scripts]`.

2. **`experiments/E000-measurement-validation/run_e000.py`** ([`run_e000.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/experiments/E000-measurement-validation/run_e000.py)): Standalone script with experiment-specific defaults and a formatted console report showing TTFT/E2E percentiles, throughputs, and GPU telemetry.

---

## 14. The Complete Data Flow

Here is the exact sequence of operations when you run `python run_e000.py --base-url http://127.0.0.1:18000`:

```
1.  Parse CLI args → BenchmarkConfig(model="Qwen/Qwen2.5-7B-Instruct", prompt_tokens=128, ...)

2.  HFTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    → Downloads tokenizer vocab/merges from HuggingFace Hub
    → Creates tokenizer with 151,665 vocabulary entries

3.  generate_synthetic_prompt(tokenizer, num_tokens=128, seed=42)
    → Random words from controlled vocabulary
    → Encode → Slice to 128 tokens → Decode → Re-encode → Verify == 128
    → Returns: "system architecture performance measurement latency ..."

4.  GPUTelemetrySampler.start()
    → asyncio.create_task(_sample_loop)
    → Every 100ms: subprocess nvidia-smi → GPUSample(memory=18914, util=94%, ...)

5.  WARMUP Phase (2 requests):
    5a. POST http://127.0.0.1:18000/v1/completions {model, prompt, max_tokens=64, stream=true}
        → vLLM prefills 128 tokens (GPU compute-bound, ~218ms COLD)
        → vLLM decodes 64 tokens (GPU memory-bound, ~19ms/token)
        → run_single_request records: start_ns, first_token_ns, completion_ns
        → RequestMeasurement(request_id="warmup-1", ttft=0.218s, e2e=1.45s)

    5b. Same for warmup-2 (now warm: ttft ≈ 40ms, e2e ≈ 1.28s)

6.  MEASURED Phase (10 requests):
    6a-6j. Same as above but request_ids are "req-1" through "req-10"
        → All warm: ttft ≈ 35-74ms, e2e ≈ 1278-1329ms

7.  GPUTelemetrySampler.stop()
    → Cancel background task
    → compute_gpu_summary(samples) → peak VRAM 18914 MiB, avg util 94.8%

8.  capture_environment()
    → git rev-parse HEAD → "7d4229a..."
    → nvidia-smi --query-gpu → RTX 3090, 24576 MiB
    → platform.python_version() → "3.11.15"
    → importlib.metadata.version("transformers") → "5.15.1"

9.  save_benchmark_run(config, environment, result, gpu_summary, gpu_samples)
    → mkdir runs/E000_20260826_095535_e6683e48/
    → Write config.json, environment.json, summary.json, requests.jsonl, telemetry.jsonl

10. Print formatted console report
```

---

## 15. Test Suite

### 58 Tests Across 13 Test Files

All tests run locally **without a GPU, without a server, without internet**:

| Test File | Count | What It Tests |
|---|---|---|
| `test_import.py` | 1 | Package imports correctly |
| `test_config.py` | 3 | Config defaults, validation errors, JSON roundtrip |
| `test_workloads.py` | 11 | Token counting, synthetic prompt exact length, seed determinism |
| `test_request_metrics.py` | 7 | RequestMeasurement invariants, TTFT/E2E calculation, nanosecond precision |
| `test_request_runner.py` | 5 | Timing capture with fake clocks, error handling, stream consumption |
| `test_vllm_backend.py` | 6 | SSE parsing, mock HTTP transport, error responses |
| `test_benchmark_runner.py` | 3 | Warmup exclusion, sequential execution, concurrency=1 |
| `test_summary_metrics.py` | 5 | Percentile math, standard deviation, throughput calculation |
| `test_environment.py` | 7 | Git parsing, nvidia-smi parsing, fail-safe on missing tools |
| `test_gpu_telemetry.py` | 6 | Sample parsing, summary aggregation, async sampler lifecycle |
| `test_results_persistence.py` | 2 | Save/load roundtrip, missing directory handling |
| `test_benchmark_engine.py` | 1 | Full end-to-end pipeline with all mocks |
| `test_cli.py` | 1 | CLI argument parsing and help text |

**Key testing pattern**: Mock everything external:
- **Fake clock**: Returns predetermined nanosecond values → exact timing assertions
- **Fake tokenizer**: Splits on spaces → no HuggingFace download needed
- **Mock HTTP transport**: `httpx.MockTransport` returns canned SSE responses → no server needed
- **Mock nvidia-smi**: Inject a `query_fn` that returns fake `GPUSample` objects → no GPU needed

---

## 16. Live Benchmark Results

### Raw Numbers from RTX 3090

| Metric | Value | Physical Meaning |
|---|---|---|
| **TTFT P50** | 39.63 ms | Prefill latency for 128 tokens through 28 transformer layers |
| **TTFT P95** | 64.52 ms | Worst-case prefill (may include OS scheduling jitter) |
| **TTFT Min** | 35.98 ms | Best-case prefill (all caches warm, no OS interference) |
| **E2E P50** | 1,282.15 ms | Total time for 128 input + 64 output tokens |
| **E2E StdDev** | 15.78 ms | 1.2% coefficient of variation — excellent reproducibility |
| **Token Throughput** | 49.65 tok/s | Sustained single-stream decode speed |
| **TPOT** | 19.45 ms/tok | Time Per Output Token = (E2E - TTFT) / 64 |
| **Peak VRAM** | 18,914 MiB | Weights (14.2 GB) + KV cache (3.6 GB) + overhead |
| **Avg GPU Util** | 94.8% | SM compute utilization averaged across benchmark |

### What the E2E Latency Tells Us

`E2E = TTFT + (output_tokens × TPOT)`
`1,282 ms ≈ 40 ms + (64 × 19.45 ms)` ✓

The decode phase dominates (97% of total time). This is expected for a 7B model — weights are large enough that reading them from VRAM is the bottleneck, not the arithmetic.

### Cold vs. Warm Comparison

| | TTFT | E2E |
|---|---|---|
| **Cold (smoke test, no warmup)** | 218.72 ms | 1,091.29 ms |
| **Warm (E000, after W=2 warmup)** | 39.63 ms (P50) | 1,282.15 ms (P50) |

The cold TTFT was **5.5× slower** due to one-time CUDA graph compilation and JIT kernel compilation. After warmup, TTFT drops to its true physical value (~40 ms). Note: E2E was actually higher for the warm run because the cold smoke test only generated ~43 tokens (it hit a stop condition) while E000 generated the full 64 tokens.

---

## 17. Key Design Decisions

### Why `dataclass(frozen=True, slots=True)` everywhere?

- **`frozen=True`**: Immutability. Measurements, configs, and summaries cannot be accidentally modified after creation. This is critical for data integrity in a benchmarking context.
- **`slots=True`**: Memory efficiency. Python normally stores instance attributes in a `__dict__` dictionary. `slots` uses a fixed-size struct instead, saving ~64 bytes per instance and improving attribute access speed.

### Why nanosecond integers instead of floating-point seconds?

Floating-point arithmetic accumulates rounding errors. `0.1 + 0.2 != 0.3` in IEEE 754. By storing timestamps as 64-bit integer nanoseconds from `time.perf_counter_ns`, we get:
- **Perfect precision**: No rounding until final display
- **Exact subtraction**: `completion_ns - start_ns` is exact integer arithmetic
- **Cross-platform consistency**: `perf_counter_ns` is monotonic (never goes backwards, unlike wall-clock time)

### Why concurrency = 1?

E000 is about **measurement validation**, not throughput optimization. Running requests sequentially eliminates all interference between requests (no queue contention, no batch scheduling, no VRAM pressure from concurrent KV caches). This gives us a clean baseline to validate our timing harness against.

### Why a Tokenizer Protocol instead of inheriting from a base class?

Python Protocols (structural typing) decouple the interface from the implementation:
- The benchmark runner only cares that something has `encode()`, `decode()`, `count_tokens()`.
- `HFTokenizer` satisfies this without inheriting from anything.
- A test fake can satisfy it with a 3-line class.
- If we later add a `TikTokenTokenizer` for OpenAI models, it just needs the same three methods.

### Why `from __future__ import annotations` in persistence.py?

This avoids a **circular import**:
- `persistence.py` needs to reference `BenchmarkResult` (defined in `runner/benchmark.py`)
- `runner/__init__.py` imports from `engine.py`, which imports from `persistence.py`
- `from __future__ import annotations` makes all type hints strings that are evaluated lazily, breaking the import cycle. Combined with `TYPE_CHECKING`, the import only happens during static analysis (mypy/pyright), not at runtime.

---

## 18. Glossary

| Term | Definition |
|---|---|
| **TTFT** | Time to First Token — wall-clock time from sending the request to receiving the first generated token |
| **E2E Latency** | End-to-End Latency — wall-clock time from sending the request to receiving the last generated token |
| **TPOT** | Time Per Output Token — average time to generate each token during decode: `(E2E - TTFT) / output_tokens` |
| **Prefill** | Phase where the GPU processes all input tokens in parallel (compute-bound, high GPU utilization) |
| **Decode** | Phase where the GPU generates output tokens one at a time (memory-bandwidth-bound, lower GPU utilization) |
| **KV Cache** | Key-Value cache — stored attention states from each transformer layer, reused during decode to avoid recomputation |
| **VRAM** | Video RAM — the GPU's dedicated high-bandwidth memory (24 GB on RTX 3090) |
| **BF16** | bfloat16 — 16-bit floating-point format preserving float32's exponent range with reduced mantissa precision |
| **SM** | Streaming Multiprocessor — a compute unit on the GPU; RTX 3090 has 82 SMs with 128 CUDA cores each |
| **SSE** | Server-Sent Events — HTTP streaming protocol where the server pushes `data:` lines to the client |
| **CUDA Graph** | Pre-compiled GPU execution plan that eliminates kernel launch overhead on repeated runs |
| **P50/P90/P95/P99** | Percentile values — P90 means "90% of measurements were at or below this value" |
| **Bessel's Correction** | Dividing variance by `n-1` instead of `n` for unbiased sample standard deviation estimation |
| **Monotonic Clock** | A clock that only moves forward (never adjusted by NTP), essential for accurate duration measurement |
| **JSONL** | JSON Lines — file format with one JSON object per line, enabling streaming reads |
| **Dependency Injection** | Passing collaborators (clock, tokenizer, HTTP client) as parameters so they can be replaced in tests |
