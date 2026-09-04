# E001-A — Input Length Scaling (Prefill Characterization)

## Primary Question

> **Why does increasing prompt context length affect first-token latency (TTFT) and total latency differently from decode latency?**

---

## Inference Systems Concept

Every autoregressive inference request consists of two distinct phases:

1. **Prefill Phase (Compute-Bound / Parallel Matrix Multiplication)**:
   - When the user sends a prompt of length $P$, the engine processes all $P$ tokens in parallel across all transformer layers.
   - Computes query, key, value projections and computes self-attention across the full sequence.
   - For short-to-medium sequence lengths on modern GPUs with tensor cores (like the NVIDIA RTX 3090), GEMM matrix-matrix multiplications ($P \times D \times D$) dominate compute.
   - Time to First Token (TTFT) directly measures this prefill duration plus network/scheduling overhead.

2. **Decode Phase (Memory-Bandwidth-Bound / Sequential Generation)**:
   - Tokens are generated one by one.
   - Each token step only computes query vectors for the single new token and attends against the accumulated Key-Value (KV) cache of past tokens.
   - Because generation is sequential ($N_{\text{out}}$ steps), each step must load all model weights from VRAM to compute activations for a single token.
   - Time Per Output Token (TPOT) measures the average duration of each decode step.

3. **KV Cache Allocation**:
   - During prefill, the KV states for all $P$ tokens are computed and stored in GPU memory ($2 \times 2 \times \text{layers} \times d_{\text{kv}} \times P \times \text{precision\_bytes}$).
   - As prompt length grows, VRAM allocated to KV cache increases monotonically.

---

## Hypothesis

1. **TTFT Scaling**: TTFT will scale monotonically with prompt token length $P$. On modern GPU hardware with FlashAttention/FlashInfer, the scaling across the prompt token range will exhibit compute-bound scaling as parallel prefill matrix multiplications expand.
2. **TPOT Invariance**: TPOT will remain largely invariant across prompt lengths at batch size 1, because each single-token decode step is dominated by loading model weights from memory bandwidth rather than the KV cache sequence length.
3. **E2E Latency**: E2E latency will increase along with the growth in TTFT, while total decode time for a fixed output token count remains constant across prompt lengths.
4. **GPU Memory**: VRAM usage will grow monotonically with prompt length due to additional KV cache memory block allocations.

---

## Expected Behavior (Qualitative)

- **TTFT**: Monotonically increases with prompt token count $P$ as prefill compute and attention operations scale.
- **TPOT**: Remains flat and constant across all prompt lengths under sequential single-request execution.
- **E2E Latency**: Shifts upward by the increase in TTFT while decode duration remains steady.
- **VRAM**: Increases monotonically as longer prompt contexts require additional KV cache memory allocation.

---

## Experimental Setup & Serving Controls

| Variable / Control | Value | Rationale |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` | Standard 7B instruction model |
| **Backend** | vLLM (OpenAI-compatible HTTP endpoint) | High-performance PagedAttention engine |
| **Precision** | BF16 (16-bit brain floating point) | Default non-quantized baseline |
| **Hardware** | 1× NVIDIA RTX 3090 (24 GB VRAM) | Fixed GPU target |
| **Prefix Caching** | **DISABLED** (`--no-enable-prefix-caching`) | Prevents KV cache hit shortcuts across repeated runs |
| **Chunked Prefill** | **DISABLED** (`--no-enable-chunked-prefill`) | Ensures pure, un-chunked full-prompt prefill measurement |
| **Concurrency** | `1` (sequential execution) | Eliminates queuing and batch scheduling interference |
| **Output Tokens ($N_{\text{out}}$)**| `128` (fixed) | Constant decode duration |
| **Input Tokens ($P$)**| `[128, 512, 2048, 4096]` | 1D parameter sweep variable |
| **Warmup Requests** | `2` | Discarded to isolate cold memory allocations |
| **Measured Requests** | `10` | Repeated runs to measure statistical variance |
| **Sampling Temperature** | `0.0` (greedy decoding) | Deterministic output lengths and generation paths |
| **Seed** | `42` | Deterministic synthetic prompt generation |

---

## Controlled vs Independent Variables

- **Independent Variable**: Prompt Token Count $P \in \{128, 512, 2048, 4096\}$
- **Controlled Variables**:
  - Model architecture & weights (`Qwen/Qwen2.5-7B-Instruct`, BF16)
  - Output token count ($128$)
  - Request concurrency ($1$)
  - GPU clock, driver, CUDA runtime, and host OS
  - Sampling parameters (greedy, $T=0.0$)
  - Disabled prefix caching (`--no-enable-prefix-caching`)
  - Disabled chunked prefill (`--no-enable-chunked-prefill`)
- **Dependent Variables (Metrics)**:
  - Time to First Token (TTFT, P50 & mean $\pm$ stddev)
  - End-to-End Latency (E2E, P50 & mean)
  - Time Per Output Token (TPOT, ms/token)
  - Peak and Average VRAM usage (MiB)
  - GPU Compute Utilization (%)

---

## Running the Benchmark

```bash
# 1. Start vLLM server with prefix caching and chunked prefill disabled:
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --port 18000 \
  --gpu-memory-utilization 0.90

# 2. Run the sweep using the CLI:
inference-os sweep --config configs/e001a_input_scaling.yaml

# Or running the standalone experiment script:
python experiments/E001-prefill-decode/E001A-input-length/run_e001a.py
```

Generated artifacts:
- `runs/E001A_<timestamp>/sweep_summary.json`
- `runs/E001A_<timestamp>/prompt_tokens_<N>/requests.jsonl`
- `runs/E001A_<timestamp>/prompt_tokens_<N>/telemetry.jsonl`
- `runs/E001A_<timestamp>/plots/ttft_vs_input_tokens.png`
- `runs/E001A_<timestamp>/plots/e2e_vs_input_tokens.png`
- `runs/E001A_<timestamp>/plots/tpot_vs_input_tokens.png`
- `runs/E001A_<timestamp>/plots/gpu_memory_vs_input_tokens.png`

---

## Limitations & Non-Proof Boundaries

- **Does NOT prove decode scaling**: Output length is held fixed at 128 tokens; decode scaling is isolated in E001-B.
- **Does NOT study chunked prefill**: Chunked prefill is disabled (`--no-enable-chunked-prefill`) to isolate un-chunked baseline prefill scaling.
- **Does NOT study multi-tenant concurrency**: Concurrency is fixed at 1. Concurrent scheduling dynamics are isolated in E002.
- **Hardware-Specific**: Results are specific to 1× NVIDIA RTX 3090 (Ampere architecture, 936 GB/s memory bandwidth) and should not be generalized to other GPU architectures without separate baseline characterization.
