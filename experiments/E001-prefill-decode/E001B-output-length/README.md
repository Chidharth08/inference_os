# E001-B — Output Length Scaling (Decode Characterization)

## Primary Question

> **Why does increasing generation length scale end-to-end latency linearly while keeping first-token latency (TTFT) and per-token decode time (TPOT) largely constant?**

---

## Inference Systems Concept

Every autoregressive inference request consists of two distinct phases:

1. **Prefill Phase (Fixed Context Baseline)**:
   - In E001-B, the input prompt token length is held fixed at $P = 128$.
   - The prefill phase parallel matrix multiplication occurs once per request, computing the initial KV cache entries for the 128 prompt tokens.
   - Because the prompt length is fixed and small, the prefill duration (Time to First Token, TTFT) remains constant and negligible compared to total generation time.

2. **Decode Phase (Memory-Bandwidth-Bound / Sequential Autoregressive Generation)**:
   - In autoregressive generation, output tokens are generated sequentially one at a time ($N_{\text{out}}$ iterations).
   - In each decode step, the model computes query projections for a single new token, attends across past KV states in the KV cache, and produces logits.
   - For batch size 1, arithmetic intensity is very low ($\sim 1 \text{ FLOP} / \text{Byte}$). The GPU spends almost all of its execution time reading the entire $\approx 15 \text{ GB}$ model weights from VRAM for each single token generated.
   - Time Per Output Token (TPOT) measures this per-token decode duration.
   - Total latency decomposes as:
     $$E2E \approx TTFT + (N_{\text{out}} - 1) \times TPOT$$
   - Consequently, $E2E$ scales strictly linearly with output length $N_{\text{out}}$, with slope equal to $TPOT$.

3. **KV Cache Growth during Generation**:
   - For every generated token, a new key and value vector is appended to the KV cache in GPU memory across all layers.
   - Total KV cache consumption per request grows proportionally to $(P + N_{\text{out}})$.
   - Tracking GPU memory telemetry highlights KV cache memory expansion during extended generation.

---

## Hypothesis

1. **TTFT Invariance**: TTFT will remain invariant across output token lengths $N_{\text{out}} \in [32, 128, 512, 1024]$, since prompt token length $P = 128$ is constant.
2. **Linear E2E Latency**: E2E latency will scale linearly ($R^2 \approx 1.0$) with output length $N_{\text{out}}$, dominated by sequential decode steps.
3. **TPOT Invariance**: TPOT will remain constant across all output lengths at batch size 1 (dominated by memory bandwidth needed to read model weights per step).
4. **VRAM Scaling**: Peak and average GPU memory usage will increase monotonically as more KV cache blocks are allocated during generation.

---

## Expected Behavior (Qualitative)

- **TTFT**: Flat line across all values of $N_{\text{out}}$ (unaffected by output length).
- **TPOT**: Flat line across all values of $N_{\text{out}}$ ($\approx 19 \text{ ms/token}$ on RTX 3090 for Qwen2.5-7B BF16).
- **E2E Latency**: Strict linear line with slope $\approx TPOT$ and intercept $\approx TTFT$.
- **VRAM**: Monotonic upward slope as KV cache blocks are allocated dynamically during generation.

---

## Experimental Setup & Serving Controls

| Variable / Control | Value | Rationale |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` | Standard 7B instruction model |
| **Backend** | vLLM (OpenAI-compatible HTTP endpoint) | High-performance PagedAttention engine |
| **Precision** | BF16 (16-bit brain floating point) | Baseline non-quantized weights |
| **Hardware** | 1× NVIDIA RTX 3090 (24 GB VRAM) | Fixed GPU target |
| **Prefix Caching** | **DISABLED** (`--no-enable-prefix-caching`) | Avoids caching interference across runs |
| **Chunked Prefill** | **DISABLED** (`--no-enable-chunked-prefill`) | Preserves controlled baseline engine settings |
| **Concurrency** | `1` (sequential execution) | Eliminates queuing and batch scheduling interference |
| **Input Tokens ($P$)**| `128` (fixed) | Constant prefill duration baseline |
| **Output Tokens ($N_{\text{out}}$)**| `[32, 128, 512, 1024]` | 1D parameter sweep variable |
| **Warmup Requests** | `2` | Discarded to isolate cold memory allocations |
| **Measured Requests** | `10` | Repeated runs to measure statistical variance |
| **Sampling Temperature** | `0.0` (greedy decoding) | Deterministic generation path |
| **Seed** | `42` | Deterministic synthetic prompt generation |

---

## Controlled vs Independent Variables

- **Independent Variable**: Output Token Count $N_{\text{out}} \in \{32, 128, 512, 1024\}$
- **Controlled Variables**:
  - Model architecture & weights (`Qwen/Qwen2.5-7B-Instruct`, BF16)
  - Input prompt token count ($128$)
  - Request concurrency ($1$)
  - GPU clock, driver, CUDA runtime, and host OS
  - Sampling parameters (greedy, $T=0.0$)
  - Disabled prefix caching (`--no-enable-prefix-caching`)
  - Disabled chunked prefill (`--no-enable-chunked-prefill`)
- **Dependent Variables (Metrics)**:
  - End-to-End Latency (E2E, P50 & mean $\pm$ stddev)
  - Time Per Output Token (TPOT, P50 & mean)
  - Time to First Token (TTFT, ms)
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
inference-os sweep --config configs/e001b_decode_scaling.yaml

# Or running the standalone experiment script:
python experiments/E001-prefill-decode/E001B-output-length/run_e001b.py
```

Generated artifacts:
- `runs/E001B_<timestamp>/sweep_summary.json`
- `runs/E001B_<timestamp>/max_output_tokens_<N>/requests.jsonl`
- `runs/E001B_<timestamp>/max_output_tokens_<N>/telemetry.jsonl`
- `runs/E001B_<timestamp>/plots/ttft_vs_output_tokens.png`
- `runs/E001B_<timestamp>/plots/e2e_vs_output_tokens.png`
- `runs/E001B_<timestamp>/plots/tpot_vs_output_tokens.png`
- `runs/E001B_<timestamp>/plots/gpu_memory_vs_output_tokens.png`

---

## Limitations & Non-Proof Boundaries

- **Does NOT prove prefill scaling**: Prompt length is held fixed at 128 tokens; prefill scaling is isolated in E001-A.
- **Does NOT study multi-tenant concurrency**: Concurrency is fixed at 1. Concurrent scheduling dynamics are isolated in E002.
- **Hardware-Specific**: Results are specific to 1× NVIDIA RTX 3090 (Ampere architecture, 936 GB/s memory bandwidth) and should not be generalized to other GPU architectures without separate baseline characterization.
