# E002 — Concurrency & Throughput Scaling

## Primary Question

> **Why can higher concurrency improve throughput while simultaneously hurting latency?**

---

## Inference Systems Concept

In single-stream execution (concurrency = 1), the GPU is predominantly memory-bandwidth-bound during the decode phase: for every single token generated, the full model parameter weights (~14.2 GB in BF16 for 7B) must be loaded from GPU HBM/DRAM into SM registers/SRAM to compute activations for only 1 token (arithmetic intensity $\approx 1$ FLOP/byte).

When client concurrency increases ($C > 1$), modern serving systems use **continuous batching** (iteration-level scheduling):
1. **Decode Batching & Memory Bandwidth Amortization**: At each decode iteration, the model weights loaded across the memory bus are reused across all $C$ active requests simultaneously. Weight-loading memory traffic remains roughly constant while arithmetic FLOPs scale linearly with $C$. This dramatically increases arithmetic intensity and hardware efficiency, driving up **token throughput** ($\text{tokens/s}$) and **request throughput** ($\text{req/s}$).
2. **Latency Trade-Off & Queueing**:
   - Each batched GEMM step takes longer than a single-request GEMM step as matrix dimensions scale.
   - More crucially, when new requests arrive while prefill or decode iterations are executing, they must wait in scheduler queues or compete for prefill slots.
   - Time to First Token (TTFT) increases due to head-of-line prefill contention and scheduling delays.
   - Time Per Output Token (TPOT) increases as batch size grows and memory bus saturation is approached.
   - End-to-End Latency (E2E) increases monotonically with concurrency according to queueing theory and Little's Law ($L = \lambda W$).
3. **KV-Cache Memory Footprint**: Each active request maintains its own KV cache in VRAM ($2 \times 2 \times \text{layers} \times d_{\text{kv}} \times \text{seq\_len} \times \text{precision}$). Memory allocation grows directly with the number of concurrent in-flight sequences.
4. **Saturation Knee**: Beyond a hardware-dependent concurrency saturation point (where memory bandwidth or compute is saturated, or KV cache blocks exhaust), throughput plateaus while latency escalates sharply.

---

## Hypotheses

1. **Throughput Scaling & Plateau**: Token throughput ($\text{tok/s}$) and request throughput ($\text{req/s}$) will scale sub-linearly with concurrency $C$ initially (e.g. from $C=1$ to $C=8$), showing large gains as weight memory transfers are amortized across batched tokens. Beyond a saturation knee ($C \ge 16$), throughput gains will diminish and plateau.
2. **Latency Growth**: TTFT (P50 and P95) and E2E latency will increase monotonically with concurrency $C$ due to batch iteration overhead and scheduling contention.
3. **TPOT Degradation**: TPOT will increase moderately as larger batched decode GEMMs require more execution time per step.
4. **Tail Latency Amplification (P95 vs P50)**: Tail latencies (P95 TTFT and P95 E2E) will spread wider at high concurrency ($C=16, 32$) due to prefill scheduling interference and queue wait times.
5. **GPU Utilization & VRAM**: Average GPU compute utilization (%) and peak VRAM allocated will increase monotonically with concurrency due to larger batch execution and KV cache allocations for $C$ active streams.
6. **Error Rate**: Under proper server sizing (`max_num_seqs >= 32`), error rate should remain 0% across $C \in [1..32]$.

---

## Experimental Setup & Serving Controls

| Variable / Control | Value | Rationale |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` | Fixed standard 7B instruction model |
| **Precision** | BF16 (`bfloat16`) | Baseline non-quantized weights |
| **Hardware** | 1× NVIDIA GeForce RTX 3090 (24 GB VRAM) | Fixed GPU target |
| **Serving Backend** | vLLM (OpenAI-compatible HTTP endpoint) | Continuous batching engine |
| **Server Concurrency Cap (`--max-num-seqs`)** | **`64`** | Must exceed max sweep concurrency ($32$) so server scheduler does not throttle artificially |
| **Prefix Caching** | **DISABLED** (`--no-enable-prefix-caching`) | Prevents KV cache hit shortcuts across concurrent requests |
| **Chunked Prefill** | **DISABLED** (`--no-enable-chunked-prefill`) | Preserves pure baseline prefill/decode scheduling |
| **GPU Memory Utilization** | **`0.90`** | Standard 90% allocation for weights + KV cache buffer |
| **Input Tokens ($P$)** | **`512`** (fixed) | Constant context length |
| **Output Tokens ($N_{\text{out}}$)** | **`128`** (fixed) | Constant decode length |
| **Concurrency Sweep Values ($C$)** | **`[1, 2, 4, 8, 16, 32]`** | Independent variable under closed-loop client load |
| **Sampling Temperature** | `0.0` (greedy decoding) | Deterministic output lengths and generation paths |
| **Seed** | `42` | Deterministic synthetic prompt generation |

---

## Controlled vs Independent Variables

- **Independent Variable**: Client Concurrency $C \in \{1, 2, 4, 8, 16, 32\}$
- **Controlled Variables**:
  - Model architecture & weights (`Qwen/Qwen2.5-7B-Instruct`, BF16)
  - Input prompt length ($512$ tokens)
  - Output generation length ($128$ tokens)
  - Single server process lifetime across the entire sweep
  - Server parameters: `--max-num-seqs 64`, `--gpu-memory-utilization 0.90`
  - Disabled prefix caching (`--no-enable-prefix-caching`)
  - Disabled chunked prefill (`--no-enable-chunked-prefill`)
  - Closed-loop concurrency model: strictly maintains up to $C$ requests in flight simultaneously
- **Dependent Variables (Metrics)**:
  - Request throughput ($\text{req/s} = \text{successful\_requests} / \text{total\_wall\_clock\_duration}$)
  - Output-token throughput ($\text{tok/s} = \text{total\_actual\_output\_tokens} / \text{total\_wall\_clock\_duration}$)
  - Time to First Token (TTFT: P50, P95, mean)
  - Time Per Output Token (TPOT: P50, P95, mean)
  - End-to-End Latency (E2E: P50, P95, mean)
  - Error rate ($\text{failed\_requests} / \text{total\_requests}$)
  - GPU Compute Utilization (Peak and Avg %)
  - GPU Memory Allocation (Peak and Avg MiB)

---

## Methodology & Closed-Loop Execution Protocol

1. **Closed-Loop Concurrency**:
   - The harness maintains an exact worker pool of size $\min(C, N)$.
   - When any request finishes, the worker immediately pulls the next request from the queue until all $N$ requests finish.
   - At no point do active in-flight requests exceed $C$.
2. **Exact Token Count Validation**:
   - Output tokens are validated and counted using the tokenizer over the generated text, never assumed from HTTP chunks.
3. **Wall-Clock Timing**:
   - Throughput metrics are calculated using monotonic nanosecond wall-clock duration spanning from the start of request generation until the last worker returns.

---

## Canonical vLLM Server Launch Command

Run on the target GPU host (RTX 3090):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.90 \
  --port 18000
```

---

## Execution Protocol

### Step 1: Pilot Verification Sweep (Smoke Test)
Before running the full canonical sweep, run a cheap pilot to verify server connectivity, concurrency scaling, telemetry, and error-free persistence:

```bash
python experiments/E002-concurrency/run_e002.py --pilot --base-url http://localhost:18000
```
- Concurrency points: $[1, 4, 8]$
- Requests per point: $15$

### Step 2: Canonical Benchmark Sweep
After the pilot passes cleanly without errors or deadlocks, run the full canonical sweep against the same running server instance:

```bash
python experiments/E002-concurrency/run_e002.py --config configs/e002_concurrency.yaml --base-url http://localhost:18000
```
- Concurrency points: $[1, 2, 4, 8, 16, 32]$
- Requests per point: $100$

---

## Core Visualizations Generated

1. `throughput_requests_vs_concurrency.png`: Concurrency vs Request throughput (req/s).
2. `throughput_tokens_vs_concurrency.png`: Concurrency vs Output token throughput (tok/s).
3. `latency_vs_concurrency.png`: Concurrency vs TTFT, TPOT, and E2E latency (P50 and P95).
4. `gpu_metrics_vs_concurrency.png`: Concurrency vs Peak VRAM and GPU Utilization.

---

## Limitations & What This Experiment Does NOT Prove

1. **Closed-Loop vs Open-Loop**: This experiment uses closed-loop concurrency (fixed number of active users waiting for responses). Real-world production traffic is open-loop (Poisson arrival processes where requests arrive independently of server response times). Saturation behavior under open-loop queues diverges earlier and more steeply.
2. **Fixed Sequence Lengths**: Prompts ($512$) and generations ($128$) are uniform. Variable prompt/output length mixtures introduce memory fragmentation and scheduling bubbles not observed here.
3. **Disabled Advanced Features**: Prefix caching and chunked prefill are disabled to isolate raw batching dynamics. In production, chunked prefill mitigates TTFT spikes, and prefix caching reduces prefill compute.
4. **Single-GPU Sizing**: Results characterize a single RTX 3090 (24 GB) running Qwen2.5-7B BF16 and do not directly translate to multi-GPU tensor-parallel configurations.
