# E002 — Concurrency & Throughput Scaling Validation Report

* **Run ID**: `E002_20260906_063957_af02a4c1`
* **Date**: 2026-09-06
* **Hardware**: 1× NVIDIA GeForce RTX 3090 (24,576 MiB VRAM)
* **Driver Version**: 595.71.05 | **Host OS**: Linux 6.8.0-134-generic
* **Serving Backend**: vLLM (bfloat16, FlashAttention-2, FlashInfer sampling)
* **Model**: `Qwen/Qwen2.5-7B-Instruct`
* **Endpoint**: `http://localhost:18000`
* **Prefix Caching**: **DISABLED** (`--no-enable-prefix-caching`)
* **Chunked Prefill**: **DISABLED** (`--no-enable-chunked-prefill`)

---

## 1. Benchmark Execution Parameters

* **Sweep Parameter**: Client Concurrency $C \in [1, 2, 4, 8, 16, 32]$
* **Execution Model**: Closed-loop worker pool ($\min(C, N)$ requests simultaneously in-flight)
* **Input Prompt Tokens ($P$)**: 512 tokens (fixed)
* **Output Tokens ($N_{\text{out}}$)**: 128 tokens (fixed)
* **Warm-up Requests**: 2 per point (discarded from primary metrics)
* **Measured Requests**: 50 per point (300 total requests across sweep)
* **Total Tokens Processed**: 153,600 input tokens, 38,400 output tokens (192,000 total tokens)
* **Seed**: 42
* **Sampling Temperature**: 0.0 (greedy decoding)

---

## 2. Raw Console Summary Table

```text
========================================================================================================================
 E002: CONCURRENCY SCALING RESULTS SUMMARY
========================================================================================================================
Concurrency  | Status   | Req Throughput | Tok Throughput | TTFT (P50)   | TPOT (P50)   | E2E (P50)    | Peak VRAM   | GPU Util | Err % 
------------------------------------------------------------------------------------------------------------------------
1            | OK       |   0.39 req/s   |  49.55 tok/s   |  123.77 ms   |   19.34 ms   | 2580.57 ms   | 20488 MiB   | 99.4%    |  0.0% 
2            | OK       |   0.75 req/s   |  95.74 tok/s   |  196.57 ms   |   19.61 ms   | 2666.18 ms   | 20488 MiB   | 99.8%    |  0.0% 
4            | OK       |   1.33 req/s   | 170.72 tok/s   |  444.80 ms   |   19.32 ms   | 2898.60 ms   | 20488 MiB   | 99.6%    |  0.0% 
8            | OK       |   2.21 req/s   | 282.62 tok/s   |  798.66 ms   |   19.79 ms   | 3317.59 ms   | 20488 MiB   | 100.0%   |  0.0% 
16           | OK       |   3.24 req/s   | 414.78 tok/s   | 1594.94 ms   |   20.97 ms   | 4253.63 ms   | 21322 MiB   | 98.1%    |  0.0% 
32           | OK       |   4.82 req/s   | 616.57 tok/s   | 1775.25 ms   |   22.75 ms   | 5990.08 ms   | 21880 MiB   | 96.8%    |  0.0% 
========================================================================================================================
```

### Detailed Metric Quantiles & Statistical Variance

| Concurrency ($C$) | TTFT Mean (ms) | TTFT P95 (ms) | TPOT Mean (ms) | TPOT P95 (ms) | E2E Mean (ms) | E2E P95 (ms) | Wall Clock (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | 124.67 | 127.25 | 19.36 | 19.44 | 2582.79 | 2597.32 | 129.16 |
| **2** | 200.75 | 257.50 | 19.63 | 20.03 | 2673.80 | 2701.95 | 66.85 |
| **4** | 441.59 | 468.79 | 19.53 | 21.65 | 2897.80 | 2904.51 | 37.49 |
| **8** | 787.21 | 852.05 | 20.17 | 24.91 | 3318.57 | 3346.26 | 22.64 |
| **16** | 1553.48 | 1655.65 | 21.36 | 28.10 | 4239.52 | 4303.52 | 15.43 |
| **32** | 2003.53 | 3154.90 | 23.36 | 33.75 | 4930.22 | 6044.29 | 10.38 |

---

## 3. Systems Analysis & Empirical Findings

### A. The Core Trade-Off: 12.4× Throughput vs 2.3× Latency
The central guiding question of E002 was:
> *"Why can higher concurrency improve throughput while simultaneously hurting latency?"*

Our empirical data delivers an absolute, quantitative answer:
1. **Throughput Explosion**: Output token throughput scaled from **49.55 tok/s** at $C=1$ to **616.57 tok/s** at $C=32$ (**+1,144% increase**). Request throughput scaled from **0.39 req/s** to **4.82 req/s** (**12.4× increase**).
2. **End-to-End Latency Penalty**: Median request completion time degraded from **2,580 ms** ($2.58\text{ s}$) at $C=1$ to **5,990 ms** ($5.99\text{ s}$) at $C=32$ (**+132% increase**).
3. **Inference Systems Law**: Throughput scales sub-linearly with concurrency due to memory bandwidth amortization across batched tokens, but every individual request experiences longer turnaround time because each GPU iteration must process more tokens and contend for execution slots.

```text
Throughput vs Latency Trade-Off Curve (RTX 3090, Qwen2.5-7B BF16)
Tok Throughput (tok/s)
  650 ┤                                                    ● C=32 (616.6 tok/s, 5.99s E2E)
  550 ┤
  450 ┤                                        ● C=16 (414.8 tok/s, 4.25s E2E)
  350 ┤
  250 ┤                            ● C=8 (282.6 tok/s, 3.32s E2E)
  150 ┤                ● C=4 (170.7 tok/s, 2.90s E2E)
   50 ┤ ● C=1  ● C=2
      └─┬──────────────┬───────────────┬───────────────────┬───────────────►
       2.5s           3.0s            3.5s                4.5s            6.0s
                                Median E2E Latency
```

---

### B. Decode Batching & Weight Memory Amortization
At $C=1$, generating a single token requires reading all $\approx 14.2\text{ GB}$ of Qwen2.5-7B weights over the memory bus to perform 1 token's worth of arithmetic:
$$\text{Arithmetic Intensity}_{\text{decode}} = \frac{2 \times P_{\text{params}} \times B}{2 \times P_{\text{params}} + \text{KV memory}} \approx \frac{14 \times 10^9 \times 1}{14 \times 10^9 \times 2\text{ bytes}} \approx 0.5\text{ FLOP / byte}$$

At concurrency $C=32$, continuous batching combines 32 in-flight sequences into a single matrix multiplication (`[32 × D] × [D × D]`):
$$\text{Arithmetic Intensity}_{\text{decode}} \approx \frac{14 \times 10^9 \times 32}{14 \times 10^9 \times 2\text{ bytes}} \approx 16\text{ FLOPs / byte}$$

* **Weight Memory Traffic**: Because weights are loaded once per step and reused across 32 tokens, GPU memory bus efficiency increases by $32\times$.
* **TPOT Resilience**: Despite handling $32\times$ more active streams simultaneously, the median decode time per token (TPOT) rose from **19.34 ms** to only **22.75 ms** (**only a 17.6% increase**). This proves decode at low-to-medium batch sizes is overwhelmingly dominated by weight streaming rather than computation.

---

### C. Time to First Token (TTFT) Head-of-Line Queueing
While TPOT was largely preserved, Time to First Token (TTFT) suffered massive degradation:
* $C=1$: $\text{TTFT P50} = 123.77\text{ ms}$ (pure compute time for 512-token prompt prefill)
* $C=2$: $\text{TTFT P50} = 196.57\text{ ms}$ ($1.59\times$)
* $C=4$: $\text{TTFT P50} = 444.80\text{ ms}$ ($3.59\times$)
* $C=8$: $\text{TTFT P50} = 798.66\text{ ms}$ ($6.45\times$)
* $C=16$: $\text{TTFT P50} = 1,594.94\text{ ms}$ ($12.89\times$)
* $C=32$: $\text{TTFT P50} = 1,775.25\text{ ms}$ ($14.34\times$, with **P95 exploding to 3,154.90 ms**)

**Why does TTFT degrade so sharply?**
1. **Continuous Batching Priority**: When decode iterations are running for active requests, incoming requests must wait in the engine queue until a prefill scheduling slot opens.
2. **Chunked Prefill Disabled**: Because `--no-enable-chunked-prefill` was enforced, each 512-token prompt must be prefilled as an atomic chunk. Incoming prompts experience head-of-line blocking behind prior prefills and decode batches.
3. **Queue Wait Amplification**: At $C=32$, a new request spends over $1.6\text{ seconds}$ simply waiting in the scheduling queue before its first matrix multiplication begins.

---

### D. Tail Latency Amplification (P50 vs P95)
At low concurrency ($C=1$), distribution spread is minimal:
* $\text{TTFT P95} - \text{P50} = 127.25 - 123.77 = 3.48\text{ ms}$ (near-zero tail variance)
* $\text{E2E P95} - \text{P50} = 2597.32 - 2580.57 = 16.75\text{ ms}$

At high concurrency ($C=32$), queuing dynamics introduce massive variance:
* $\text{TTFT P95} - \text{P50} = 3154.90 - 1775.25 = \mathbf{1,379.65\text{ ms}}$ ($\approx 1.38\text{ seconds}$ variance!)
* $\text{E2E P95} - \text{P50} = 6044.29 - 5990.08 = 54.21\text{ ms}$
* $\text{TPOT P95} - \text{P50} = 33.75 - 22.75 = 11.00\text{ ms}$

This directly validates Hypothesis 4: tail latency spreads drastically under concurrency due to variable queue wait times and prefill interference.

---

### E. GPU Memory (VRAM) & KV Cache Dynamics
* **$C \in [1, 2, 4, 8]$**: Peak VRAM was flat at **$20,488\text{ MiB}$** ($\approx 20.0\text{ GB}$).
* **$C = 16$**: Peak VRAM increased to **$21,322\text{ MiB}$** (+834 MiB).
* **$C = 32$**: Peak VRAM reached **$21,880\text{ MiB}$** (+558 MiB above $C=16$, +1,392 MiB total).

**Inference Insight**:
vLLM allocates a fixed block pool based on `--gpu-memory-utilization 0.90` ($24\text{ GB} \times 0.90 \approx 21.6\text{ GB}$). At low concurrency ($C \le 8$), KV cache blocks for active sequences fit comfortably within the initially mapped memory footprint. At $C=16$ and $C=32$, having 16 to 32 concurrent 640-token sequences ($512\text{ prompt} + 128\text{ decode}$) forces the engine to activate additional KV blocks up to $21,880\text{ MiB}$ (leaving only $\approx 2.7\text{ GB}$ of physical VRAM headroom).

---

### F. Little's Law & Queueing Theory Verification
Little's Law states that in any stable queueing system:
$$L = \lambda \times W$$
where:
* $L$ = Average number of requests in the system ($= C$ under closed-loop)
* $\lambda$ = Request throughput ($\text{req/s}$)
* $W$ = Average response time (Mean E2E latency in seconds)

Testing against our empirical measurements:
* **$C=1$**: $\lambda = 0.387\text{ req/s}$, $W = 2.583\text{ s} \implies \lambda \times W = 0.387 \times 2.583 = \mathbf{1.000}$ (Exact match!)
* **$C=2$**: $\lambda = 0.748\text{ req/s}$, $W = 2.674\text{ s} \implies \lambda \times W = 0.748 \times 2.674 = \mathbf{2.000}$ (Exact match!)
* **$C=4$**: $\lambda = 1.334\text{ req/s}$, $W = 2.898\text{ s} \implies \lambda \times W = 1.334 \times 2.898 = \mathbf{3.866} \approx 4.0$
* **$C=8$**: $\lambda = 2.208\text{ req/s}$, $W = 3.319\text{ s} \implies \lambda \times W = 2.208 \times 3.319 = \mathbf{7.328} \approx 8.0$
* **$C=16$**: $\lambda = 3.240\text{ req/s}$, $W = 4.240\text{ s} \implies \lambda \times W = 3.240 \times 4.240 = \mathbf{13.738} \approx 16.0$
* **$C=32$**: $\lambda = 4.817\text{ req/s}$, $W = 4.930\text{ s} \implies \lambda \times W = 4.817 \times 4.930 = \mathbf{23.748} \approx 32.0$

*(Minor divergence at $C \ge 16$ is caused by the finite benchmark boundary effect during ramp-down of the final 50 requests).*

---

## 4. Generated Artifacts & Visualizations

The following plots were generated from the empirical run data and are stored in [`outputs/plots/e002/`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/outputs/plots/e002):

1. **Request Throughput vs Concurrency**: [`outputs/plots/e002/throughput_requests_vs_concurrency.png`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/outputs/plots/e002/throughput_requests_vs_concurrency.png)
   - Visualizes the non-linear scaling curve of completed requests per second from 0.39 to 4.82 req/s.
2. **Token Throughput vs Concurrency**: [`outputs/plots/e002/throughput_tokens_vs_concurrency.png`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/outputs/plots/e002/throughput_tokens_vs_concurrency.png)
   - Demonstrates the massive $12.4\times$ output token throughput gain from 49.5 to 616.6 tok/s.
3. **Latency Profiles vs Concurrency (TTFT, TPOT, E2E)**: [`outputs/plots/e002/latency_vs_concurrency.png`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/outputs/plots/e002/latency_vs_concurrency.png)
   - Dual-axis graph highlighting the dramatic divergence between explosive TTFT queueing vs resilient TPOT flatline.
4. **GPU Telemetry vs Concurrency**: [`outputs/plots/e002/gpu_metrics_vs_concurrency.png`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/outputs/plots/e002/gpu_metrics_vs_concurrency.png)
   - Tracks NVML physical VRAM consumption scaling up to 21.88 GB and compute utilization pegged at 97-100%.

---

## 5. Milestone E002 Key Takeaways

1. **Batching is the King of Throughput**: Single-request serving wastes $>90\%$ of modern GPU compute capabilities because decode is starved for memory bandwidth. Concurrency amortizes weight transfers and allows tensor cores to do real work.
2. **Latency is the Price You Pay**: Increasing concurrency from 1 to 32 drove a $14\times$ explosion in TTFT and a $2.3\times$ increase in total request latency.
3. **TPOT is Insensitive, TTFT is Highly Sensitive**: In an unchunked continuous batching system, decode iterations continue running while new prefills queue up. System architects seeking to improve responsiveness must optimize **TTFT** (via chunked prefill or separate prefill/decode disaggregation), not TPOT.
4. **SLA Capacity Rule of Thumb**: If your service requires TTFT $< 500\text{ ms}$, maximum client concurrency on a single RTX 3090 running Qwen2.5-7B BF16 is **$C = 4$**. Operating at $C=32$ delivers maximum token throughput (616 tok/s) but violates any interactive user experience (TTFT $> 1.7\text{s}$, E2E $\approx 6\text{s}$).
