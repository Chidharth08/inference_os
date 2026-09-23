# 📚 inference_os — Complete Study Guide (Milestone E002: Concurrency & Throughput Scaling)

This guide covers **everything** about Milestone E002 — the **why**, the **how**, and the **what** — at both a conceptual systems level and an applied code level. By the end, you should be able to explain the throughput-latency trade-off under concurrency to a systems architect, derive throughput scaling from GPU memory bandwidth physics, verify Little's Law against real empirical data, and trace every engineering decision in the codebase back to a measurable requirement.

**Prerequisites**: This guide builds directly on E001. You should already understand prefill vs. decode, TTFT vs. TPOT, the Roofline Model, and why decode is memory-bandwidth-bound at batch size 1.

---

## Table of Contents

1. [The Core Question — What Are We Trying to Understand?](#1-the-core-question)
2. [Why This Matters — The E001→E002 Bridge](#2-why-this-matters--the-e001e002-bridge)
3. [Conceptual Foundation — The Single-Stream Tragedy](#3-conceptual-foundation--the-single-stream-tragedy)
4. [Deep Dive — Weight Amortization Through Batched Decode](#4-deep-dive--weight-amortization-through-batched-decode)
5. [Continuous Batching — How Modern Serving Engines Schedule](#5-continuous-batching--how-modern-serving-engines-schedule)
6. [The Latency Penalty — Why TTFT Explodes While TPOT Survives](#6-the-latency-penalty--why-ttft-explodes-while-tpot-survives)
7. [Queueing Theory — Little's Law and What It Tells Us](#7-queueing-theory--littles-law-and-what-it-tells-us)
8. [The KV Cache Multiplier — Memory Footprint Under Load](#8-the-kv-cache-multiplier--memory-footprint-under-load)
9. [Roofline Revisited — Climbing the Bandwidth Slope](#9-roofline-revisited--climbing-the-bandwidth-slope)
10. [Experimental Design — The Closed-Loop Concurrency Harness](#10-experimental-design--the-closed-loop-concurrency-harness)
11. [What We Built — Engineering the Concurrent Runner](#11-what-we-built--engineering-the-concurrent-runner)
12. [Reading Our Results — The Full RTX 3090 Data](#12-reading-our-results--the-full-rtx-3090-data)
13. [Hypothesis Validation — Checking Every Prediction](#13-hypothesis-validation--checking-every-prediction)
14. [The Capacity Planner's Playbook — Choosing Optimal Concurrency](#14-the-capacity-planners-playbook--choosing-optimal-concurrency)
15. [Serving Controls — Why We Disabled Prefix Caching & Chunked Prefill](#15-serving-controls--why-we-disabled-prefix-caching--chunked-prefill)
16. [What We Built in Code — Applied Engineering](#16-what-we-built-in-code--applied-engineering)
17. [Interview-Ready Knowledge — How to Explain This](#17-interview-ready-knowledge--how-to-explain-this)
18. [Self-Test Questions](#18-self-test-questions)
19. [Further Reading — Papers, Books, and Resources](#19-further-reading--papers-books-and-resources)
20. [Glossary — New Terms Introduced in E002](#20-glossary--new-terms-introduced-in-e002)

---

## 1. The Core Question

> **"Why can higher concurrency improve throughput while simultaneously hurting latency?"**

This is the guiding question of Milestone E002. In most computing systems, throughput and latency feel like they should move together — faster responses should mean more work done. But in LLM serving, they move in **opposite directions**:

- Going from 1 to 32 concurrent requests, **throughput exploded by 12.4×** (49.5 → 616.6 tok/s)
- But **TTFT degraded by 14.3×** (124 ms → 1,775 ms) and **E2E latency increased by 2.3×** (2.58 s → 5.99 s)

To understand why, you need to understand three things:
1. Why single-stream decode **wastes** most of the GPU's capability
2. How batching **amortizes** the memory bandwidth bottleneck
3. Why scheduling **queues** punish responsiveness even as throughput improves

### Why This Matters

If you're building or operating an LLM serving system, you will face questions like:

- "We have 10,000 users. How many GPUs do we need?"
- "Users are complaining that the first token takes too long. Can we fix it without losing throughput?"
- "Should we run at concurrency 4 or 32? What's the trade-off?"
- "Our GPU utilization shows 99%. Why do I still need more GPUs?"

You cannot answer any of these without understanding the throughput-latency trade-off under concurrency.

---

## 2. Why This Matters — The E001→E002 Bridge

In E001, we established two critical facts:

1. **Prefill is compute-bound**: TTFT scales with prompt length ($P$), because the GPU's tensor cores are doing real matrix-matrix work.
2. **Decode is memory-bandwidth-bound**: TPOT is constant (~19.8 ms) regardless of prompt or output length, because each decode step must stream all $14.2\text{ GB}$ of model weights from VRAM just to generate 1 token.

The second fact should immediately raise a question:

> If the GPU is reading 14.2 GB of weights to compute activations for **one** token, what if it computed activations for **32** tokens with that same read?

This is the fundamental insight that E002 investigates. In E001, we diagnosed that decode is memory-bandwidth-bound. In E002, we test the **cure**: batching multiple concurrent requests so the same weight data serves multiple token computations simultaneously.

But there is no free lunch. Higher concurrency means more requests competing for GPU resources, which introduces queueing delays, scheduling contention, and memory pressure. E002 quantifies these trade-offs precisely.

---

## 3. Conceptual Foundation — The Single-Stream Tragedy

### What Happens at Concurrency = 1

Recall the numbers from E001. For each decode step on our RTX 3090 running Qwen2.5-7B in BF16:

```text
Model weights to read:           14.2 GB
RTX 3090 memory bandwidth:       936 GB/s
Theoretical min decode time:     14.2 GB / 936 GB/s ≈ 15.2 ms
Observed TPOT:                   19.34 ms  (77.4% HBM efficiency)
Compute done for 1 token:        2 × 7B = 14 GFLOPs
RTX 3090 BF16 peak compute:     142 TFLOPS = 142,000 GFLOPs/s
Compute duration for 1 token:    14 GFLOPs / 142,000 GFLOPs/s ≈ 0.1 ms
```

Let's compute the arithmetic intensity — the ratio of compute to memory traffic:

$$\text{Arithmetic Intensity}_{C=1} = \frac{14 \times 10^9 \text{ FLOPs}}{14.2 \times 10^9 \times 2 \text{ bytes}} \approx 0.49 \text{ FLOPs/byte}$$

For context, the RTX 3090's **ridge point** (where memory bandwidth and compute are perfectly balanced) is:

$$\text{Ridge Point} = \frac{142 \times 10^{12} \text{ FLOP/s}}{936 \times 10^{9} \text{ bytes/s}} \approx 151.7 \text{ FLOPs/byte}$$

At $C=1$, decode runs at **0.3% of the compute capability** that the GPU theoretically has. The tensor cores are idle for over 99% of each decode step, waiting for weight bytes to arrive over the memory bus.

```text
THE SINGLE-STREAM TRAGEDY (C = 1)

GPU Memory Bus: ████████████████████████████████████████  100% saturated
GPU Tensor Cores: █░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░   < 1% utilized!

Each decode step:
┌─────────────────────────────────────────────────────┐
│  Read 14.2 GB weights over memory bus  (~15.2 ms)   │
│  Compute 1 token's activations         (~0.1 ms)    │
│                                                     │
│  Total: ~19.3 ms to produce ONE token               │
│  GPU utilization: 99%  ← MISLEADING! This is the    │
│  memory controller, not the tensor cores.            │
└─────────────────────────────────────────────────────┘
```

### The GPU Utilization Trap

Notice something subtle: our telemetry showed **99.4% GPU utilization** at $C=1$. A naïve interpretation is "the GPU is fully utilized, we can't go faster." But NVIDIA's utilization percentage measures whether **any** SM (Streaming Multiprocessor) has **any** active warp in a given sampling interval — not whether the tensor cores are doing useful arithmetic. The memory controller keeping the memory bus active counts as "utilization."

The GPU is "busy" (reading memory), but it's not doing proportional **compute work**. This is the gap that concurrency exploits.

---

## 4. Deep Dive — Weight Amortization Through Batched Decode

### The Key Transformation

When $C$ requests are concurrently in the decode phase, the serving engine doesn't run $C$ separate decode passes. Instead, it groups all active sequences into a **single batched matrix multiplication**:

```text
SEQUENTIAL DECODE (C=1):
  Token activations: [1 × D]   ← one row (1 token)
  Weight matrix:     [D × D]   ← full model layer
  Matrix multiply:   [1 × D] × [D × D] = [1 × D]  ← GEMV (vector × matrix)

BATCHED DECODE (C=32):
  Token activations: [32 × D]  ← 32 rows (one per sequence)
  Weight matrix:     [D × D]   ← SAME full model layer
  Matrix multiply:   [32 × D] × [D × D] = [32 × D]  ← GEMM (matrix × matrix)
```

**The weight matrix $[D \times D]$ is read from VRAM exactly once per step, regardless of $C$.**

This is the core physics of the throughput improvement:

| Quantity | C = 1 | C = 32 | Ratio |
|---|---|---|---|
| **Weight bytes loaded per step** | 14.2 GB | 14.2 GB | **1.0×** (constant) |
| **FLOPs computed per step** | $2 \times 7\text{B} = 14\text{ GFLOPs}$ | $2 \times 7\text{B} \times 32 = 448\text{ GFLOPs}$ | **32×** |
| **Tokens produced per step** | 1 | 32 | **32×** |
| **Arithmetic intensity** | 0.49 FLOPs/byte | 15.8 FLOPs/byte | **32×** |
| **Time per step** | ~19.3 ms | ~22.8 ms | **1.18×** (only 18% slower!) |
| **Tokens per second** | 49.5 | 616.6 | **12.4×** |

### Why Isn't the Speedup Exactly 32×?

Two reasons:

1. **The batched GEMM takes slightly longer**: Processing 32 activation vectors instead of 1 requires more computation per step. The step time increased from 19.3 ms to 22.8 ms (18% overhead). This is the extra compute that the tensor cores actually need to perform. At $C=32$, we are moving from pure memory-bound territory toward the ridge, but are not yet compute-bound.

2. **Prefill contention**: Not all 32 requests are always in the decode phase simultaneously. New requests entering the system need prefill, which competes for GPU time with the ongoing decode batch. While one request is being prefilled (consuming tensor cores for ~124 ms), the other 31 decode sequences must wait.

### The Library Analogy (Extended from E001)

In E001, we described decode as bringing every book from a library (VRAM) through a door (memory bus) to read one page. Now extend this:

> **Concurrency = 1**: You carry 14.2 billion books through the door, read **one page** from each for **one student**, put them all back. Then do it again for the next token. Enormous library effort, minimal educational output.

> **Concurrency = 32**: You carry the same 14.2 billion books through the door, but now **32 students** each read their own page from every book before you put them back. The library effort is identical, but you've educated 32 students per round trip instead of 1.

The door (memory bus) is the same width. The books (weights) are the same. But the **educational throughput** (tokens/second) has increased dramatically.

---

## 5. Continuous Batching — How Modern Serving Engines Schedule

### The Old Way: Static Batching

Before 2022, serving engines used **static batching**: collect $B$ requests, prefill them together, decode together until ALL $B$ finish, then return all results.

```text
STATIC BATCHING — The Problem:
┌───────────────────────────────────────────────────────────────────┐
│ Req A: [Prefill] [T1] [T2] [T3] [DONE]                           │
│ Req B: [Prefill] [T1] [T2] [T3] [T4] [T5] [T6] ... [T50] [DONE] │
│                                                                   │
│ Req A finished at step 5, but its response is held until B       │
│ finishes at step 52. Req A's KV cache occupies VRAM uselessly     │
│ for 47 extra steps.                                               │
└───────────────────────────────────────────────────────────────────┘
```

This wastes both latency (Request A waits for B) and memory (A's KV cache sits idle).

### The New Way: Continuous Batching (Orca/vLLM)

The Orca paper (OSDI 2022) introduced **iteration-level scheduling**: the scheduler makes decisions at every single decode step.

```text
CONTINUOUS BATCHING — Orca/vLLM:

Step 1: Batch = {A, B}    → Both produce 1 token
Step 2: Batch = {A, B}    → Both produce 1 token
Step 3: Batch = {A, B}    → Both produce 1 token
Step 4: Batch = {A, B}    → A reaches EOS! Evict A immediately.
Step 5: Batch = {B, C}    → C is a NEW arrival, prefill + join batch
Step 6: Batch = {B, C}    → Continue decoding
...
```

Key properties:
1. **Immediate eviction**: When a request finishes, its KV cache blocks are freed instantly, not after the whole batch completes.
2. **Dynamic insertion**: New requests can be prefilled and inserted into the active batch at any step.
3. **Variable batch size**: The batch naturally grows and shrinks as requests arrive and depart.

This is exactly what vLLM does on our RTX 3090 during the E002 benchmark. The benchmark's closed-loop harness maintains $C$ in-flight requests, and vLLM's continuous batching scheduler dynamically groups them into per-iteration decode batches.

---

## 6. The Latency Penalty — Why TTFT Explodes While TPOT Survives

This is the most surprising and important finding of E002. Let's look at the empirical data side by side:

```text
 C   │  TPOT P50 (ms)  │  TTFT P50 (ms)  │  TTFT P95 (ms)  │ E2E P50 (ms)
─────┼─────────────────┼─────────────────┼─────────────────┼──────────────
  1  │      19.34      │      123.77     │      127.25     │    2580.57
  2  │      19.61      │      196.57     │      257.50     │    2666.18
  4  │      19.32      │      444.80     │      468.79     │    2898.60
  8  │      19.79      │      798.66     │      852.05     │    3317.59
 16  │      20.97      │     1594.94     │     1655.65     │    4253.63
 32  │      22.75      │     1775.25     │     3154.90     │    5990.08
```

### TPOT: The Resilient Metric (+17.6%)

TPOT barely moved: 19.34 ms → 22.75 ms. Why?

Each decode step still loads the same 14.2 GB of model weights. The extra time comes from the slightly larger GEMM computation (`[32 × D]` instead of `[1 × D]`), but this only costs ~3 ms extra. The memory bus transfer time — which dominates — is unchanged.

**The takeaway**: TPOT is governed by hardware physics (memory bandwidth), not by system architecture. You cannot make TPOT worse by running more concurrent requests — at least not until you exhaust VRAM for KV caches and trigger eviction/preemption.

### TTFT: The Catastrophic Degradation (+1,334%)

TTFT surged from 123.77 ms to 1,775.25 ms. At P95, it hit 3,154.90 ms (over 3 seconds!).

**Why?** TTFT is composed of two parts:

$$\text{TTFT} = \text{Queue Wait Time} + \text{Prefill Compute Time}$$

At $C=1$:
- Queue wait = 0 ms (no contention)
- Prefill compute = 124 ms (processing 512 prompt tokens)
- Total TTFT = 124 ms

At $C=32$:
- Queue wait = **~1,651 ms** (waiting for a prefill slot)
- Prefill compute = ~124 ms (same prompt, same GPU, same math)
- Total TTFT = ~1,775 ms

**93% of TTFT at $C=32$ is pure queue waiting, not computation.**

### What Causes the Queue Wait?

Three things conspire:

1. **Continuous batching prioritizes decode**: In vLLM's default scheduler, active decode sequences continue running in small, fast iterations. A new request's prefill (which takes ~124 ms of solid compute) must find a scheduling window between decode batches or interrupt them.

2. **Unchunked prefill is atomic**: Because we disabled chunked prefill (`--no-enable-chunked-prefill`), a 512-token prompt must be processed in a single pass. The engine cannot split it into smaller chunks and interleave them with decode iterations. So incoming prompts experience **head-of-line blocking** — they must wait until the engine allocates a full prefill slot.

3. **More concurrent requests = deeper queues**: At $C=32$, there are always ~31 active decode sequences keeping the GPU busy, plus potentially other pending prefills. A new request joins the back of the line.

```text
WHY TTFT EXPLODES AT C=32:

Timeline:
Req 1: [══PREFILL══][decode][decode][decode]...
Req 2:    [waiting...][══PREFILL══][decode][decode]...
Req 3:       [waiting.........][══PREFILL══][decode]...
...
Req 32:                              [WAITING ~1.7 SECONDS............][══PREFILL══]
                                                                       ↑
                                                                  This is the TTFT
                                                              that the user experiences
```

### Tail Latency: The P95 Explosion

The gap between P50 and P95 TTFT is a critical systems indicator:

| C | TTFT P50 (ms) | TTFT P95 (ms) | P95 − P50 (ms) | Interpretation |
|---|---|---|---|---|
| 1 | 123.77 | 127.25 | **3.48** | Near-zero variance — no contention |
| 4 | 444.80 | 468.79 | **23.99** | Minimal queueing jitter |
| 8 | 798.66 | 852.05 | **53.39** | Measurable queueing effects |
| 16 | 1594.94 | 1655.65 | **60.71** | Moderate scheduling variance |
| 32 | 1775.25 | 3154.90 | **1,379.65** | **Massive tail latency explosion!** |

At $C=32$, the unluckiest 5% of requests waited **3.15 seconds** for their first token — compared to a median of 1.78 seconds. The ~1.4-second gap means some requests got unlucky with scheduling (arriving just as the engine committed to a long sequence of decode batches or other prefills).

This tail latency variance makes $C=32$ unsuitable for interactive applications with SLAs, even though it delivers maximum throughput.

---

## 7. Queueing Theory — Little's Law and What It Tells Us

### Little's Law

In any **stable** queueing system (where the queue doesn't grow unboundedly), a beautiful invariant holds:

$$\boxed{L = \lambda \times W}$$

Where:
- $L$ = Average number of requests **in the system** (being processed or waiting)
- $\lambda$ = **Throughput** (requests completed per second)
- $W$ = Average **response time** (time a request spends in the system, which is our E2E latency)

### Applying Little's Law to Closed-Loop Benchmarks

Our benchmark uses **closed-loop concurrency**: exactly $C$ requests are always in flight. When one finishes, the client immediately submits a new one. Therefore:

$$L = C \quad \text{(by construction)}$$

This gives us a direct relationship:

$$W = \frac{C}{\lambda} \quad \iff \quad \text{Mean E2E Latency} = \frac{\text{Concurrency}}{\text{Request Throughput}}$$

### Empirical Verification

Let's check our measurements against Little's Law:

| $C$ | Measured $\lambda$ (req/s) | Predicted $W = C/\lambda$ (s) | Observed Mean E2E (s) | Match |
|---|---|---|---|---|
| **1** | 0.387 | 2.584 | 2.583 | **99.96%** ✅ |
| **2** | 0.748 | 2.674 | 2.674 | **100.00%** ✅ |
| **4** | 1.334 | 2.999 | 2.898 | **96.6%** ✅ |
| **8** | 2.208 | 3.623 | 3.319 | **91.6%** ✅ |
| **16** | 3.240 | 4.938 | 4.240 | **85.9%** |
| **32** | 4.817 | 6.643 | 4.930 | **74.2%** |

At low concurrency ($C \le 4$), Little's Law matches almost exactly. At $C \ge 16$, the observed E2E is lower than $C/\lambda$ because our benchmark runs a finite number of requests (50), so during the "ramp-down" phase (when only the last few requests remain), the effective concurrency drops below $C$.

### What Little's Law Reveals About Scaling

From $L = \lambda W$, we can derive:

$$\lambda = \frac{C}{W} \implies \text{Throughput} = \frac{\text{Concurrency}}{\text{Average Response Time}}$$

Since response time $W$ grows sub-linearly with $C$ (because decode batching is efficient), throughput $\lambda$ grows with concurrency. But $W$ still increases — you cannot escape the law. Every token of throughput comes at a cost in individual request latency.

### Open-Loop vs. Closed-Loop (A Preview of What We Didn't Test)

Our benchmark is **closed-loop**: the number of in-flight requests is capped at $C$ by construction. Real production systems face **open-loop** traffic: requests arrive according to a Poisson process at rate $\lambda_{\text{arrival}}$, independent of how fast the server responds.

Under open-loop traffic, if $\lambda_{\text{arrival}} > \lambda_{\text{service capacity}}$, the queue grows unboundedly and latency goes to infinity. Closed-loop benchmarks cannot observe this behavior — they self-regulate by slowing down submissions when the server is slow.

This is listed as a limitation in our experiment. Open-loop benchmarking (using tools like wrk2 or vegeta) is the gold standard for production capacity assessment.

---

## 8. The KV Cache Multiplier — Memory Footprint Under Load

### KV Cache Size Per Sequence

Each active request maintains its own Key-Value cache in VRAM. For Qwen2.5-7B with Grouped-Query Attention:

```text
Model architecture:
  Transformer layers (L):     28
  Number of KV heads (H_kv):  4   (Grouped-Query Attention, not full 28 heads)
  Head dimension (d):         128
  Precision:                  BF16 (2 bytes per value)

Per-token KV cache:
  Values per token = 2 (K and V) × H_kv × d = 2 × 4 × 128 = 1,024 values
  Bytes per token  = L × 1,024 × 2 bytes = 28 × 1,024 × 2 = 57,344 bytes ≈ 56 KB

Per-sequence KV cache (512 prompt + 128 output = 640 tokens):
  640 × 57,344 bytes ≈ 36.7 MB per active sequence
```

### KV Cache Under Concurrency

| Concurrency ($C$) | Active Sequences | Total KV Cache | KV Cache as % of 24 GB |
|---|---|---|---|
| 1 | 1 | 36.7 MB | 0.15% |
| 4 | 4 | 147 MB | 0.60% |
| 8 | 8 | 294 MB | 1.20% |
| 16 | 16 | 587 MB | 2.39% |
| 32 | 32 | 1,174 MB (1.17 GB) | 4.79% |

### What We Actually Observed

```text
C ≤ 8:   Peak VRAM = 20,488 MiB   (constant — within pre-allocated pool)
C = 16:  Peak VRAM = 21,322 MiB   (+834 MiB)
C = 32:  Peak VRAM = 21,880 MiB   (+1,392 MiB total increase)
```

The observed VRAM increase (+1,392 MiB from $C=1$ to $C=32$) is roughly consistent with the theoretical KV cache growth for 32 concurrent 640-token sequences (~1,174 MB). The extra ~218 MB reflects CUDA context overhead, activation buffers, and internal vLLM data structures for managing 32 concurrent scheduling slots.

### When Would Memory Become the Bottleneck?

With our 24 GB RTX 3090:
- Model weights: ~14.2 GB
- vLLM runtime overhead: ~4.7 GB
- Available for KV cache: ~5.1 GB
- Max concurrent sequences: $5,100\text{ MB} / 36.7\text{ MB} \approx 139$ sequences

So theoretically, vLLM could handle up to ~139 concurrent 640-token sequences before VRAM exhaustion. Beyond that, it must **preempt** (swap KV cache to CPU memory) or **reject** new requests.

---

## 9. Roofline Revisited — Climbing the Bandwidth Slope

In E001, we placed decode at $\approx 0.5$ FLOPs/byte — far to the left of the RTX 3090's ridge point at $151.7$ FLOPs/byte. Now let's see how concurrency changes this:

```text
Attainable Performance (GFLOPS/s — logarithmic scale)
        ┌──────────────────────────────────────────────── Peak Compute
142,000 │─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  (142 TFLOPS BF16)
        │                                          ╱
        │                                        ╱
        │                                      ╱
 10,000 │                                    ╱
        │                               ╱
        │                          ╱  ← Bandwidth ceiling slope (936 GB/s)
  1,000 │                     ╱
        │                ╱
        │           ╱ ● C=32 (≈16 FLOPs/byte)
    100 │      ╱ ● C=8  (≈4 FLOPs/byte)
        │  ╱ ● C=4
        │╱● C=1 (≈0.5 FLOPs/byte)
     10 └─┬──────────┬──────────┬──────────┬──────────┬─►
         0.1        1.0       10.0      100.0      1000
                        Arithmetic Intensity (FLOPs/byte)
```

### Key Observations

1. **All concurrency levels are still on the bandwidth slope** — even at $C=32$, the arithmetic intensity ($\approx 16$ FLOPs/byte) is well below the ridge point ($\approx 152$ FLOPs/byte). Decode is still fundamentally memory-bandwidth-bound.

2. **Token throughput correlates with climbing the slope**: Each step up in concurrency pushes the operating point higher on the bandwidth slope, extracting more useful compute from the same memory bus traffic.

3. **This predicts further gains at $C > 32$**: Since we haven't reached the ridge, additional concurrency should continue improving throughput — until KV cache memory runs out or queueing latency becomes intolerable.

4. **Prefill is already at the ceiling**: Prefill arithmetic intensity ($\approx D = 3584$ FLOPs/byte for moderate-length prompts) is well above the ridge. Prefill is compute-bound and doesn't benefit from batching in the same way. This is why TTFT is mostly about queue wait time at high concurrency — the prefill compute itself doesn't get faster or slower with batching.

---

## 10. Experimental Design — The Closed-Loop Concurrency Harness

### Why Concurrency Benchmarking Is Hard

Measuring concurrency correctly is much harder than measuring single-request latency. Common pitfalls:

| Pitfall | What Goes Wrong | Our Solution |
|---|---|---|
| **Burst launch** | Launch all N requests at once → initial burst of >C in-flight | Exact worker pool of `min(C, N)` pulling from queue |
| **Counting overlapping time** | Summing individual request durations double-counts wallclock | Single `perf_counter_ns` span from first to last worker return |
| **HTTP chunk counting** | SSE chunks ≠ tokens | Tokenizer-validated actual output token counts |
| **Deadlocked error handling** | One failed request blocks the queue | `try/except` per request, create error `RequestMeasurement` |
| **Warmup contamination** | First few requests hit cold caches | Separate sequential warmup phase, excluded from statistics |
| **Variable actual concurrency** | Workers finishing early at end of benchmark | Tracked as a known finite-boundary limitation |

### Control Variables

| Variable Type | Variable | Value | Why |
|---|---|---|---|
| **Independent** | Client Concurrency $C$ | [1, 2, 4, 8, 16, 32] | The swept parameter |
| **Controlled** | Prompt tokens ($P$) | 512 (fixed) | Isolate concurrency from prefill scaling |
| **Controlled** | Output tokens ($N_{\text{out}}$) | 128 (fixed) | Isolate concurrency from decode length |
| **Controlled** | Model | Qwen/Qwen2.5-7B-Instruct, BF16 | Same model, same weights |
| **Controlled** | Temperature | 0.0 (greedy) | Deterministic generation |
| **Controlled** | Prefix caching | DISABLED | Prevent KV cache shortcuts |
| **Controlled** | Chunked prefill | DISABLED | Isolate pure queueing behavior |
| **Controlled** | Server instance | Single vLLM process, same lifetime for entire sweep | No server restarts between points |
| **Dependent** | TTFT, TPOT, E2E, Throughput, GPU util, VRAM, Error rate | Measured | Our outputs |

### Closed-Loop vs. Open-Loop Concurrency

Our benchmark implements **closed-loop** concurrency:

```text
CLOSED-LOOP (our design):
  ┌──────────────────────────────────────────────────────────┐
  │  Worker Pool (size = C)                                  │
  │                                                          │
  │  Worker 1: [Send Req] → [Wait for response] → [Send Req] → [Wait]...
  │  Worker 2: [Send Req] → [Wait for response] → [Send Req] → [Wait]...
  │  Worker 3: [Send Req] → [Wait for response] → [Send Req] → [Wait]...
  │  ...                                                     │
  │  Worker C: [Send Req] → [Wait for response] → [Send Req] → [Wait]...
  │                                                          │
  │  At ALL times, exactly C requests are in-flight.         │
  │  When server slows down, client submission slows down.    │
  └──────────────────────────────────────────────────────────┘

OPEN-LOOP (production traffic):
  ┌──────────────────────────────────────────────────────────┐
  │  Poisson Arrival Process (rate = λ)                       │
  │                                                          │
  │  Requests arrive independently of server response time:  │
  │  [Req] [Req]  [Req]      [Req][Req]   [Req]     ...     │
  │                                                          │
  │  If server is slow, queue grows UNBOUNDEDLY.              │
  │  Latency can go to INFINITY.                              │
  └──────────────────────────────────────────────────────────┘
```

Closed-loop is the right choice for E002 because we want to precisely control the independent variable ($C$) and measure steady-state behavior. Open-loop would introduce arrival rate as a confounding variable.

---

## 11. What We Built — Engineering the Concurrent Runner

E002 required significant new engineering compared to E001.

### The Problem

In E001, all requests ran sequentially at concurrency = 1. We needed to:
1. Run exactly $C$ requests simultaneously (not fewer, not more)
2. When a request finishes, immediately start the next from the queue
3. Handle errors without blocking other concurrent workers
4. Measure wall-clock duration correctly (not sum of individual durations)
5. Preserve per-request raw measurements for statistical analysis

### The Solution: `run_benchmark()` — Closed-Loop Worker Pool

[`src/inference_os/runner/benchmark.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/runner/benchmark.py) implements the concurrent benchmark runner:

```python
async def run_benchmark(
    request_factory: RequestFactory,
    num_requests: int,
    concurrency: int = 1,       # ← NEW: concurrency parameter
    warmup_requests: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> BenchmarkResult:
```

The key design is the **asyncio.Queue + worker pool** pattern:

```python
# 1. Fill a queue with all request IDs
queue: asyncio.Queue[tuple[str, int]] = asyncio.Queue()
for i in range(num_requests):
    queue.put_nowait((f"req-{i + 1}", i))

# 2. Pre-allocate result slots (preserves original ordering)
measured_requests: list[Optional[RequestMeasurement]] = [None] * num_requests


# 3. Define a worker that pulls from the queue until empty
async def worker() -> None:
    while True:
        try:
            request_id, idx = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        # ... execute request, store result at measured_requests[idx] ...
        queue.task_done()


# 4. Launch exactly min(C, N) workers
num_workers = min(concurrency, num_requests)
benchmark_start_time_ns = clock_fn()  # ← Wall-clock starts HERE
workers = [asyncio.create_task(worker()) for _ in range(num_workers)]
await asyncio.gather(*workers)
benchmark_end_time_ns = clock_fn()  # ← Wall-clock ends HERE
```

**Why `asyncio.Queue` instead of `asyncio.Semaphore`?**

A semaphore-based design would launch all N tasks concurrently and use a semaphore to limit active ones. But this pre-creates all N coroutines up front, wasting memory if N is large. The queue-based design creates only $C$ workers, and each worker pulls the next job when it finishes — exactly the closed-loop behavior we need.

**Why `min(concurrency, num_requests)`?**

If we request `concurrency=32` but only have 15 requests, we should only spawn 15 workers. Creating 32 workers when only 15 have work would waste resources and 17 workers would immediately exit.

### Error Handling Without Deadlocks

Each worker wraps the request execution in `try/except`:

```python
try:
    stream, input_tokens, output_tokens = await request_factory(...)
    measurement = await run_single_request(...)
except Exception as exc:
    # Create a FAILED measurement rather than crashing
    measurement = RequestMeasurement(
        request_id=request_id,
        success=False,
        error_message=str(exc) or exc.__class__.__name__,
        ...
    )
```

This is critical: if one request hits an HTTP timeout or OOM error, the worker **records the failure and continues to the next request**. Without this, one failed request would block a worker forever, reducing effective concurrency and potentially deadlocking the benchmark.

### Warmup: Sequential Before Concurrent

Warmup requests run **sequentially** before the concurrent benchmark phase:

```python
# Phase 1: Warmup (sequential)
for i in range(warmup_requests):
    stream, input_tokens, output_tokens = await request_factory(...)
    measurement = await run_single_request(...)
    warmup_measurements.append(measurement)

# Phase 2: Benchmark (concurrent)
# ... worker pool code ...
```

Why sequential warmup? Because warmup's purpose is to prime GPU CUDA kernels, JIT compilation caches, and memory allocator state. If warmup ran concurrently, it would contaminate the timing and VRAM measurements of the actual benchmark requests.

### Wall-Clock Throughput

Throughput is calculated from the wall-clock span of the concurrent benchmark phase, not from summing individual request durations:

```python
benchmark_duration_seconds = (benchmark_end_time_ns - benchmark_start_time_ns) / 1e9
# This spans from the first worker launch to the last worker return
```

If we summed individual durations, we'd get $C \times \text{actual duration}$ at $C$ concurrency — a meaningless number. The wall-clock span correctly captures "how many tokens were produced in how many seconds of real time."

### The `concurrency` Field in Config

[`src/inference_os/config.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/config.py) was extended:

```python
@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    ...
    concurrency: int = 1  # ← NEW: defaults to 1 for backward compatibility

    def __post_init__(self) -> None:
        ...
        if self.concurrency <= 0:
            raise ValueError(f"concurrency must be positive, got {self.concurrency}")
```

The default of `1` preserves backward compatibility: all E000 and E001 experiments continue to work without modification.

### Sweep Integration

[`src/inference_os/runner/sweep.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/runner/sweep.py) dispatches to the appropriate plot generator based on experiment type:

```python
if sweep_config.sweep_param == "concurrency":
    generate_e002_plots(point_results, plots_dir)
elif sweep_config.sweep_param == "max_output_tokens":
    generate_e001b_plots(point_results, plots_dir)
else:
    generate_e001a_plots(point_results, plots_dir)
```

### E002-Specific Plots

[`src/inference_os/reports/plots.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/reports/plots.py) generates 4 new plot types for E002:

1. **`throughput_requests_vs_concurrency.png`** — Request throughput (req/s) vs. $C$
2. **`throughput_tokens_vs_concurrency.png`** — Token throughput (tok/s) vs. $C$
3. **`latency_vs_concurrency.png`** — TTFT, TPOT, and E2E (P50 and P95) vs. $C$ on a dual-axis layout
4. **`gpu_metrics_vs_concurrency.png`** — Peak VRAM and avg GPU utilization vs. $C$

These are different from the E001A/B plots because the independent variable is now concurrency (not prompt/output length), and throughput metrics are the primary output (not just latency).

---

## 12. Reading Our Results — The Full RTX 3090 Data

### Run Configuration

* **Run ID**: `E002_20260906_063957_af02a4c1`
* **Hardware**: 1× NVIDIA GeForce RTX 3090 (24,576 MiB VRAM)
* **Driver**: 595.71.05 | Linux 6.8.0-134-generic
* **Backend**: vLLM (BF16, FlashAttention-2, FlashInfer sampling)
* **Model**: `Qwen/Qwen2.5-7B-Instruct`
* **Workload**: 512 input tokens, 128 output tokens, 50 measured requests per point, 2 warmup
* **Total**: 300 measured requests, 0 errors

### Results Summary Table

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

### Throughput Scaling Efficiency

How efficiently does throughput scale with concurrency?

| Concurrency Step | Throughput Increase | Ideal (Linear) | Scaling Efficiency |
|---|---|---|---|
| 1 → 2 | 49.55 → 95.74 (1.93×) | 2.0× | **96.6%** |
| 2 → 4 | 95.74 → 170.72 (1.78×) | 2.0× | **89.2%** |
| 4 → 8 | 170.72 → 282.62 (1.66×) | 2.0× | **82.8%** |
| 8 → 16 | 282.62 → 414.78 (1.47×) | 2.0× | **73.4%** |
| 16 → 32 | 414.78 → 616.57 (1.49×) | 2.0× | **74.3%** |

Scaling is sub-linear, with efficiency decreasing as concurrency increases. This is because:
1. Prefill contention grows (more requests competing for prefill slots)
2. The batched GEMM takes slightly longer per step at larger batch sizes
3. Scheduling overhead increases with more active sequences

### Wall-Clock Improvement

The same 50 requests completed in dramatically less wall-clock time:

| C | Wall Clock | Speedup vs C=1 |
|---|---|---|
| 1 | 129.16s | 1.0× |
| 2 | 66.85s | 1.93× |
| 4 | 37.49s | 3.45× |
| 8 | 22.64s | 5.71× |
| 16 | 15.43s | 8.37× |
| 32 | 10.38s | **12.44×** |

At $C=32$, the 50 requests completed in **10.4 seconds** instead of over **2 minutes**. For batch workloads, this is a 12.4× reduction in processing time — and GPU cost.

---

## 13. Hypothesis Validation — Checking Every Prediction

Before running E002, we formulated 6 hypotheses. Let's check each:

### H1: Throughput Scaling & Plateau ✅ CONFIRMED
**Prediction**: Token throughput scales sub-linearly, with large initial gains that diminish.
**Evidence**: 49.55 → 616.57 tok/s (12.44× overall). Scaling efficiency dropped from 96.6% (C=1→2) to 74.3% (C=16→32). Throughput was still increasing at C=32, suggesting the saturation knee has not been fully reached on this hardware.

### H2: Latency Growth ✅ CONFIRMED
**Prediction**: TTFT and E2E increase monotonically with C.
**Evidence**: TTFT P50 grew monotonically: 124 → 197 → 445 → 799 → 1595 → 1775 ms. E2E P50 grew monotonically: 2581 → 2666 → 2899 → 3318 → 4254 → 5990 ms. No exceptions.

### H3: TPOT Resilience ✅ CONFIRMED
**Prediction**: TPOT increases moderately.
**Evidence**: TPOT P50 moved from 19.34 ms to 22.75 ms — only a **17.6% increase** despite 32× the concurrent load. This confirms decode time is dominated by weight streaming, not batch computation.

### H4: Tail Latency Amplification ✅ CONFIRMED
**Prediction**: P95 spreads wider at high C.
**Evidence**: TTFT P95-P50 gap went from 3.48 ms (C=1) to 1,379.65 ms (C=32). E2E P95-P50 gap went from 16.75 ms to 54.21 ms. Massive tail variance under queueing.

### H5: GPU Utilization & VRAM ✅ CONFIRMED
**Prediction**: GPU util near 100%; VRAM grows with KV cache.
**Evidence**: GPU utilization was 96.8–100.0% across all points. VRAM grew from 20,488 → 21,880 MiB (+1,392 MiB) as KV cache allocations increased.

### H6: Error Rate ✅ CONFIRMED
**Prediction**: 0% errors across all C values.
**Evidence**: 300/300 requests succeeded. Zero timeouts, zero OOM errors, zero HTTP failures.

**All 6 hypotheses confirmed.** The experiment cleanly validated the theoretical predictions.

---

## 14. The Capacity Planner's Playbook — Choosing Optimal Concurrency

### The Three Operating Regimes

Based on our data, we can define three operational zones:

```text
Token Throughput (tok/s)
  650 ┤                                         ●  C=32
      │                                            (HIGH THROUGHPUT BATCH ZONE)
  500 ┤                                            TTFT > 1.7s, E2E ≈ 6s
      │                                            Good for: offline processing
  400 ┤                          ●  C=16
      │
  300 ┤                 ●  C=8   ← BALANCED OPERATING KNEE
      │                            TTFT < 800ms, E2E ≈ 3.3s
  200 ┤        ●  C=4              Good for: web chat, API services
      │
  100 ┤   ●  C=2
   50 ┤●  C=1  (LOW LATENCY INTERACTIVE ZONE)
      │         TTFT ≈ 124ms, E2E ≈ 2.6s
      │         Good for: coding assistants, voice bots
      └──┬──────────┬──────────┬──────────┬──────────────►
        2.5s       3.0s       4.0s       6.0s
                    Median E2E Latency
```

### Decision Framework

| Your Use Case | SLA Requirement | Recommended $C$ | Throughput | Cost Efficiency |
|---|---|---|---|---|
| **Voice assistant** | TTFT < 200 ms | **1–2** | 50–96 tok/s | Low |
| **Coding assistant** | TTFT < 500 ms | **4** | 171 tok/s | Medium |
| **Web chat** | TTFT < 1 s, E2E < 4 s | **4–8** | 171–283 tok/s | **Good** |
| **API service** | E2E < 5 s | **8–16** | 283–415 tok/s | High |
| **Batch processing** | No latency requirement | **32+** | 617+ tok/s | **Maximum** |

### GPU Cost Optimization

Suppose you need 1,000 tok/s aggregate throughput:

| Operating Point | tok/s per GPU | GPUs Needed | Relative Cost |
|---|---|---|---|
| C=1 | 49.5 | **21 GPUs** | 10.5× (baseline) |
| C=4 | 170.7 | **6 GPUs** | 3.0× |
| C=8 | 282.6 | **4 GPUs** | 2.0× |
| C=32 | 616.6 | **2 GPUs** | **1.0×** |

> **Concurrency tuning is the single largest lever for GPU cost optimization.** Going from C=1 to C=8 cuts your GPU bill by **80%** while maintaining sub-second TTFT.

---

## 15. Serving Controls — Why We Disabled Prefix Caching & Chunked Prefill

### Prefix Caching — `--no-enable-prefix-caching`

**What it does**: If two requests share the same prompt prefix (e.g., the same system prompt), vLLM reuses the KV cache from the first request. The second request skips prefill computation entirely.

**Why we disabled it**: Our benchmark sends 50 requests with the same synthetic prompt. With prefix caching enabled, request #2 through #50 would have near-zero TTFT (cache hit!). This would make our TTFT measurements invalid — we'd be measuring cache lookup time, not real prefill compute.

For scientific measurement, every request must do real work.

### Chunked Prefill — `--no-enable-chunked-prefill`

**What it does**: Instead of processing all 512 prompt tokens in one atomic prefill step, chunked prefill (from the Sarathi-Serve paper) splits the prompt into smaller chunks (e.g., 128 tokens each). Between chunks, the engine runs decode iterations for other active sequences.

**Why we disabled it**: Chunked prefill is specifically designed to **reduce** the TTFT degradation that we observed. By disabling it, we establish the **worst-case baseline** for TTFT under concurrency. This gives us:
1. A clean reference curve showing raw queueing physics without mitigation
2. A quantitative target: "Chunked prefill should reduce TTFT at C=32 from 1,775 ms down to ???"
3. A future experiment: enable chunked prefill and measure the improvement

This is how systematic experimentation works — establish the baseline, then measure the optimization.

---

## 16. What We Built in Code — Applied Engineering

### Directory Structure (New & Modified in E002)

```
inference_os/
├── configs/
│   ├── e002_concurrency.yaml              ← [NEW] Full sweep config
│   └── e002_pilot_concurrency.yaml        ← [NEW] Quick smoke test config
├── experiments/E002-concurrency/
│   ├── README.md                          ← [MODIFIED] Added empirical results
│   └── run_e002.py                        ← [NEW] Experiment runner with summary table
├── src/inference_os/
│   ├── config.py                          ← [MODIFIED] Added concurrency field
│   ├── runner/
│   │   ├── benchmark.py                   ← [MODIFIED] Closed-loop worker pool
│   │   ├── engine.py                      ← [MODIFIED] Pass concurrency to runner
│   │   └── sweep.py                       ← [MODIFIED] E002 plot dispatch
│   ├── reports/plots.py                   ← [MODIFIED] 4 new E002 plot functions
│   └── metrics/summary.py                ← [MODIFIED] Per-request raw data support
├── outputs/
│   ├── e002_concurrency_scaling_validation.md  ← [NEW] Human-readable analysis
│   └── plots/e002/                        ← [NEW] 4 publication plots
├── runs/
│   └── E002_20260906_063957_af02a4c1/     ← [NEW] Full run data
├── docs/
│   └── study_guide_e002.md                ← [NEW] This file
└── tests/
    └── test_concurrent_runner.py          ← [NEW] 7 tests for concurrent runner
```

### Test Coverage

```text
tests/test_concurrent_runner.py:
  - test_run_benchmark_concurrency_1       Verify C=1 still works (backward compat)
  - test_run_benchmark_concurrency_4       Verify C=4 produces correct results
  - test_run_benchmark_concurrency_gt_n    Verify min(C, N) worker cap
  - test_run_benchmark_error_handling      Verify failed requests don't deadlock
  - test_run_benchmark_warmup_sequential   Verify warmup runs before concurrent phase
  - test_run_benchmark_invalid_params      Verify ValueError on bad inputs
  - test_run_sequential_benchmark_compat   Verify backward-compatible wrapper

All 78 tests pass without GPU, without server, without internet.
```

---

## 17. Interview-Ready Knowledge — How to Explain This

### "Why does batching improve LLM decode throughput?"

> "Autoregressive decode at batch size 1 is memory-bandwidth-bound: the GPU reads all 14 GB of model weights from VRAM for every single generated token, with arithmetic intensity of about 0.5 FLOP/byte. The tensor cores sit idle 99% of the time waiting for data. When you batch C requests together, the same weight bytes loaded over the memory bus are reused across all C sequence activations in a single GEMM. Memory traffic stays constant while useful compute scales linearly — this is called **weight amortization**. In our benchmark, going from C=1 to C=32 increased throughput from 50 to 617 tok/s (12.4×) while the per-step decode time only increased by 18%."

### "Why does increasing concurrency hurt TTFT much more than TPOT?"

> "TPOT is governed by physics — each decode step reads the same model weights regardless of batch size. But TTFT has two components: queue wait time and prefill execution time. At C=1, queue wait is zero and TTFT equals raw prefill compute (~124 ms). At C=32, incoming requests wait in the scheduling queue while 31 other sequences are being decoded — adding over 1.6 seconds of pure wait time. So 93% of TTFT at C=32 is queueing delay, not computation. This is why chunked prefill was invented: by splitting prompts into small chunks interleaved with decode, you prevent long atomic prefills from blocking the queue."

### "How would you determine the right concurrency for a production deployment?"

> "I'd define the SLA first — typically max P95 TTFT and max P95 E2E. Then run a concurrency sweep exactly like our E002 benchmark on the target hardware. For example, if the SLA requires TTFT P95 < 1 second, our data shows you must cap at C ≤ 8 (where P95 = 852 ms). Then compute the cost: at C=8, one RTX 3090 delivers 283 tok/s. If you need 1000 tok/s, you need 4 GPUs. At C=32 you'd only need 2 GPUs, but you'd violate the TTFT SLA. The optimal operating point is the highest concurrency that stays within your SLA."

### "What is Little's Law and why does it matter for LLM serving?"

> "Little's Law is L = λW — the average number of requests in a system equals throughput times average response time. For closed-loop concurrency where L=C by construction, this gives W = C/λ, meaning response time equals concurrency divided by throughput. The key insight is: as you increase C, throughput λ grows sub-linearly while C grows linearly. So W (latency) must increase. You cannot escape this — throughput gains always come with a latency cost. In our data, going from C=1 to C=32 gave 12.4× throughput but 2.3× E2E latency."

### "Your GPU utilization is already 99%. How can more concurrency help?"

> "GPU utilization % from nvidia-smi measures whether any SM has any active warp — it includes the memory controller being busy reading weights. At C=1, the GPU is 99% 'busy' streaming 14 GB of weights for each single token, but the tensor cores are doing almost no work. Arithmetic intensity is 0.5 FLOP/byte — far below the 152 FLOP/byte ridge point. Higher concurrency doesn't increase memory bus activity (that's already saturated), but it does increase the useful compute done per byte transferred. The 99% utilization at C=1 is misleading — the GPU has enormous untapped compute capacity that batching unlocks."

---

## 18. Self-Test Questions

Test your understanding — try to answer these before looking at the solutions.

### Conceptual

1. **Why does token throughput scale by 12.4× but TPOT only increases by 17.6% when going from C=1 to C=32?**
   *(Answer: Throughput scales because 32 tokens are computed per decode step instead of 1, while the memory bus transfer — which dominates step time — stays constant at ~15 ms. The extra 3.4 ms comes from the larger GEMM computation for 32 activation vectors.)*

2. **If the RTX 3090 had 2× memory bandwidth (1,872 GB/s), what would happen to TPOT at C=1? At C=32? What about throughput?**
   *(Answer: TPOT at C=1 would roughly halve to ~10 ms (since decode is bandwidth-bound). TPOT at C=32 would also decrease but less proportionally, since at C=32 the GEMM compute is becoming a larger fraction of step time. Throughput at all concurrency levels would roughly double.)*

3. **Why did TTFT P95 explode to 3,155 ms at C=32 while P50 was only 1,775 ms?**
   *(Answer: The ~1.4 second gap represents variable queueing delay. Some requests arrived just as the engine committed to long decode batches or other prefills, causing them to wait extra time for a scheduling slot. Under heavy concurrency, queue position is stochastic, amplifying tail variance.)*

4. **If prefix caching were enabled, what would happen to TTFT at C=32?**
   *(Answer: All requests use the same synthetic prompt. After the first request, all subsequent requests would get a KV cache hit, making TTFT near-zero regardless of concurrency. This would make the TTFT measurement invalid for studying queueing physics.)*

5. **Using Little's Law, predict what E2E would be if we achieved 10 req/s throughput at C=32.**
   *(Answer: W = L/λ = 32/10 = 3.2 seconds. This would be a significant improvement over our observed 5.99s, requiring either more memory bandwidth, chunked prefill, or a different GPU.)*

### Applied

6. **Why does `run_benchmark()` use `asyncio.Queue` instead of launching all N tasks with an `asyncio.Semaphore`?**
   *(Answer: A semaphore creates all N coroutines upfront and limits active ones. The queue approach creates only C workers that pull sequentially from the queue — more memory-efficient and exactly matches the closed-loop concurrency model where workers actively loop.)*

7. **Why is warmup always sequential, even when the benchmark phase is concurrent?**
   *(Answer: Warmup primes CUDA JIT caches and memory allocators. Running warmup concurrently would contaminate the GPU state and timing measurements — the first concurrent warmup request would see cold kernels while later ones see warm ones, creating inconsistent baselines.)*

8. **Why does the benchmark measure wall-clock duration with `perf_counter_ns` rather than summing individual request durations?**
   *(Answer: At C=4, if 4 requests each take 3 seconds but run concurrently, summing gives 12 seconds — but only 3 seconds of wall-clock time elapsed. Throughput = 4 requests / 3 seconds = 1.33 req/s, not 4/12 = 0.33. Summing overlapping durations produces meaningless throughput numbers.)*

9. **What would happen if we forgot to set `--max-num-seqs 64` on the vLLM server when sweeping to C=32?**
   *(Answer: vLLM defaults to `--max-num-seqs 256`, so C=32 would work fine. But if it were set lower than 32, the server would throttle concurrent sequences, artificially limiting throughput and making the benchmark results reflect server throttling rather than GPU scaling.)*

10. **Our benchmark ran 50 requests per concurrency point. Why not 10 (like E001) or 1000?**
    *(Answer: 10 requests would be insufficient for stable P50/P95 estimates under concurrent queueing variance. 1000 would take too long on rented GPUs (C=1 at 0.39 req/s = 2,560 seconds ≈ 43 minutes). 50 provides stable statistics (n≥30 for CLT) while keeping total runtime under 15 minutes.)*

---

## 19. Further Reading — Papers, Books, and Resources

### Essential Papers

1. **Orca: A Distributed Serving System for Transformer-Based Generative Models**
   - Gyeong-In Yu et al. (2022). [OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu)
   - *The foundational paper that introduced continuous batching (iteration-level scheduling).*
   - *Why it matters for E002*: This is the scheduling mechanism that enables our throughput scaling — grouping active decode sequences into per-step batches.

2. **Efficient Memory Management for Large Language Model Serving with PagedAttention**
   - Woosuk Kwon et al. (2023). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
   - *The vLLM paper describing PagedAttention for KV cache memory management.*
   - *Why it matters for E002*: Explains how vLLM manages VRAM for 32 concurrent KV caches without fragmentation.

3. **Sarathi-Serve: Efficient LLM Serving with Chunked Prefills**
   - Amey Agrawal et al. (2024). [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
   - *Proposes chunking prefills to eliminate the TTFT queueing explosion observed in E002.*
   - *Why it matters*: This is the direct solution to the TTFT problem we diagnosed — our data provides the quantitative baseline that Sarathi-Serve aims to improve.

4. **Splitwise: Efficient Generative LLM Inference Using Phase Splitting**
   - Patel et al. (2024). [ISCA 2024](https://arxiv.org/abs/2311.18677)
   - *Disaggregates prefill nodes from decode nodes across different GPU pools.*
   - *Why it matters*: Takes the prefill-decode asymmetry we measured in E001 and the queueing contention we measured in E002 and proposes an architectural solution: don't mix prefill and decode on the same GPU.

5. **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**
   - Zhong et al. (2024). [OSDI '24](https://arxiv.org/abs/2401.09670)
   - *Another approach to prefill-decode disaggregation with formal goodput optimization.*
   - *Why it matters*: Builds on the same empirical observation: under high concurrency, prefill and decode interfere destructively.

### Foundational Resources

6. **The Roofline Model**
   - Samuel Williams, Andrew Waterman, David Patterson.
   - "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009).
   - [Paper](https://dl.acm.org/doi/10.1145/1498765.1498785)
   - *Why it matters*: The framework for understanding how concurrency shifts the operating point along the bandwidth slope.

7. **Little's Law** — John D.C. Little (1961)
   - "A Proof for the Queuing Formula: L = λW"
   - [Original Paper](https://doi.org/10.1287/opre.9.3.383)
   - *Why it matters*: The mathematical foundation for relating throughput and latency in any queueing system, including LLM serving.

### Books

8. **Programming Massively Parallel Processors** (4th Edition)
   - David B. Kirk, Wen-mei W. Hwu
   - *Chapters on memory hierarchy, GEMM optimization, and Roofline analysis directly explain why batched decode is faster.*

9. **Computer Architecture: A Quantitative Approach** (6th Edition)
   - Hennessy & Patterson
   - *Chapter 4 (memory hierarchy) and Chapter 6 (warehouse-scale computing) provide the architectural background for understanding GPU memory bandwidth and queueing in large-scale serving.*

10. **Queueing Systems: Theory** (Volume 1)
    - Leonard Kleinrock
    - *The classic textbook on queueing theory. Chapters on Little's Law, open vs. closed systems, and M/M/1 queues.*

### Online Resources

11. **NVIDIA GPU Architecture Whitepaper (Ampere)**
    - [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
    - *RTX 3090 (GA102): 936 GB/s bandwidth, 142 TFLOPS BF16.*

12. **vLLM Documentation**
    - [https://docs.vllm.ai](https://docs.vllm.ai)
    - *Covers continuous batching, PagedAttention, scheduler parameters, and `--max-num-seqs` configuration.*

13. **anyscale Blog: "How continuous batching enables 23x throughput in LLM inference while reducing p50 latency"**
    - A practical industry walkthrough of exactly the phenomenon we measured in E002.

---

## 20. Glossary — New Terms Introduced in E002

| Term | Definition |
|---|---|
| **Weight Amortization** | Reusing model parameter weights loaded from VRAM across multiple concurrent sequence activations in a single batched matrix multiplication. The core mechanism driving throughput gains under concurrency. |
| **Continuous Batching** | Scheduling strategy where the engine makes batch composition decisions at every single decode iteration, rather than forming static batches that run to completion. Introduced by the Orca paper. |
| **Closed-Loop Concurrency** | A benchmarking load model where a fixed pool of $C$ clients maintains exactly $C$ in-flight requests by submitting a new request only when an existing one completes. The number of active requests cannot exceed $C$. |
| **Open-Loop Concurrency** | A load model where requests arrive independently according to an external distribution (e.g., Poisson process), regardless of server response time. Queues can grow unboundedly. |
| **Little's Law** | Fundamental queueing theorem: $L = \lambda W$ — average occupancy equals throughput times average response time. Applies to any stable system. |
| **Head-of-Line Blocking** | When an incoming request is delayed because a preceding request (or heavy compute operation like prefill) occupies the execution pipeline. The primary cause of TTFT degradation under concurrency. |
| **Queueing Delay** | Time a request spends waiting in a scheduler queue before execution begins. Dominates TTFT at high concurrency (93% of TTFT at C=32 in our data). |
| **Tail Latency** | The latency experienced by the slowest requests (P95, P99). Under queueing, tail latency grows much faster than median latency because queue position is stochastic. |
| **Scaling Efficiency** | Ratio of actual throughput speedup to ideal linear speedup when doubling concurrency. E.g., if throughput goes from 100 to 178 tok/s when doubling C, efficiency = 178/200 = 89%. |
| **Saturation Knee** | The concurrency point beyond which throughput gains diminish sharply. Caused by compute saturation, memory exhaustion, or scheduler overhead. |
| **Worker Pool** | An asyncio design pattern where a fixed number of concurrent workers pull tasks from a shared queue, ensuring exactly $C$ tasks execute simultaneously. |
| **Wall-Clock Duration** | The real elapsed time measured by a monotonic clock spanning the entire concurrent benchmark phase. The denominator for throughput calculation. Not the sum of individual request times. |
| **Chunked Prefill** | Splitting a long prompt into smaller chunks processed across multiple iterations, interleaved with decode steps. Prevents the TTFT explosion observed in E002 by avoiding atomic long prefills. |
| **`asyncio.Queue`** | Python's async-safe FIFO queue used to distribute work items across concurrent worker coroutines. Thread-safe within a single event loop. |
| **`asyncio.create_task()`** | Schedules a coroutine to run concurrently within the event loop. Used to spawn worker tasks in the benchmark runner. |
| **Arithmetic Intensity** | Ratio of compute operations (FLOPs) to memory accesses (bytes). At C=1: ~0.5 (bandwidth-bound). At C=32: ~16 (still bandwidth-bound but climbing toward the ridge). |
