# E000 — Measurement Validation

## Primary Question

> Can the benchmark harness make trustworthy and reproducible measurements?

## Status: COMPLETED ✅

Validation was successfully executed on live hardware (**1× NVIDIA RTX 3090 24GB VRAM**) against **`Qwen/Qwen2.5-7B-Instruct`** (BF16) using vLLM 0.27.1.

---

## Key Experimental Results

| Metric | Measured Value | Analysis & Systems Interpretation |
| :--- | :--- | :--- |
| **Model** | `Qwen/Qwen2.5-7B-Instruct` | BF16 precision, FlashAttention-2, FlashInfer sampling |
| **Hardware** | 1× NVIDIA GeForce RTX 3090 | 24 GB VRAM, Driver 595.84, CUDA 13.2 |
| **Prompt Tokens** | 128 tokens | Generated deterministically with `HFTokenizer` & exact token verification |
| **Output Tokens** | 64 tokens | Fixed output limit per request |
| **Warm-up Requests** | 2 requests | Warmup timing separated and discarded from benchmark summaries |
| **Measured Requests** | 10 requests | Concurrency = 1, 100% success rate (10/10) |
| **TTFT (P50)** | **`39.63 ms`** | Stable prefill latency (down from 218 ms cold startup) |
| **TTFT (P95)** | **`64.52 ms`** | P95 latency variance is minimal (<25 ms spread) |
| **E2E Latency (P50)**| **`1282.15 ms`** | Total request turnaround time |
| **E2E Latency StdDev**| **`15.78 ms`** | **1.2% variance** across 10 repeated runs (highly reproducible) |
| **Token Throughput**| **`49.65 tok/s`** | Single-stream decode speed ($\approx 19.45\text{ ms/token}$) |
| **Request Throughput**| **`0.78 req/s`** | Sequential throughput at concurrency 1 |
| **Peak VRAM** | **`18,914 MiB`** | Model weights (~14.2 GB) + KV cache buffer (~3.6 GB) + activations |
| **Avg GPU Utilization**| **`94.8%`** | Sustained GPU execution |

---

## Answer to the Systems Question

> **How do you know that an LLM inference benchmark is measuring what you think it is measuring?**

1. **Token Accuracy**: We verify that prompt token counts are exact at the tokenizer level before sending to the server, preventing token count drift.
2. **Warm-up Isolation**: We separate cold compilation/allocation runs ($W=2$) from steady-state measurements ($N=10$), preventing TTFT skew ($218\text{ ms} \rightarrow 40\text{ ms}$).
3. **Low Variance ($<1.5\%$)**: Across 10 sequential runs with identical synthetic prompts, E2E latency varied by only $15.78\text{ ms}$ (standard deviation), proving high measurement stability.
4. **Physical Telemetry Verification**: Real-time GPU VRAM (18.9 GB) and compute % (94.8%) align with the theoretical footprint of a 7B parameter BF16 model with pre-allocated KV cache blocks.
5. **Lossless Persistence**: Every nanosecond timestamp, raw SSE chunk, and GPU sample is preserved on disk in `runs/` for post-hoc validation.
