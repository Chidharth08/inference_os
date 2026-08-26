# E000 — Measurement Validation Benchmark Run

* **Run ID**: `E000_20260826_095535_e6683e48`
* **Date**: 2026-08-26
* **Hardware**: 1× NVIDIA GeForce RTX 3090 (24 GB VRAM) on Vast.ai
* **Driver Version**: 595.84 | **CUDA Version**: 13.2
* **Serving Backend**: vLLM 0.27.1 (bfloat16, FlashAttention-2, FlashInfer sampling)
* **Model**: `Qwen/Qwen2.5-7B-Instruct`
* **Endpoint**: `http://127.0.0.1:18000`

---

## 1. Benchmark Execution Parameters

* **Target Prompt Length**: 128 tokens
* **Max Output Tokens**: 64 tokens
* **Warm-up Requests**: 2 (discarded from primary summary metrics)
* **Measured Requests**: 10 (concurrency = 1)
* **Seed**: 42
* **Sampling Temperature**: 0.0 (greedy decoding)

---

## 2. Raw Console Output

```text
======================================================================
 inference_os — E000 Measurement Validation Benchmark
======================================================================
 Model:               Qwen/Qwen2.5-7B-Instruct
 Endpoint:            http://127.0.0.1:18000
 Target Prompt Tokens:128
 Max Output Tokens:   64
 Warm-up Requests:    2
 Measured Requests:   10
 Seed:                42
----------------------------------------------------------------------
Executing benchmark pipeline...
----------------------------------------------------------------------
 Benchmark Execution Completed Successfully!
======================================================================
 EXECUTION SUMMARY
   Total Measured:     10
   Successful:         10
   Failed:             0
   Total Input Tokens: 1280
   Total Output Tokens:640
   Benchmark Duration: 12.8904 s

 THROUGHPUT
   Request Throughput: 0.78 req/s
   Token Throughput:   49.65 tok/s

 TIME TO FIRST TOKEN (TTFT)
   Mean:      44.29 ms (0.0443 s)
   P50:       39.63 ms (0.0396 s)
   P90:       54.74 ms (0.0547 s)
   P95:       64.52 ms (0.0645 s)
   Min:       35.98 ms (0.0360 s)
   Max:       74.29 ms (0.0743 s)
   StdDev:    11.57 ms (0.0116 s)

 END-TO-END LATENCY (E2E)
   Mean:    1289.02 ms (1.2890 s)
   P50:     1282.15 ms (1.2822 s)
   P90:     1305.70 ms (1.3057 s)
   P95:     1317.54 ms (1.3175 s)
   Min:     1278.84 ms (1.2788 s)
   Max:     1329.38 ms (1.3294 s)
   StdDev:    15.78 ms (0.0158 s)

 GPU TELEMETRY
   VRAM Usage:         Peak: 18914 MiB / Avg: 18914.0 MiB (Total: 24576 MiB)
   GPU Compute:        Peak: 100% / Avg: 94.8%

 Results Saved To: /workspace/inference_os/runs/E000_20260826_095535_e6683e48
======================================================================
```

---

## 3. Systems Analysis & Validation Insights

### A. Warmed-Up Prefill Latency (TTFT)
* **P50 TTFT**: `39.63 ms` (Minimum: `35.98 ms`).
* **Comparison with Cold Start**: In our initial cold smoke test without warm-up, TTFT was **`218.72 ms`**. 
* **The Difference**: Running $W=2$ warmup requests allowed vLLM to compile CUDA graphs, initialize FlashInfer kernels, and allocate memory pages ahead of time, dropping TTFT by **~82%** into true physical prefill latency (~40 ms for 128 tokens).

### B. Decode Throughput & Latency
* **Token Throughput**: `49.65 tokens/sec`.
* **Time Per Output Token (TPOT)**:
  $$\text{TPOT} = \frac{\text{E2E} - \text{TTFT}}{\text{Output Tokens}} = \frac{1289.02\text{ ms} - 44.29\text{ ms}}{64} \approx 19.45\text{ ms/token}$$
* **Generation Uniformity**: Standard deviation across 10 sequential runs was only **`15.78 ms`** ($\approx 1.2\%$ variance), demonstrating outstanding measurement reproducibility.

### C. Physical GPU Telemetry
* **Peak VRAM Allocated**: `18,914 MiB` ($\approx 76.9\%$ of the 24,576 MiB total VRAM), matching model weights (~14.2 GB) + KV cache buffer allocation (~3.6 GB) + runtime activation overhead.
* **Compute Utilization**: `94.8% average`, confirming continuous GPU kernel saturation during sequential execution.
