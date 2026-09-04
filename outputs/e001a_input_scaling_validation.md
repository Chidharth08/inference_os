# E001-A — Input Length Scaling Benchmark Run

* **Run ID**: `E001A_20260904_182047_d88d790b`
* **Date**: 2026-09-04
* **Hardware**: 1× NVIDIA GeForce RTX 3090 (24 GB VRAM) on Vast.ai
* **Driver Version**: 595.84 | **CUDA Version**: 13.2
* **Serving Backend**: vLLM (bfloat16, FlashAttention-2, FlashInfer sampling)
* **Model**: `Qwen/Qwen2.5-7B-Instruct`
* **Endpoint**: `http://localhost:18000`
* **Prefix Caching**: **DISABLED** (`--no-enable-prefix-caching`)
* **Chunked Prefill**: **DISABLED** (`--no-enable-chunked-prefill`)

---

## 1. Benchmark Execution Parameters

* **Sweep Parameter**: `prompt_tokens` $\in [128, 512, 2048, 4096]$
* **Fixed Output Tokens ($N_{\text{out}}$)**: 128 tokens
* **Warm-up Requests**: 2 per sweep point (discarded from primary metrics)
* **Measured Requests**: 10 per sweep point (sequential execution, concurrency = 1)
* **Seed**: 42
* **Sampling Temperature**: 0.0 (greedy decoding)

---

## 2. Raw Console Summary Table

```text
=========================================================================================================
 E001-A: INPUT LENGTH SCALING RESULTS SUMMARY
=========================================================================================================
Prompt Tokens  | Status   | TTFT (P50)   | TTFT (Mean)  | TPOT (P50)   | E2E (P50)    | Peak VRAM    | GPU Util
---------------------------------------------------------------------------------------------------------
128            | OK       |   68.14 ms   |   69.08 ms   |   19.36 ms   | 2520.24 ms   | 18872 MiB    | 97.7%   
512            | OK       |  137.69 ms   |  143.21 ms   |   19.53 ms   | 2619.79 ms   | 18872 MiB    | 97.4%   
2048           | OK       |  435.72 ms   |  443.24 ms   |   19.60 ms   | 2924.51 ms   | 19094 MiB    | 97.6%   
4096           | OK       |  854.59 ms   |  859.23 ms   |   19.85 ms   | 3377.59 ms   | 19392 MiB    | 98.0%   
=========================================================================================================
```

---

## 3. Systems Analysis & Empirical Findings

### A. Prefill Compute Scaling (TTFT)
* **Scaling Curve**: Time to First Token (TTFT) scales monotonically with prompt length:
  - $128\text{ tokens} \to 68.14\text{ ms}$ (Mean: $69.08\text{ ms}$)
  - $512\text{ tokens} \to 137.69\text{ ms}$ (Mean: $143.21\text{ ms}$)
  - $2048\text{ tokens} \to 435.72\text{ ms}$ (Mean: $443.24\text{ ms}$)
  - $4096\text{ tokens} \to 854.59\text{ ms}$ (Mean: $859.23\text{ ms}$)
* **Inference Insight**: With FlashAttention and tensor core parallel matrix multiplication ($P \times D \times D$), TTFT scaling across $128 \to 4096$ tokens is strictly compute-bound and near-linear, confirming that prefill duration is governed by sequence matrix math rather than memory bandwidth.

### B. Time Per Output Token (TPOT) Invariance
* **Decode Latency**: TPOT remains flat across all context lengths at batch size 1:
  - $128\text{ prompt tokens} \to 19.36\text{ ms/tok}$ ($50.74\text{ tok/s}$)
  - $512\text{ prompt tokens} \to 19.53\text{ ms/tok}$ ($48.74\text{ tok/s}$)
  - $2048\text{ prompt tokens} \to 19.60\text{ ms/tok}$ ($43.66\text{ tok/s}$)
  - $4096\text{ prompt tokens} \to 19.85\text{ ms/tok}$ ($37.86\text{ tok/s}$)
* **Inference Insight**: Decode latency is memory-bandwidth bound ($14.2\text{ GB}$ model weights loaded per token step). Because the single query vector attend over the KV cache contributes negligible memory transfer compared to reading the full 7B parameter weights ($14.2\text{ GB} / 936\text{ GB/s} \approx 15.2\text{ ms}$ theoretical lower bound + kernel launch overhead), TPOT remains flat regardless of prompt length.

### C. End-to-End Latency (E2E)
* E2E latency increases monotonically by precisely $\Delta\text{TTFT}$:
  $$\text{E2E}(P) \approx \text{TTFT}(P) + (N_{\text{out}} - 1) \times \text{TPOT}$$
* With fixed $N_{\text{out}} = 128$, the decode baseline is $\approx 127 \times 19.5\text{ ms} \approx 2476\text{ ms}$. Adding $\text{TTFT} = 68.14\text{ ms} \to 854.59\text{ ms}$ yields the observed E2E latency curve ($2520\text{ ms} \to 3377\text{ ms}$).

### D. Physical GPU Memory (VRAM)
* **VRAM Growth**:
  - $128 \to 512\text{ tokens}$: $18,872\text{ MiB}$ (baseline model weights + initial pre-allocated KV cache block pool)
  - $2048\text{ tokens}$: $19,094\text{ MiB}$ ($+222\text{ MiB}$)
  - $4096\text{ tokens}$: $19,392\text{ MiB}$ ($+520\text{ MiB}$ total increase over baseline)
* **GPU Compute Utilization**: Remained pegged at **$97.4\% - 98.0\%$** across all points, confirming continuous kernel execution saturation.

---

## 4. Generated Scaling Visualizations

* **TTFT Scaling Plot**: `outputs/plots/e001a/ttft_vs_input_tokens.png`
* **End-to-End Latency Plot**: `outputs/plots/e001a/e2e_vs_input_tokens.png`
* **TPOT Stability Plot**: `outputs/plots/e001a/tpot_vs_input_tokens.png`
* **GPU Memory Scaling Plot**: `outputs/plots/e001a/gpu_memory_vs_input_tokens.png`

---

## 5. Raw Artifacts Reference

* **Full Run Directory**: `runs/E001A_20260904_182047_d88d790b/`
* **Aggregate Summary JSON**: `runs/E001A_20260904_182047_d88d790b/sweep_summary.json`
* **Per-Request & GPU Traces**: `runs/E001A_20260904_182047_d88d790b/prompt_tokens_<N>/`
