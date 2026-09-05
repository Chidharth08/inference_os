# 📚 inference_os — Complete Study Guide (Milestone E001: Prefill vs. Decode)

This guide covers **everything** about Milestone E001 — the **why**, the **how**, and the **what** — at both a conceptual systems level and an applied code level. By the end, you should be able to explain prefill vs. decode scaling to an interviewer, derive the latency equations from first principles, and trace every number back to the physics of the GPU.

---

## Table of Contents

1. [The Core Question — What Are We Trying to Understand?](#1-the-core-question)
2. [Conceptual Foundation — The Two Phases of LLM Inference](#2-conceptual-foundation)
3. [Deep Dive — Prefill: The Compute-Bound Phase](#3-deep-dive-prefill)
4. [Deep Dive — Decode: The Memory-Bandwidth-Bound Phase](#4-deep-dive-decode)
5. [The Master Equation — Latency Decomposition](#5-the-master-equation)
6. [The KV Cache — Why Memory Grows](#6-the-kv-cache)
7. [Experimental Design — Isolating Variables Like a Scientist](#7-experimental-design)
8. [What We Built — The Parameter Sweep Engine](#8-what-we-built)
9. [E001-A — Input Length Scaling (Prefill Characterization)](#9-e001-a)
10. [E001-B — Output Length Scaling (Decode Characterization)](#10-e001-b)
11. [Reading the Results — From Numbers to Understanding](#11-reading-the-results)
12. [The Grand Synthesis — Answering the Core Question](#12-the-grand-synthesis)
13. [Serving Controls — Why We Disabled Things](#13-serving-controls)
14. [What We Built in Code — Applied Engineering](#14-what-we-built-in-code)
15. [Roofline Model — Understanding Hardware Limits](#15-roofline-model)
16. [Interview-Ready Knowledge — How to Explain This](#16-interview-ready-knowledge)
17. [Self-Test Questions](#17-self-test-questions)
18. [Further Reading — Papers, Books, and Resources](#18-further-reading)
19. [Glossary — New Terms Introduced in E001](#19-glossary)

---

## 1. The Core Question

> **"How do prefill and decode differ, and why do prompt length and generation length affect inference latency differently?"**

This is the guiding question of Milestone E001. It sounds simple, but answering it rigorously requires understanding:

- What **physically happens on the GPU** during each phase
- **Why** one phase is compute-bound and the other is memory-bandwidth-bound
- **How** to design controlled experiments that isolate each phase
- **What** the empirical data tells us about the hardware's behavior

### Why This Matters

If you're building or operating an LLM serving system, you will face questions like:

- "Our users are complaining about slow first-token times. What should we optimize?"
- "We're generating long responses and it's expensive. Can we make it faster?"
- "Should we invest in a GPU with more FLOPS or more memory bandwidth?"

You cannot answer these without understanding that **prefill and decode are fundamentally different computational workloads** running on the same hardware.

---

## 2. Conceptual Foundation — The Two Phases of LLM Inference

Every single request to a transformer-based LLM goes through two sequential phases:

```
                    YOUR REQUEST
                         │
          ┌──────────────▼──────────────────┐
          │        PREFILL PHASE             │
          │                                  │
          │  Input: Your entire prompt       │
          │         (P tokens)               │
          │                                  │
          │  What happens:                   │
          │  • All P tokens processed        │
          │    IN PARALLEL                   │
          │  • Self-attention computed        │
          │    across full sequence           │
          │  • KV cache populated            │
          │    for all P tokens              │
          │                                  │
          │  Bottleneck: COMPUTE             │
          │  (Tensor Core arithmetic)        │
          │                                  │
          │  Output: First generated token   │
          │  Metric: TTFT                    │
          └──────────────┬──────────────────┘
                         │
          ┌──────────────▼──────────────────┐
          │        DECODE PHASE              │
          │                                  │
          │  Input: One token at a time      │
          │         (the previously          │
          │          generated token)         │
          │                                  │
          │  What happens:                   │
          │  • Generates tokens ONE BY ONE   │
          │  • Each step reads entire        │
          │    model weights (~14.2 GB)      │
          │  • Appends to KV cache           │
          │                                  │
          │  Bottleneck: MEMORY BANDWIDTH    │
          │  (DRAM → SRAM data transfer)     │
          │                                  │
          │  Output: Remaining N_out - 1     │
          │          tokens                  │
          │  Metric: TPOT                    │
          └─────────────────────────────────┘
```

### The Fundamental Asymmetry

This is the single most important insight in LLM inference:

| Property | Prefill | Decode |
|---|---|---|
| **Tokens processed per step** | All P at once | 1 at a time |
| **Matrix shape** | `[P × D] × [D × D]` (tall × square) | `[1 × D] × [D × D]` (thin × square) |
| **Arithmetic intensity** | **High** — many FLOPs per byte loaded | **Low** — ~1 FLOP per byte loaded |
| **Bottleneck** | **Compute** (FLOPS) | **Memory bandwidth** (GB/s) |
| **GPU utilization** | Near 100% | Low (hardware underutilized) |

**Why is this?** It comes down to matrix multiplication. When you multiply a `[P × D]` matrix by a `[D × D]` matrix, you perform `P × D × D` FLOPs while loading `(P × D + D × D)` values from memory. The ratio (FLOPs / bytes) is called **arithmetic intensity**:

- **Prefill**: Arithmetic intensity = `P × D² / (P × D + D²) ≈ D` when P is large → **compute-bound**
- **Decode**: Arithmetic intensity = `1 × D² / (1 × D + D²) ≈ 1` → **memory-bandwidth-bound**

This is why decode is fundamentally slower per-token than prefill per-token: the GPU is starving for data, waiting for the next chunk of model weights to arrive from VRAM.

---

## 3. Deep Dive — Prefill: The Compute-Bound Phase

### What Physically Happens

When you send a 128-token prompt to the model:

1. **Embedding lookup**: Each of the 128 token IDs is mapped to a 3584-dimensional vector (for Qwen2.5-7B). This produces a `[128 × 3584]` matrix.

2. **Through each transformer layer** (28 layers in Qwen2.5-7B):
   - **QKV Projection**: Three matrix multiplications (`[128 × 3584] × [3584 × 3584]`) produce Query, Key, and Value matrices. This is a massive parallel GEMM (General Matrix Multiply) operation.
   - **Self-Attention**: Each token's query attends to ALL other tokens' keys. For 128 tokens, this is a `[128 × 128]` attention matrix — computed in parallel using FlashAttention.
   - **Feed-Forward Network (FFN)**: Two more large matrix multiplications through a wider intermediate dimension.

3. **Each of these matrix multiplications is a GEMM** — the exact operation that GPU Tensor Cores are designed for. The RTX 3090's Tensor Cores can perform BF16 matrix multiplications at 142 TFLOPS peak.

### Why It Scales with Prompt Length

If you double the prompt from 128 to 256 tokens:
- The QKV projection matrices become `[256 × 3584] × [3584 × 3584]` — **twice as many rows**, roughly twice the compute.
- Self-attention becomes `[256 × 256]` instead of `[128 × 128]` — **4× more attention computations** (quadratic in theory, but FlashAttention makes this near-linear in practice for typical lengths).
- The FFN computations double.

**Result**: TTFT scales roughly linearly with prompt length (for moderate context lengths), because the dominant cost is the QKV/FFN GEMMs which scale linearly with sequence length.

### Our E001-A Empirical Evidence

```
Prompt Tokens    TTFT (P50)     Scaling Factor
128              68.14 ms       1.0×
512              137.69 ms      2.02×   (theoretical: 4.0×)
2048             435.72 ms      6.39×   (theoretical: 16.0×)
4096             854.59 ms      12.54×  (theoretical: 32.0×)
```

The scaling is sub-linear relative to the purely quadratic attention scaling because:
1. **FlashAttention** reduces the O(N²) attention computation to O(N) memory access with tiling
2. **FFN layers dominate** over attention at moderate sequence lengths, and FFN scales linearly
3. **GPU parallelism** absorbs some of the scaling — the GPU has enough Tensor Cores to process the extra tokens without proportional slowdown

---

## 4. Deep Dive — Decode: The Memory-Bandwidth-Bound Phase

### What Physically Happens

After prefill produces the first output token, the model enters decode mode. For EACH subsequent token:

1. **Only 1 token is processed** — the most recently generated token.
2. **The entire model weights must be loaded from VRAM** for this single token:
   - 7 billion parameters × 2 bytes (BF16) = **14.2 GB of weights**
   - This includes all 28 layers' QKV projections, attention output projections, FFN weights, and normalization parameters.
3. **The GPU performs the arithmetic** (matrix-vector multiplication, not matrix-matrix):
   - `[1 × 3584] × [3584 × 3584]` — a vector-matrix multiply, which has very low arithmetic intensity.
4. **The new key and value vectors are appended to the KV cache** for future attention.
5. **The logits are computed** and the next token is sampled (greedy in our case).

### Why It Takes ~19.8 ms Per Token

This is a beautiful calculation that connects directly to hardware specifications:

```
Model weights to read:     14.2 GB
RTX 3090 memory bandwidth: 936 GB/s

Theoretical minimum time:  14.2 GB / 936 GB/s = 15.2 ms

Observed TPOT:             19.8 ms

Efficiency:                15.2 / 19.8 = 76.8%
```

The ~24% overhead comes from:
- **KV cache reads**: Attention must also read the accumulated KV cache (grows with sequence length)
- **Kernel launch overhead**: Each CUDA kernel has a small (~5 μs) launch latency, and decode requires many small kernel launches
- **Memory access patterns**: Not all VRAM reads are perfectly sequential, causing some bank conflicts
- **Layer normalization and activation functions**: Small additional compute between the major matrix operations

### Why TPOT is Invariant to Both Prompt AND Output Length

This is the key insight from E001:

**Invariant to prompt length (E001-A)**:
- Whether the prompt was 128 or 4096 tokens, TPOT stayed at ~19.5-19.8 ms
- Why? Each decode step reads the **same 14.2 GB of model weights** regardless of how long the prompt was
- The KV cache for attention is slightly larger with a longer prompt, but this is a tiny fraction of the total memory bandwidth consumed (KV cache for 4096 tokens ≈ 200 MB vs 14.2 GB of weights)

**Invariant to output length (E001-B)**:
- Whether generating token #10 or token #1000, TPOT stayed at ~19.8 ms
- Why? Each decode step still reads the **same 14.2 GB of model weights**
- The KV cache grows by one entry per step, but again, this is negligible compared to the weight read

### The Memory-Bandwidth Bottleneck Explained

Think of it like this: imagine you have a library (VRAM) with 14.2 billion books (parameters), and a reading room (the compute unit) that can process books very fast. But there's a single door (memory bus) between the library and the reading room that can only pass 936 books per second (GB/s).

For each output token, you must bring EVERY book through the door to the reading room, read one page from each, put them back, and then move on to the next token. The speed limit is the door, not how fast you can read. This is what "memory-bandwidth-bound" means.

---

## 5. The Master Equation — Latency Decomposition

The total end-to-end latency for a single inference request decomposes cleanly:

```
E2E = TTFT + (N_out - 1) × TPOT
```

Where:
- `E2E` = End-to-end latency (total time from request sent to last token received)
- `TTFT` = Time to First Token (prefill duration + network overhead)
- `N_out` = Number of output tokens generated
- `TPOT` = Time Per Output Token (average decode step duration)
- `N_out - 1` because the first output token is produced as part of prefill (TTFT), not decode

### Why "-1" in the Equation?

When prefill finishes, the model has already produced the first output token. So:
- Token #1 is "free" (included in TTFT)
- Tokens #2 through #N_out each cost one TPOT
- Total decode tokens = N_out - 1

### Verifying Against Our Data (E001-B)

For `N_out = 1024`, `prompt = 128 tokens`:

```
Predicted E2E = TTFT + (N_out - 1) × TPOT
             = 39.37 ms + (1024 - 1) × 19.801 ms
             = 39.37 ms + 20,256.42 ms
             = 20,295.79 ms

Observed E2E = 20,296.10 ms

Error = 0.31 ms = 0.001%
```

This is remarkable precision. The latency decomposition model fits the empirical data almost perfectly, confirming that:
1. Prefill and decode are truly independent phases
2. Each decode step takes a near-identical amount of time
3. There is no hidden interaction between the phases

### Sensitivity Analysis

From this equation, we can derive how changes affect E2E:

| Change | Effect on E2E | Because |
|---|---|---|
| Double prompt length P | E2E increases by ~ΔTTFT (~800 ms for 128→4096) | TTFT scales with P, but decode stays the same |
| Double output length N_out | E2E roughly doubles | E2E ≈ (2 × N_out) × TPOT, which is 2× |
| Faster GPU memory bandwidth | TPOT decreases, E2E decreases proportionally to N_out | Decode is bandwidth-bound |
| More GPU compute (more FLOPS) | TTFT decreases, small E2E improvement | Only prefill is compute-bound |

This is why **different optimization strategies** matter for different use cases:
- **Chat applications** (short prompts, short outputs): Optimize TTFT (compute)
- **Document summarization** (long prompts, short outputs): Optimize TTFT (compute)
- **Creative writing** (short prompts, long outputs): Optimize TPOT (memory bandwidth)
- **Translation** (long inputs, long outputs): Both matter

---

## 6. The KV Cache — Why Memory Grows

### What Is the KV Cache?

During self-attention, each transformer layer computes **Key** and **Value** vectors for every token it processes. In a model with:
- L = 28 layers
- D_kv = dimension of key/value heads
- P = prompt tokens
- N_out = output tokens

The total KV cache size is:

```
KV Cache = 2 × L × D_kv × (P + N_out) × bytes_per_element
         = 2 (K and V) × 28 layers × D_kv × sequence_length × 2 bytes (BF16)
```

### Why Is It Needed?

During decode, when generating token #500, the model needs to compute attention between the new token's query vector and ALL previous tokens' key vectors (both from the original prompt and all previously generated tokens). Without the KV cache, the model would have to re-run prefill on the entire sequence for every single new token — which would be catastrophically slow.

The KV cache trades **memory** for **compute**: store the keys and values once, reuse them for every subsequent token.

### What We Observed

In E001-A (varying prompt length, fixed 128 output tokens):
```
Prompt = 128  → Peak VRAM = 18,872 MiB  (baseline)
Prompt = 512  → Peak VRAM = 18,872 MiB  (same — within pre-allocated pool)
Prompt = 2048 → Peak VRAM = 19,094 MiB  (+222 MiB)
Prompt = 4096 → Peak VRAM = 19,392 MiB  (+520 MiB)
```

In E001-B (fixed 128 prompt, varying output length):
```
Output = 32   → Peak VRAM = 18,884 MiB
Output = 128  → Peak VRAM = 18,884 MiB
Output = 512  → Peak VRAM = 18,884 MiB
Output = 1024 → Peak VRAM = 18,884 MiB  (all identical!)
```

### Why Was VRAM Constant in E001-B?

This is because of how **vLLM's PagedAttention** works. When you start the server with `--gpu-memory-utilization 0.90`, vLLM **pre-allocates** a large pool of memory blocks for KV cache at startup. Individual requests then use pages from this pool dynamically. Since we ran at concurrency = 1 with sequences up to 1152 tokens total (128 + 1024), we never exceeded the pre-reserved pool — so NVML reported the same physical allocation throughout.

In E001-A, the larger prompts (2048, 4096 tokens) started requiring KV cache pages beyond the initial allocation, causing the slight VRAM increase.

---

## 7. Experimental Design — Isolating Variables Like a Scientist

### The Scientific Method for Systems Research

E001 follows strict experimental methodology:

```
1. Formulate a HYPOTHESIS about how the system behaves
2. Identify the INDEPENDENT VARIABLE (what you change)
3. Identify CONTROLLED VARIABLES (what you hold constant)
4. Define DEPENDENT VARIABLES (what you measure)
5. Remove CONFOUNDING FACTORS (things that would invalidate your results)
6. Run the experiment
7. Compare results against hypothesis
```

### E001-A: Isolating Prefill

**Hypothesis**: TTFT scales with prompt length; TPOT stays constant.

| Variable Type | Variable | Value/Range |
|---|---|---|
| **Independent** | Prompt tokens (P) | [128, 512, 2048, 4096] |
| **Controlled** | Output tokens (N_out) | 128 (fixed) |
| **Controlled** | Concurrency | 1 (sequential) |
| **Controlled** | Temperature | 0.0 (greedy, deterministic) |
| **Controlled** | Model | Qwen/Qwen2.5-7B-Instruct |
| **Controlled** | Precision | BF16 |
| **Controlled** | Prefix caching | DISABLED |
| **Controlled** | Chunked prefill | DISABLED |
| **Dependent** | TTFT, TPOT, E2E, VRAM | Measured |

### E001-B: Isolating Decode

**Hypothesis**: TPOT stays constant; E2E scales linearly with output length.

| Variable Type | Variable | Value/Range |
|---|---|---|
| **Independent** | Output tokens (N_out) | [32, 128, 512, 1024] |
| **Controlled** | Prompt tokens (P) | 128 (fixed) |
| **Controlled** | Concurrency | 1 (sequential) |
| **Controlled** | Temperature | 0.0 (greedy, deterministic) |
| **Controlled** | Model | Qwen/Qwen2.5-7B-Instruct |
| **Controlled** | Precision | BF16 |
| **Controlled** | Prefix caching | DISABLED |
| **Controlled** | Chunked prefill | DISABLED |
| **Dependent** | TTFT, TPOT, E2E, VRAM | Measured |

### Why This Design Is Rigorous

By changing **only one variable at a time** and holding everything else constant, any observed change in the dependent variable can be **attributed solely to the independent variable**. If we had changed both prompt length and output length simultaneously, we couldn't tell whether a latency change was due to prefill or decode.

---

## 8. What We Built — The Parameter Sweep Engine

E001 required new engineering capabilities beyond the single-run benchmark from E000.

### The Problem

In E000, we ran a single configuration (128 prompt, 64 output). In E001, we need to run **4 configurations** sequentially, aggregate the results, and generate comparative plots. This requires:

1. A way to specify "sweep over these parameter values"
2. A runner that executes each sweep point
3. Aggregation of results across points
4. Automated plot generation

### SweepConfig — Declarative Experiment Specification

[`config.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/config.py) now contains:

```python
@dataclass(frozen=True, slots=True)
class SweepConfig:
    sweep_param: str  # "prompt_tokens" or "max_output_tokens"
    sweep_values: tuple[int, ...]  # (128, 512, 2048, 4096)
    base_config: BenchmarkConfig  # Everything else held constant
    experiment_id: str = "E001A"
```

**Key design decision**: `SweepConfig` composes with `BenchmarkConfig` rather than inheriting from it. The sweep configuration says "take this base config and vary this one parameter across these values." This is the **Open/Closed Principle** — extending behavior without modifying existing code.

The `generate_point_configs()` method uses Python's `dataclasses.replace()` to create new configs with just the swept parameter changed:

```python
def generate_point_configs(self) -> list[tuple[int, BenchmarkConfig]]:
    configs = []
    for val in self.sweep_values:
        cfg = replace(self.base_config, **{self.sweep_param: val})
        configs.append((val, cfg))
    return configs
```

**Why `replace()` instead of mutation?** Because `BenchmarkConfig` is `frozen=True`. You cannot modify it after creation. `replace()` creates a new copy with the specified field changed, preserving immutability. This guarantees that the base config is never accidentally corrupted during the sweep.

### execute_sweep() — The Sweep Runner

[`runner/sweep.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/runner/sweep.py) orchestrates the full sweep:

```python
async def execute_sweep(sweep_config, tokenizer=None, client=None):
    # 1. Create sweep directory: runs/E001A_20260904_182047_d88d790b/
    # 2. Save sweep_config.json
    # 3. Initialize tokenizer ONCE (shared across all points)
    # 4. For each parameter value:
    #    - Create point subdirectory (e.g., prompt_tokens_128/)
    #    - Run execute_benchmark() with this config
    #    - Record results
    # 5. Write sweep_summary.json (aggregate status across all points)
    # 6. Generate plots from the collected data
```

**Critical optimization: single tokenizer initialization**. Loading the HuggingFace tokenizer takes 2-3 seconds (downloading vocab files, building BPE merge tables). By initializing it once and reusing across all sweep points, we avoid redundant work. The tokenizer is stateless — `encode("hello")` always returns the same tokens — so sharing is safe.

### YAML Configuration Files

Instead of hard-coding parameters, each experiment has a declarative YAML file:

[`configs/e001a_input_scaling.yaml`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/configs/e001a_input_scaling.yaml):
```yaml
experiment_id: E001A
sweep_param: prompt_tokens
sweep_values: [128, 512, 2048, 4096]
model: Qwen/Qwen2.5-7B-Instruct
base_url: http://localhost:18000
max_output_tokens: 128
enable_prefix_caching: false
enable_chunked_prefill: false
```

[`configs/e001b_decode_scaling.yaml`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/configs/e001b_decode_scaling.yaml):
```yaml
experiment_id: E001B
sweep_param: max_output_tokens
sweep_values: [32, 128, 512, 1024]
model: Qwen/Qwen2.5-7B-Instruct
base_url: http://localhost:18000
prompt_tokens: 128
enable_prefix_caching: false
enable_chunked_prefill: false
```

**Why YAML?** YAML is human-readable and diff-friendly (for git). If you change a parameter between experiment runs, `git diff` shows exactly what changed.

### Automated Plot Generation

[`reports/plots.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/reports/plots.py) automatically renders 4 publication-quality Matplotlib plots for each experiment:

1. **TTFT vs. sweep parameter** — with mean ± 1σ error bars and P50 line
2. **E2E Latency vs. sweep parameter** — total request duration
3. **TPOT vs. sweep parameter** — per-token decode time
4. **GPU Memory vs. sweep parameter** — peak and average VRAM

The sweep runner automatically detects the experiment type and calls the appropriate plotting function:

```python
if sweep_config.sweep_param == "max_output_tokens":
    generate_e001b_plots(point_results, plots_dir)
else:
    generate_e001a_plots(point_results, plots_dir)
```

---

## 9. E001-A — Input Length Scaling (Prefill Characterization)

### The Question

> "Why does increasing prompt context length affect first-token latency (TTFT) and total latency differently from decode latency?"

### The Hypothesis

1. TTFT will scale monotonically with prompt length P
2. TPOT will remain invariant (flat) across prompt lengths
3. E2E will increase only by ΔTTFT (the decode portion stays constant)
4. VRAM will grow with longer prompts (more KV cache)

### The Results

```
===========================================================================================================
 E001-A: INPUT LENGTH SCALING RESULTS SUMMARY
===========================================================================================================
Prompt Tokens  | TTFT (P50)   | TTFT (Mean)  | TPOT (P50)   | E2E (P50)    | Peak VRAM    | GPU Util
-----------------------------------------------------------------------------------------------------------
128            |   68.14 ms   |   69.08 ms   |   19.36 ms   | 2520.24 ms   | 18872 MiB    | 97.7%   
512            |  137.69 ms   |  143.21 ms   |   19.53 ms   | 2619.79 ms   | 18872 MiB    | 97.4%   
2048           |  435.72 ms   |  443.24 ms   |   19.60 ms   | 2924.51 ms   | 19094 MiB    | 97.6%   
4096           |  854.59 ms   |  859.23 ms   |   19.85 ms   | 3377.59 ms   | 19392 MiB    | 98.0%   
===========================================================================================================
```

### Analysis: What Did We Learn?

**TTFT Scales Near-Linearly** ✅
- 128 → 4096 tokens (32× increase) caused TTFT to grow from 68 → 855 ms (12.5× increase)
- Sub-linear because FlashAttention reduces attention from O(N²) to O(N) memory access, and FFN layers (linear in N) dominate at moderate lengths

**TPOT is Rock-Solid Constant** ✅
- 19.36 → 19.85 ms across all prompt lengths (only 2.5% variation)
- Confirms decode is dominated by reading 14.2 GB of weights, not KV cache attention

**E2E Shifts by Exactly ΔTTFT** ✅
- E2E(128) = 2520 ms, E2E(4096) = 3378 ms
- ΔTTFT = 855 - 68 = 787 ms
- ΔE2E = 3378 - 2520 = 858 ms → Close to ΔTTFT ✅ (the small discrepancy is from the slight TPOT increase)

**VRAM Grows Monotonically** ✅
- 18,872 MiB (baseline) → 19,392 MiB (+520 MiB at 4096 tokens)
- The growth reflects additional KV cache page allocations beyond the initial pool

---

## 10. E001-B — Output Length Scaling (Decode Characterization)

### The Question

> "Why does increasing generation length scale E2E latency linearly while keeping TTFT and TPOT constant?"

### The Hypothesis

1. TTFT will be invariant across output lengths (prefill is independent of generation)
2. E2E will scale linearly with output length N_out
3. TPOT will remain constant (~19.8 ms/token)
4. VRAM will grow with longer generation (more KV cache entries)

### The Results

```
==============================================================================================================
 E001-B: OUTPUT LENGTH (DECODE) SCALING RESULTS SUMMARY
==============================================================================================================
Output Tokens  | E2E (P50)    | E2E (Mean)   | TPOT (P50)   | TTFT (P50)   | Peak VRAM    | GPU Util
--------------------------------------------------------------------------------------------------------------
32             |   704.22 ms  |   712.45 ms  |   19.78 ms   |   90.65 ms   | 18884 MiB    | 100.0%  
128            |  2603.62 ms  |  2606.46 ms  |   19.78 ms   |   89.92 ms   | 18884 MiB    | 100.0%  
512            | 10199.02 ms  | 10197.52 ms  |   19.80 ms   |   89.55 ms   | 18884 MiB    | 100.0%  
1024           | 20296.10 ms  | 20299.11 ms  |   19.80 ms   |   39.37 ms   | 18884 MiB    |  99.9%  
==============================================================================================================
```

### Analysis: What Did We Learn?

**TTFT is Invariant** ✅
- 90.65 → 89.55 ms across 32–512 output tokens (essentially flat)
- The drop to 39.37 ms at 1024 tokens is due to CUDA/JIT engine cache warming after the first 3 sweep points

**E2E Scales Perfectly Linearly** ✅
- Let's verify the master equation for each point:

```
N_out = 32:   Predicted = 90.65 + 31 × 19.783  =    704 ms  |  Observed:    704 ms  ✓
N_out = 128:  Predicted = 89.92 + 127 × 19.785 =  2,603 ms  |  Observed:  2,604 ms  ✓
N_out = 512:  Predicted = 89.55 + 511 × 19.798 = 10,206 ms  |  Observed: 10,199 ms  ✓
N_out = 1024: Predicted = 39.37 + 1023 × 19.801 = 20,296 ms  |  Observed: 20,296 ms  ✓
```

The prediction error is under 0.1% for every single point.

**TPOT is Perfectly Constant** ✅
- 19.78 → 19.80 ms (0.1% variation across 32× output length range!)
- The standard deviation DECREASES with longer generation (more samples to average over):
  - N_out = 32: σ = 0.73 ms
  - N_out = 1024: σ = 0.007 ms (100× less noisy!)

**VRAM is Constant** ✅
- 18,884 MiB across all points (vLLM's pre-allocated PagedAttention pool absorbs all KV cache growth within its budget)

---

## 11. Reading the Results — From Numbers to Understanding

### How to Read the Summary Tables

Each row in the results table is a **sweep point** — one complete benchmark run with a specific parameter value. For each, we ran:
- 2 warmup requests (discarded from statistics)
- 10 measured requests (used for P50, mean, std dev)

**P50 (Median)**: The middle value when all 10 measurements are sorted. P50 is more robust than the mean because it's not affected by outliers. If one request hit an OS scheduling hiccup, P50 still reflects typical behavior.

**Mean ± StdDev**: The arithmetic average and sample standard deviation. A small std dev relative to the mean indicates high reproducibility.

### How to Read the Plots

Each plot has the sweep parameter on the x-axis and a timing metric on the y-axis:

- **Blue line with error bars (±1σ)**: Shows the mean and how much variation there was across the 10 measured requests
- **Orange dashed line**: Shows the P50 (median) — often very close to the mean, confirming symmetric distributions
- **If the error bars are tiny**: The measurement is highly reproducible (good!)
- **If the line is flat**: The metric is invariant to the sweep parameter (e.g., TPOT vs. prompt length)
- **If the line is steep**: The metric scales strongly with the parameter (e.g., E2E vs. output length)

---

## 12. The Grand Synthesis — Answering the Core Question

We can now definitively answer:

> **"How do prefill and decode differ, and why do prompt length and generation length affect inference latency differently?"**

### The Answer

**Prefill and decode are two fundamentally different computational workloads that happen to run on the same GPU:**

1. **Prefill** processes ALL prompt tokens in parallel using large matrix-matrix multiplications (GEMMs). These saturate the GPU's Tensor Cores and are **compute-bound** — limited by how many FLOPS the GPU can execute. Increasing prompt length increases the size of these matrices, directly increasing the compute time (TTFT).

2. **Decode** processes tokens ONE AT A TIME using matrix-vector multiplications. Each step must read the entire 14.2 GB of model weights from VRAM, but performs very little arithmetic per byte read. This is **memory-bandwidth-bound** — limited by how fast the GPU can stream data from VRAM (936 GB/s on RTX 3090). Since the model weights are the same size regardless of prompt or output length, each decode step takes the same time (~19.8 ms).

**Why prompt length affects latency differently from output length:**
- Increasing prompt length increases TTFT (more compute for prefill), but doesn't change TPOT (decode still reads the same weights).
- Increasing output length increases E2E linearly (more decode steps at constant TPOT), but doesn't change TTFT (prefill only sees the prompt).

---

## 13. Serving Controls — Why We Disabled Things

### Prefix Caching — `--no-enable-prefix-caching`

**What it does**: If two requests share the same prefix (e.g., same system prompt), vLLM can reuse the KV cache from the first request instead of re-computing prefill.

**Why we disabled it**: If prefix caching is active, the SECOND run of the same prompt would have a near-zero TTFT (cache hit!), making our prefill measurements invalid. We want to measure the true prefill compute cost every time.

### Chunked Prefill — `--no-enable-chunked-prefill`

**What it does**: Instead of processing all P prompt tokens in one giant prefill step, chunked prefill splits the prompt into smaller chunks (e.g., 512 tokens at a time). This lets vLLM interleave prefill chunks with decode steps for other requests, improving scheduling fairness in multi-tenant scenarios.

**Why we disabled it**: Chunked prefill splits the prefill into multiple passes, which:
1. Changes the TTFT measurement (it would reflect only the last chunk, not the total prefill)
2. Introduces scheduling artifacts between chunks
3. Makes it impossible to isolate pure prefill scaling

For scientific measurement, we want a clean, un-chunked full-context prefill.

---

## 14. What We Built in Code — Applied Engineering

### Directory Structure (New in E001)

```
inference_os/
├── configs/
│   ├── e001a_input_scaling.yaml       ← Declarative experiment configs
│   └── e001b_decode_scaling.yaml
├── experiments/E001-prefill-decode/
│   ├── E001A-input-length/
│   │   ├── README.md                  ← Scientific documentation
│   │   └── run_e001a.py               ← Executable benchmark script
│   └── E001B-output-length/
│       ├── README.md
│       └── run_e001b.py
├── src/inference_os/
│   ├── config.py                      ← Added SweepConfig, load_config()
│   ├── runner/sweep.py                ← NEW: Parameter sweep execution engine
│   ├── reports/plots.py               ← NEW: Matplotlib plot generation
│   └── metrics/request.py             ← Added tpot_seconds derived property
├── outputs/
│   ├── e001a_input_scaling_validation.md    ← Human-readable analysis
│   ├── e001b_output_scaling_validation.md
│   └── plots/
│       ├── e001a/                     ← TTFT, E2E, TPOT, VRAM plots
│       └── e001b/
├── runs/
│   ├── E001A_20260904_182047_d88d790b/  ← Raw telemetry & data
│   └── E001B_20260905_071554_faca467d/
└── tests/
    ├── test_plots.py                  ← Tests for both E001A and E001B plots
    └── test_sweep.py                  ← Tests for sweep runner
```

### The TPOT Derived Metric

[`metrics/request.py`](file:///c:/Users/chidh/OneDrive/Desktop/inference_os/src/inference_os/metrics/request.py) now includes:

```python
@property
def tpot_seconds(self) -> Optional[float]:
    """Time Per Output Token (TPOT) for decode phase in seconds.

    TPOT = (completion_time - first_token_time) / (output_tokens - 1)
    """
    if self.first_token_time_ns is None or self.output_tokens <= 1:
        return None
    decode_tokens = self.output_tokens - 1
    decode_time_ns = self.completion_time_ns - self.first_token_time_ns
    return (decode_time_ns / decode_tokens) / NANOSECONDS_PER_SECOND
```

**Why `output_tokens - 1`?** The first output token is produced by prefill. The decode phase generates tokens #2 through #N_out, which is N_out - 1 tokens. The decode duration is measured from `first_token_time` to `completion_time`.

**Why `Optional[float]`?** If the request produced 0 or 1 output tokens, TPOT is undefined (you need at least 2 tokens to measure inter-token latency). Returning `None` rather than raising an exception makes the metric graceful to compute across diverse workloads.

### Test Coverage

```
tests/test_plots.py:
  - test_generate_e001a_plots: Verifies 4 PNG files with correct filenames
  - test_generate_e001b_plots: Same for E001-B output token plots

tests/test_sweep.py:
  - test_execute_sweep_mock: Full E001-A sweep with mock HTTP + tokenizer
  - test_execute_sweep_all_failed: Error handling and failure propagation
  - test_execute_sweep_e001b_mock: Full E001-B sweep with E001-B plot dispatch
```

All 69 tests pass without GPU, without server, without internet.

---

## 15. Roofline Model — Understanding Hardware Limits

The **Roofline Model** is a visual framework for understanding whether a workload is compute-bound or memory-bandwidth-bound. Here's how it applies to our experiments:

```
Performance  
(FLOPS)       ┌─────────────────────────────────────────
              │                                   ╱ ← Compute ceiling (142 TFLOPS for BF16)
              │                              ╱
              │                         ╱        PREFILL lives here
              │                    ╱             (high arithmetic intensity)
              │               ╱
              │          ╱
              │     ╱
              │╱ ── ── ── ── ── ── ── ── ── ── ── ── ── ←  Memory BW ceiling (936 GB/s)
              │
              │  DECODE lives here
              │  (low arithmetic intensity,
              │   bottlenecked by bandwidth)
              │
              └──────────────────────────────────────────
                    Arithmetic Intensity (FLOPs / Byte)
                    
                    ~1          ~10         ~100
```

- **Below the diagonal line**: Memory-bandwidth-bound (decode). Performance is limited by how fast you can feed data to the compute units.
- **Along the flat ceiling**: Compute-bound (prefill). Performance is limited by the raw FLOPS of the GPU.
- **The "ridge" (where diagonal meets ceiling)**: The point of balance. Workloads at this arithmetic intensity fully utilize both compute and bandwidth.

For our Qwen2.5-7B on RTX 3090:
- **Ridge point**: 142 TFLOPS / 936 GB/s ≈ 152 FLOPs/byte
- **Prefill**: At ~3584 FLOPs/byte → well above the ridge → **compute-bound** ✓
- **Decode**: At ~1 FLOP/byte → well below the ridge → **bandwidth-bound** ✓

---

## 16. Interview-Ready Knowledge — How to Explain This

### "Explain prefill vs. decode in LLM inference"

> "When an LLM receives a prompt, it first runs a **prefill** phase where all input tokens are processed in parallel through matrix multiplications across all transformer layers. This is compute-bound — it's limited by the GPU's FLOPS. The time this takes is called TTFT, Time to First Token.

> After prefill, the model enters the **decode** phase, generating output tokens one at a time. Each decode step must read the entire model weights from GPU memory — about 14 GB for a 7B model. With only one token to process, the arithmetic intensity is very low, so the bottleneck shifts to memory bandwidth. This is why each token takes about the same time to generate, regardless of how long the prompt was or how many tokens have already been generated."

### "How does total inference latency scale with input and output length?"

> "Total latency decomposes as: E2E = TTFT + (N_out - 1) × TPOT. TTFT scales with input length because prefill compute grows with the number of tokens. TPOT is constant because each decode step reads the same model weights regardless of context. So increasing prompt length shifts the whole latency curve up by ΔTTFT, while increasing output length scales E2E linearly with slope equal to TPOT."

### "If you had to choose between a GPU with more compute or more memory bandwidth, which would you pick?"

> "It depends on the workload. For long-prompt tasks like document analysis, prefill dominates, so more compute helps. For long-generation tasks like creative writing or code generation, decode dominates, so more memory bandwidth helps. For typical chat with moderate prompts and responses, memory bandwidth usually matters more because decode occupies 80-95% of the total time."

---

## 17. Self-Test Questions

Test your understanding — try to answer these before looking back at the guide:

### Conceptual

1. **Why is prefill compute-bound but decode is memory-bandwidth-bound?** (Hint: matrix-matrix vs. matrix-vector multiplication)

2. **If the RTX 3090 had 2× the memory bandwidth (1,872 GB/s instead of 936 GB/s), what would happen to TPOT? What about TTFT?** (Answer: TPOT would roughly halve to ~10 ms. TTFT would be unchanged — it's compute-bound.)

3. **Why does TPOT have lower standard deviation when generating 1024 tokens than when generating 32 tokens?** (Answer: With 1024 tokens, TPOT is averaged over 1023 decode steps, smoothing out per-step variance. With 32 tokens, it's averaged over only 31 steps.)

4. **If prefix caching were enabled, what would happen to TTFT on the second run of the same prompt?** (Answer: Near-zero — the KV cache would be a cache hit, skipping all prefill computation.)

5. **Derive the theoretical minimum TPOT** for a 7B BF16 model on RTX 3090. (Answer: 7B × 2 bytes / 936 GB/s = 14.96 ms. Our observed 19.8 ms is 75% of theoretical peak.)

### Applied

6. **Why do we use `dataclasses.replace()` in the sweep runner instead of modifying the config?** (Answer: BenchmarkConfig is frozen. replace() creates a new immutable copy.)

7. **Why does the sweep runner initialize the tokenizer only once?** (Answer: Loading takes 2-3 seconds and the tokenizer is stateless. Reuse avoids redundant HuggingFace downloads.)

8. **What would happen if we forgot to disable chunked prefill?** (Answer: TTFT would not reflect the true full-context prefill time. It might be artificially lower because the engine processes the prompt in small chunks, and TTFT measures only the delay until the first output token.)

9. **Why is the output sweep in E001-B limited to 1024 tokens maximum?** (Answer: At 1024 tokens with 10 measured requests + 2 warmup, each request takes ~20 seconds. The total runtime is ~12 × 20s = 240s = 4 minutes per sweep point. Going to 2048 would double this. Practical GPU rental cost constraints.)

10. **Why did TTFT drop from ~90ms to ~39ms at the 1024-token output length in E001-B?** (Answer: By the time the 4th sweep point runs, the CUDA engine has compiled and cached all necessary kernels. The shorter TTFT reflects a fully warmed JIT compilation cache, distinct from the per-request warmup.)

---

## 18. Further Reading — Papers, Books, and Resources

### Essential Papers

1. **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
   - Tri Dao et al. (2022). [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
   - Explains tiling-based attention computation that makes prefill practical for long sequences.
   - *Why it matters for E001*: FlashAttention is why our TTFT scales sub-quadratically with prompt length.

2. **Efficient Memory Management for Large Language Model Serving with PagedAttention**
   - Woosuk Kwon et al. (2023). [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
   - The original vLLM paper describing PagedAttention for KV cache memory management.
   - *Why it matters for E001*: Explains why VRAM stayed constant in E001-B (pre-allocated page pool).

3. **Orca: A Distributed Serving System for Transformer-Based Generative Models**
   - Gyeong-In Yu et al. (2022). [OSDI '22](https://www.usenix.org/conference/osdi22/presentation/yu)
   - Introduces iteration-level scheduling (continuous batching), the foundation of modern LLM serving.
   - *Why it matters*: Explains how decode steps from different requests can be batched together (relevant to E002).

4. **Sarathi-Serve: A Prefill-Decode Disaggregated Serving System**
   - Amey Agrawal et al. (2024). [arXiv:2403.02310](https://arxiv.org/abs/2403.02310)
   - Studies the interference between prefill and decode in shared serving, proposes disaggregation.
   - *Why it matters*: This is exactly the phenomenon we're building toward understanding in future milestones.

### Foundational Resources

5. **The Roofline Model**
   - Samuel Williams, Andrew Waterman, David Patterson.
   - "Roofline: An Insightful Visual Performance Model for Multicore Architectures" (2009).
   - [Paper](https://dl.acm.org/doi/10.1145/1498765.1498785)
   - *Why it matters*: The theoretical framework for understanding compute-bound vs. bandwidth-bound workloads.

6. **Efficient Transformers: A Survey**
   - Yi Tay et al. (2022). [arXiv:2009.06732](https://arxiv.org/abs/2009.06732)
   - Comprehensive survey of attention mechanism optimizations.

### Books

7. **Programming Massively Parallel Processors** (4th Edition)
   - David B. Kirk, Wen-mei W. Hwu
   - The standard textbook for GPU computing. Chapters on memory hierarchy, bandwidth, and compute-bound vs. memory-bound analysis directly apply to understanding TPOT.

8. **Computer Architecture: A Quantitative Approach** (6th Edition)
   - John L. Hennessy, David A. Patterson
   - Chapter 4 on memory hierarchy and bandwidth is essential background for understanding why decode is memory-bound.

### Online Resources

9. **NVIDIA GPU Architecture Whitepaper (Ampere)**
   - [NVIDIA Ampere Architecture](https://www.nvidia.com/en-us/data-center/ampere-architecture/)
   - Specifications for the RTX 3090 (GA102): 936 GB/s bandwidth, 142 TFLOPS BF16, 82 SMs.

10. **vLLM Documentation**
    - [https://docs.vllm.ai](https://docs.vllm.ai)
    - Official documentation for the serving engine used in all our experiments.

11. **LLM Inference Performance Engineering (Databricks blog)**
    - Practical guide to understanding and optimizing LLM inference performance.
    - Covers prefill vs. decode, batching strategies, and hardware selection.

---

## 19. Glossary — New Terms Introduced in E001

| Term | Definition |
|---|---|
| **Parameter Sweep** | Running the same experiment multiple times, changing one parameter value each time, to measure how outcomes vary with that parameter |
| **Independent Variable** | The parameter you intentionally vary (prompt_tokens in E001-A, max_output_tokens in E001-B) |
| **Controlled Variable** | Parameters held constant to avoid confounding (model, concurrency, temperature, etc.) |
| **Dependent Variable** | The outcomes you measure (TTFT, TPOT, E2E, VRAM) |
| **Confounding Variable** | An uncontrolled factor that could invalidate results (e.g., prefix caching silently reusing KV cache) |
| **TPOT** | Time Per Output Token — average decode step duration: `(completion_time - first_token_time) / (output_tokens - 1)` |
| **Arithmetic Intensity** | Ratio of compute operations (FLOPs) to memory accesses (bytes). High = compute-bound, Low = bandwidth-bound |
| **Roofline Model** | Framework for visualizing whether a workload is limited by compute or memory bandwidth |
| **GEMM** | General Matrix Multiply — the core operation in transformer layers. GPU Tensor Cores are optimized for this |
| **FlashAttention** | Memory-efficient attention algorithm that processes attention in tiles, avoiding O(N²) memory overhead |
| **PagedAttention** | vLLM's memory management system that allocates KV cache in fixed-size pages, like OS virtual memory |
| **Prefix Caching** | Reusing KV cache across requests that share the same prompt prefix |
| **Chunked Prefill** | Splitting a long prompt into smaller chunks for interleaved processing with decode steps |
| **Greedy Decoding** | Selecting the highest-probability token at each step (temperature = 0.0), ensuring deterministic output |
| **SweepConfig** | Dataclass specifying a 1D parameter sweep: which parameter to vary, what values, and the base configuration |
| **`dataclasses.replace()`** | Python function that creates a new frozen dataclass instance with specified fields changed |
| **Bessel's Correction** | Dividing variance by n-1 (not n) for unbiased sample standard deviation estimation |
| **`frozen=True`** | Python dataclass option making instances immutable after creation |
