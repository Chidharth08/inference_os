# inference_os

A reproducible framework for designing, running, measuring, and analyzing LLM inference experiments.

## Project Objective

`inference_os` is an open-source-quality engineering project for building a trustworthy and reproducible LLM inference experimentation framework.

The **engineering goal** is to create a system that allows an engineer to:

- define a controlled inference experiment,
- run it against an LLM serving backend,
- generate repeatable workloads,
- collect request-level latency and throughput measurements,
- capture GPU telemetry,
- persist raw results,
- capture enough environment metadata to reproduce the run,
- compare controlled configurations,
- generate plots and experiment reports,
- reason about why the observed behavior occurred.

The project is not intended to be only a collection of notebooks, benchmark scripts, or one-off performance tests.

**The framework itself is the primary software artifact being built.**

### Target User

The target user is an engineer who wants to run controlled, reproducible experiments to understand how LLM workloads and serving configurations affect inference performance.

A future user should eventually be able to run something conceptually similar to:

```bash
inference-os run configs/context_scaling.yaml
```

and receive a reproducible experiment output containing raw measurements, summaries, environment metadata, plots, and an experiment report.

For example:

```text
runs/E001-2026-08-10/
├── config.yaml
├── environment.json
├── requests.parquet
├── gpu_metrics.parquet
├── summary.json
├── plots/
│   ├── ttft_vs_context.png
│   ├── memory_vs_context.png
│   └── throughput.png
└── report.md
```

In early versions, `inference_os` focuses primarily on **measurement, experimentation, and explanation**.

Optimization capabilities should be introduced only after the framework can reliably characterize how and why a serving configuration behaves the way it does.

---

# Why This Project Exists

This project has three separate goals.

## 1. Engineering Goal

Build a reusable and reproducible **LLM inference benchmarking and experimentation framework**.

The framework should make it possible to study inference behavior without rewriting measurement, telemetry, result storage, workload generation, and experiment-control infrastructure for every new question.

The engineering system is the primary project artifact.

---

## 2. Learning Goal

Use the framework to develop a deep understanding of LLM inference systems through controlled experiments.

The preferred learning loop is:

**concept → hypothesis → experiment → measurement → interpretation → deeper question**

The project should help answer questions such as:

- What happens during prefill versus decode?
- How does prompt length affect TTFT?
- How does output length affect decode latency?
- How does concurrency affect throughput and latency?
- Why does batching improve GPU utilization?
- When does throughput stop scaling with concurrency?
- How does KV-cache usage change with workload characteristics?
- What causes a serving workload to become compute-, memory-, or scheduling-bound?

Learning is therefore a major purpose of the project, but the project is **not a learning platform**.

The learning happens by building, validating, and using the experimentation framework.

---

## 3. Long-Term Product Direction

The long-term direction is to evolve the framework approximately from:

**measurement → comparison → experimentation → optimization → decision support**

A future version may help answer questions such as:

> Given a model, workload, GPU, traffic target, and latency SLA, which serving configuration should I use?

Potential future capabilities may include:

- comparing multiple serving backends,
- comparing precision and quantization strategies,
- identifying saturation points,
- automatically searching concurrency settings,
- estimating serving cost,
- comparing hardware configurations,
- detecting performance regressions,
- characterizing workload types,
- recommending serving configurations,
- capacity planning.

These are **long-term directions**, not V1 requirements.

The project does not need novel research or unique functionality in its first versions.

The immediate priority is to build a small system correctly, understand the measurements deeply, and expand only when experiments justify additional complexity.

---

# Open-Source Philosophy

`inference_os` should be developed with open-source-quality engineering practices.

That means another engineer should eventually be able to:

- clone the repository,
- install the project,
- understand how experiments are defined,
- reproduce benchmark runs,
- inspect raw results,
- understand the measurement methodology,
- extend the framework,
- add new experiments or backends without reverse-engineering the project.

Open source is the **distribution and engineering model**, not the project objective itself.

The project does not need immediate adoption, GitHub stars, external contributors, or novel research to be successful.

If meaningful differentiation emerges later through experimentation or usage, it can be added naturally.

---

# What Success Looks Like

A successful V1 is not simply:

> I learned about TTFT, KV cache, batching, and concurrency.

A successful V1 is:

> I built a trustworthy single-backend inference experimentation framework, used it to run controlled studies of prompt length, output length, and concurrency, and can explain both the framework design and the observed systems behavior.

By the end of V1, the framework should be capable of:

1. accepting an experiment configuration,
2. generating controlled workloads,
3. running requests against vLLM,
4. collecting request-level measurements,
5. collecting GPU telemetry,
6. capturing environment metadata,
7. storing raw results,
8. computing aggregate metrics,
9. generating basic plots,
10. producing reproducible experiment reports.

The V1 experiments serve two purposes:

- they investigate real inference behavior,
- they introduce engineering requirements that progressively shape the framework.

---

# V1 Scope

V1 is intentionally small.

We will use:

- **Model:** `Qwen/Qwen2.5-7B-Instruct`
- **Serving backend:** vLLM
- **Precision:** BF16
- **Hardware:** 1× NVIDIA RTX 3090 GPU (24 GB VRAM) on Vast.ai *(temporary compute target due to cloud quota limits; future NVIDIA L4 or other GPU results must be captured as separate hardware environments and not mixed directly)*
- **Workload:** controlled synthetic prompts
- **Primary experimental variables:**
  - prompt/input length
  - generated/output length
  - request concurrency

### Hardware Environment & Ephemeral GPU Workflow

* **Hardware Isolation:** V1 benchmark runs currently target a 1× RTX 3090 (24 GB VRAM) instance on Vast.ai. Benchmark runs must capture explicit hardware metadata. Results from different hardware environments (e.g. RTX 3090 vs L4) must be treated as distinct environments and never aggregated or directly compared without hardware isolation.
* **Ephemeral GPU Workflow:** All software development, architecture design, documentation, unit testing, and analysis take place locally. Vast.ai instances are provisioned ephemerally only when active GPU benchmark runs are required. Focused GPU sessions typically run 1–3 hours. All code and raw result artifacts must be committed and pushed to GitHub before destroying the instance. Instances must be destroyed (not paused) after sessions to prevent unnecessary stopped storage costs (~$0.32/day).

V1 contains three experiment groups.

---

## E000 — Measurement Validation

### Question

> Can the benchmark harness make trustworthy and reproducible measurements?

We will validate:

- prompt token counts,
- generated token counts,
- request timing,
- TTFT measurement,
- end-to-end latency measurement,
- warm-up behavior,
- repeated-run variance,
- GPU telemetry,
- environment capture,
- error handling.

### Engineering Capability Introduced

E000 should establish the minimum trustworthy benchmark pipeline:

```text
experiment configuration
        ↓
workload generation
        ↓
benchmark runner
        ↓
request-level measurements
        ↓
GPU telemetry
        ↓
raw result persistence
        ↓
environment capture
```

### Systems / Interview Question

By the end of E000, we should be able to answer:

> How do you know that an LLM inference benchmark is measuring what you think it is measuring?

---

## E001 — Prefill and Decode Scaling

E001 contains two controlled experiments.

### E001-A — Input Length Scaling

Hold output length and concurrency constant.

Vary prompt length.

Study:

- TTFT,
- end-to-end latency,
- TPOT,
- GPU utilization,
- GPU memory.

Main question:

> Why does increasing context length affect first-token latency?

### E001-B — Output Length Scaling

Hold prompt length and concurrency constant.

Vary generated output length.

Study:

- TTFT,
- end-to-end latency,
- TPOT,
- GPU utilization.

Main question:

> Why does increasing generation length affect total latency differently from increasing prompt length?

### Engineering Capability Introduced

E001 should require the framework to support:

- parameter sweeps,
- controlled-variable experiments,
- repeated runs,
- result aggregation,
- plot generation,
- experiment comparison.

### Systems / Interview Question

By the end of E001, we should be able to answer:

> How do prefill and decode differ, and why do prompt length and generation length affect inference latency differently?

---

## E002 — Concurrency

Hold input/output lengths constant.

Vary request concurrency.

Study:

- request throughput,
- output-token throughput,
- TTFT P50/P95,
- TPOT P50/P95,
- E2E P50/P95,
- GPU utilization,
- GPU memory,
- error rate.

Main question:

> Why can higher concurrency improve throughput while simultaneously hurting latency?

### Engineering Capability Introduced

E002 should require the framework to support:

- concurrent workload generation,
- percentile metrics,
- throughput measurement,
- saturation analysis,
- error-rate tracking.

### Systems / Interview Question

By the end of E002, we should be able to answer:

> Why can concurrency and batching increase serving throughput, and why does latency eventually worsen as load increases?

---

# Explicitly Out of Scope for V1

Do NOT add these yet:

- Hugging Face Transformers backend,
- SGLang,
- TensorRT-LLM,
- quantization,
- INT8,
- INT4,
- FP8,
- prefix caching experiments,
- speculative decoding,
- RAG workloads,
- chat workload suites,
- code-generation workload suites,
- multiple GPU models,
- multi-GPU inference,
- tensor parallelism,
- pipeline parallelism,
- distributed benchmarking,
- capacity planning,
- configuration recommendation,
- automatic cost optimization,
- cloud provisioning,
- Kubernetes,
- web dashboard,
- hosted service,
- authentication,
- database server,
- plugin architecture,
- Triton kernels,
- custom CUDA kernels.

Do not add infrastructure unless it directly improves:

1. experiment execution,
2. measurement correctness,
3. reproducibility,
4. analysis,
5. systems understanding.

---

# Engineering Principles

## 1. Experiments Drive Architecture

Do not introduce abstractions merely because they may be useful someday.

Build abstractions only after a concrete experiment requires them.

Each experiment should have two outputs:

1. a systems conclusion,
2. a justified improvement to the experimentation framework when necessary.

---

## 2. Raw Measurements Are First-Class Data

Never persist only aggregate statistics.

Each request should eventually have raw measurements such as:

- request ID,
- requested input tokens,
- actual input tokens,
- requested output tokens,
- actual output tokens,
- request start timestamp,
- first-token timestamp,
- completion timestamp,
- status/error.

Aggregates should always be reproducible from raw results.

---

## 3. Benchmarking and Profiling Are Different

Benchmark runs should avoid heavyweight profiling instrumentation.

Profilers such as PyTorch Profiler or Nsight Systems should be used later on short representative workloads to investigate specific observations.

Never use profiler-instrumented runs as canonical performance measurements.

---

## 4. Reproducibility Is Mandatory

Every benchmark run should eventually capture enough metadata to reproduce the environment, including:

- Git commit,
- model name,
- model revision,
- backend,
- backend version,
- Python version,
- PyTorch version,
- CUDA version,
- NVIDIA driver version,
- GPU model,
- precision,
- serving configuration,
- workload configuration,
- generation parameters,
- random seed,
- warm-up policy,
- timestamps.

---

## 5. Controlled Experiments

When studying one variable, keep the others fixed.

Avoid large benchmark matrices until the behavior of individual variables is understood.

Every experiment should clearly state:

- independent variables,
- controlled variables,
- dependent variables / metrics.

---

## 6. Never Hide Benchmark Limitations

Every important experiment must document:

- limitations,
- possible confounders,
- assumptions,
- what the experiment does NOT prove.

Results should not be generalized beyond the tested model, hardware, software environment, and workload without evidence.

---

## 7. The Framework Must Remain Explainable

The project owner should be able to explain every major design decision in a technical interview.

Avoid:

- unnecessary complexity,
- premature abstraction,
- excessive generated boilerplate,
- infrastructure that does not support a current experiment.

When the framework becomes more sophisticated, its complexity should be motivated by experiments that required it.

---

# Experiment Report Standard

Every major experiment should have a permanent report under:

```text
experiments/
```

Each experiment report should contain:

1. Question
2. Relevant inference concept
3. Hypothesis
4. Expected behavior
5. Experimental setup
6. Independent variables
7. Controlled variables
8. Hardware/software environment
9. Metrics
10. Measurement methodology
11. Results
12. Plots
13. Interpretation
14. Unexpected behavior
15. Limitations
16. What this experiment does NOT prove
17. Next question raised

The report should explain **why** observed behavior occurs, not merely state which configuration was faster.

The report should also note whether the experiment exposed any missing capability in the framework.

---

# Initial Repository Structure

Start with:

```text
inference_os/
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
│
├── src/
│   └── inference_os/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       │
│       ├── backends/
│       │   ├── __init__.py
│       │   └── base.py
│       │
│       ├── workloads/
│       │   ├── __init__.py
│       │   └── base.py
│       │
│       ├── runner/
│       │   └── __init__.py
│       │
│       ├── metrics/
│       │   └── __init__.py
│       │
│       ├── telemetry/
│       │   └── __init__.py
│       │
│       ├── results/
│       │   └── __init__.py
│       │
│       └── reports/
│           └── __init__.py
│
├── configs/
│
├── experiments/
│   ├── E000-measurement-validation/
│   ├── E001-prefill-decode/
│   └── E002-concurrency/
│
├── docs/
│   ├── methodology.md
│   ├── metrics.md
│   └── reproducibility.md
│
├── tests/
│
└── .github/
    └── workflows/
```

This is an initial structure, not a permanent architecture.

Directories should only gain complexity when experiments justify it.

---

# Development Tooling

For the initial project foundation:

- **Python:** 3.11+
- **Dependency / environment management:** `uv`
- **Project metadata:** `pyproject.toml`
- **Package layout:** `src/`
- **Testing:** `pytest`
- **Linting / formatting:** `Ruff`
- **CI:** GitHub Actions

Additional dependencies should not be added until a concrete feature requires them.

In particular, V1 setup should not immediately install the full inference stack merely to establish the repository structure.

---

# Initial Development Rules

For now:

- use Python 3.11+,
- use `uv` for local dependency/environment management,
- use `pyproject.toml` as the project configuration source,
- use a `src/` package layout,
- use type hints,
- keep functions small and understandable,
- prefer standard library functionality when reasonable,
- write tests for measurement, math, configuration, and result-processing logic,
- do not add unnecessary frameworks,
- do not add databases,
- do not add web frameworks,
- do not add frontend code,
- do not add Docker/Kubernetes unless later required for reproducibility,
- do not implement speculative abstractions,
- do not optimize code before measuring it,
- do not generate large amounts of boilerplate.

When implementing something non-trivial, first explain:

1. what problem it solves,
2. why the current milestone needs it,
3. what the simplest reasonable design is,
4. what alternatives exist,
5. why the chosen approach is appropriate.

The code should remain understandable enough that the project owner can explain every major design decision in a technical interview.

---

# Planned Core Concepts

The project should explore these approximately in this order:

1. autoregressive generation,
2. tokenization and token counts,
3. prefill,
4. decode,
5. TTFT,
6. TPOT / inter-token latency,
7. end-to-end latency,
8. GPU execution basics,
9. context-length scaling,
10. output-length scaling,
11. batching,
12. continuous batching,
13. concurrency,
14. latency-throughput trade-offs,
15. GPU utilization,
16. GPU memory,
17. KV cache,
18. serving scheduling,
19. profiling,
20. bottleneck identification.

Advanced topics should appear only when experiments motivate them.

---

# Project Evolution

The intended progression is approximately:

```text
V1
Trustworthy experiment runner
(single model / backend / GPU)
        ↓
V2
Configuration comparison
(backends / precision / quantization)
        ↓
V3
Workload characterization
(chat / RAG / code / prefill-heavy / decode-heavy)
        ↓
V4
Automated experimentation
(saturation search / regression testing / configuration sweeps)
        ↓
V5
Decision support
(SLA-aware configuration comparison / capacity planning)
```

This roadmap is directional, not a fixed commitment.

Each stage should only be attempted after the previous stage produces reliable measurements and useful engineering lessons.

---

# Current Milestone

We are currently at:

## Milestone 0 — Repository Foundation

Do NOT implement the complete benchmarking system yet.

The immediate goal is only to:

1. establish the Python project,
2. create the repository structure,
3. make the package importable,
4. create a minimal CLI,
5. establish linting/testing,
6. write the first documentation skeletons,
7. establish CI,
8. verify the project installs cleanly,
9. commit this clean starting point.

After that, we will design **E000 — Measurement Validation** before implementing the vLLM benchmark client.

---

# Milestone 0 Definition of Done

Milestone 0 is complete when all of the following work:

```bash
uv sync
uv run pytest
uv run ruff check .
uv run inference-os --help
```

The repository should also contain:

- an importable `inference_os` package,
- a minimal CLI,
- a valid `pyproject.toml`,
- basic tests,
- basic CI,
- methodology/metrics/reproducibility documentation skeletons.

Milestone 0 should contain **no actual inference benchmarking implementation**.

---

# Current Engineering Question

At this stage, the engineering question to keep in mind is:

> How would you design an inference benchmark so that its results are reproducible and its measurements can be trusted?

Every V1 architecture decision should help us eventually answer that question well.

---

# Project Summary

In one sentence:

> `inference_os` is a reproducible LLM inference experimentation framework built to measure, analyze, and explain serving behavior, while gradually evolving toward configuration comparison and capacity-planning tooling.

The framework is what we are building.

The experiments are how we validate and extend it.

The systems knowledge is what we gain by doing so.
