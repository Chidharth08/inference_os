# inference_os — Study Guide for E003 and E004

## Application-Shaped Workloads, Heterogeneity, and Tail Latency

This guide connects the conceptual systems knowledge behind E003 and E004 to
the code, experiment design, raw artifacts, and empirical results in this
repository. It assumes you understand the E001 distinction between prefill and
decode and the E002 trade-off between concurrency, throughput, and latency.

By the end, you should be able to:

- describe an LLM workload as a distribution rather than one sequence length,
- explain why request throughput and token throughput can tell different stories,
- distinguish configured, requested, and actual token counts,
- explain why equal average sequence lengths do not imply equal tail latency,
- calculate mean, variance, coefficient of variation, and percentile changes,
- trace a workload from YAML configuration to persisted request measurements,
- interpret length-bucket results without overstating causality,
- explain why E003 and E004 still cannot measure sustainable production load,
- design the transition to E005's open-loop saturation experiment.

---

## Table of Contents

1. [Where E003 and E004 Fit](#1-where-e003-and-e004-fit)
2. [A Request-Shape Model](#2-a-request-shape-model)
3. [Metrics You Must Keep Separate](#3-metrics-you-must-keep-separate)
4. [Load Model: Closed Loop](#4-load-model-closed-loop)
5. [E003 — Application-Shaped Workloads](#5-e003--application-shaped-workloads)
6. [Reading the E003 Results](#6-reading-the-e003-results)
7. [Why E003 Alone Is Not Enough](#7-why-e003-alone-is-not-enough)
8. [E004 — Same Mean, Different Variance](#8-e004--same-mean-different-variance)
9. [Reading the E004 Results](#9-reading-the-e004-results)
10. [Length-Bucket Analysis](#10-length-bucket-analysis)
11. [Code Walkthrough: Configuration to Evidence](#11-code-walkthrough-configuration-to-evidence)
12. [How to Audit a Run](#12-how-to-audit-a-run)
13. [The ReadTimeout Lesson](#13-the-readtimeout-lesson)
14. [Common Interpretation Mistakes](#14-common-interpretation-mistakes)
15. [Applied Decision Framework](#15-applied-decision-framework)
16. [Preparing for E005](#16-preparing-for-e005)
17. [Interview-Ready Explanations](#17-interview-ready-explanations)
18. [Exercises](#18-exercises)
19. [Exercise Solutions](#19-exercise-solutions)
20. [Glossary](#20-glossary)
21. [Final Mental Model](#21-final-mental-model)

---

## 1. Where E003 and E004 Fit

The experiments build one inference-systems argument in stages:

| Experiment | Independent variable | Main question |
| :--- | :--- | :--- |
| E001 | Input or output length | How do prefill and decode scale? |
| E002 | Closed-loop concurrency | Why does batching increase throughput while hurting latency? |
| E003 | Application-shaped token distribution | How do different request shapes behave under the same serving controls? |
| E004 | Token-length variance at a controlled mean | Why is average sequence length insufficient? |
| E005 | Open-loop offered request rate | Where is the saturation knee, and what load satisfies an SLO? |

E001 and E002 study isolated axes. E003 and E004 move closer to realistic
serving workloads, where requests are not all the same size.

The progression is:

```text
one fixed request shape
        ↓
many concurrent copies of that shape
        ↓
different application-shaped distributions
        ↓
same mean, different variance
        ↓
external arrival rate and saturation
```

The most important conceptual shift is this:

> A production workload is not “a 4K-token request.” It is a distribution of
> prompt lengths, generation lengths, arrival times, and reuse patterns.

E003 introduces token-shape distributions. E004 proves that even the mean of
those distributions is not enough.

---

## 2. A Request-Shape Model

Represent request `i` as:

```text
R_i = (P_i, O_i)
```

where:

- `P_i` is the target input-token count,
- `O_i` is the maximum output-token count.

For a collection of requests, the workload is a distribution:

```text
W = distribution over (P, O)
```

In this project, E003 and E004 configure the input and output distributions
separately. They are sampled independently, so the framework does not force a
long prompt to have either a long or short output.

That choice is useful for controlled experiments, but it is also a limitation.
Real applications often have correlated shapes:

- a long retrieved context may lead to a short factual answer,
- a long document may lead to a long summary,
- a short creative-writing prompt may lead to a very long output.

Later experiments or production traces should model the joint distribution
`Pr(P, O)`, not only `Pr(P) × Pr(O)`.

### Target, cap, and actual counts

Three values must not be confused:

1. **Target input tokens**: requested synthetic prompt length.
2. **Maximum output tokens**: an upper bound sent to vLLM.
3. **Actual output tokens**: what the model generated before stopping.

The model can emit an end-of-sequence token before reaching the cap. Therefore:

```text
actual_output_tokens ≤ max_output_tokens
```

E003 demonstrates this directly. The chat profile requested output caps totaling
4,128 tokens but generated 3,974 actual tokens. RAG and summarization reached
their caps in that run. E004 reached every cap in both profiles.

This distinction matters because output throughput must use actual generated
tokens, not requested caps.

---

## 3. Metrics You Must Keep Separate

### Time to First Token

```text
TTFT = first_token_time - request_start_time
```

TTFT includes:

- client/server overhead,
- scheduler waiting,
- prompt prefill,
- interference from concurrent work.

It is not a pure kernel-level prefill measurement once concurrency is greater
than one.

### Time Per Output Token

For a successful request with more than one output token:

```text
TPOT = (completion_time - first_token_time) / (actual_output_tokens - 1)
```

The first token belongs to TTFT, so the denominator is `N - 1`.

### End-to-End latency

```text
E2E = completion_time - request_start_time
```

For an individual request, the familiar approximation is:

```text
E2E ≈ TTFT + (actual_output_tokens - 1) × TPOT
```

At concurrency, each component includes scheduling and batching effects, but the
decomposition still helps explain why long input primarily affects TTFT and long
output primarily affects E2E.

### Request throughput

```text
request_throughput = successful_requests / benchmark_wall_time
```

Request throughput answers: “How many whole jobs finish each second?” It is
strongly affected by the amount of work in each job.

### Input-, output-, and total-token throughput

```text
input_token_throughput  = total_actual_input_tokens / wall_time
output_token_throughput = total_actual_output_tokens / wall_time
total_token_throughput  = (input_tokens + output_tokens) / wall_time
```

These are not interchangeable:

- Input tok/s measures prompt-processing volume.
- Output tok/s measures generated-token delivery.
- Total tok/s combines computationally different prefill and decode tokens.

A system can have high total tok/s because it processes large prompts while
still providing low request throughput and poor user latency.

### Percentiles

- P50 is the median request.
- P95 is a tail metric: 95% of observations are at or below it.
- P99 focuses further into the tail.

Percentiles describe the observed sample, not a timeless property of the
server. With 50 requests, P99 lies near the largest observation. With only 10
requests in an E004 tail bucket, P95 and P99 are descriptive interpolations and
should not be treated as precise population estimates.

### Error rate

```text
error_rate = failed_requests / total_requests
```

A latency number based only on successful requests can look excellent while
slow requests time out and disappear. Always read success count, error rate, and
latency together.

---

## 4. Load Model: Closed Loop

E003 and E004 use closed-loop concurrency 4. Four workers submit requests. When
one finishes, that worker immediately submits the next request.

```text
worker 1: request → wait → completion → next request
worker 2: request → wait → completion → next request
worker 3: request → wait → completion → next request
worker 4: request → wait → completion → next request
```

This guarantees at most four in-flight requests. It is useful for controlled
comparisons, but it self-throttles: if the server becomes slower, clients wait
longer and automatically send requests less frequently.

Therefore E003/E004 measure performance at a controlled concurrency, not at a
controlled arrival rate. They cannot answer:

- How many requests per second can the server sustainably accept?
- At what offered load does the queue grow without bound?
- What request rate meets a TTFT or E2E SLO?

Those are E005 questions.

---

## 5. E003 — Application-Shaped Workloads

### Research question

> How does application request shape affect performance when the model,
> hardware, backend, and concurrency remain unchanged?

E003 compares three synthetic profiles. The names are interpretations of token
shape, not implementations of actual products.

| Profile | Input distribution | Output-cap distribution | Intended shape |
| :--- | :--- | :--- | :--- |
| Chat | 128/256/512/1024 | 32/64/128/256 | Short prompt, short response |
| RAG | 2048/4096/8192 | 128/256/512 | Long context, moderate response |
| Summarization | 2048/4096/8192 | 256/512/1024 | Long context, long response |

The canonical configurations are:

- [`e003_chat_like.yaml`](../configs/e003_chat_like.yaml)
- [`e003_rag_like.yaml`](../configs/e003_rag_like.yaml)
- [`e003_summarization_like.yaml`](../configs/e003_summarization_like.yaml)

### Controlled variables

All profiles use:

- Qwen2.5-7B-Instruct,
- BF16 on one RTX 3090,
- vLLM with prefix caching and chunked prefill disabled,
- closed-loop concurrency 4,
- 5 warm-up and 50 measured requests,
- temperature 0.0 and seed 42,
- the same server process and configuration.

The independent variable is the input/output token distribution.

### Why disable prefix caching?

Repeated prefixes could allow one profile to avoid prefill work through cache
hits. That would confound request shape with cache reuse. E003 generates
distinct deterministic prompts and disables prefix caching.

### Why disable chunked prefill?

Chunked prefill divides large prefills into smaller scheduling units. That is a
valuable production feature, but it changes the interference pattern. E003
disables it to establish a simpler baseline in which long prefills remain
visible.

### IID sampling

E003 uses independent identically distributed draws from each configured
finite distribution. For value set `x_j` and normalized weights `p_j`:

```text
Pr(X = x_j) = p_j
```

The same seed reproduces the same ordered request plan. A 50-request sample does
not have to match the theoretical mean exactly; that finite-sample deviation is
part of IID sampling.

The exact realized plan is saved to `workload.jsonl`, so the experiment can be
audited after the GPU instance is destroyed.

---

## 6. Reading the E003 Results

Canonical run: [`E003_20260924_051307_29213e50`](../runs/E003_20260924_051307_29213e50/)

| Profile | Req/s | Input tok/s | Output tok/s | Total tok/s | TTFT P50 | E2E P50 | E2E P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat | 1.566 | 577.28 | 124.46 | 701.74 | 216.50 ms | 1.810 s | 4.891 s |
| RAG | 0.372 | 1,815.57 | 84.87 | 1,900.44 | 2,026.88 ms | 8.676 s | 20.965 s |
| Summarization | 0.218 | 1,082.44 | 112.94 | 1,195.38 | 2,131.41 ms | 17.293 s | 33.633 s |

All 150 measured requests succeeded.

### Result 1: small jobs maximize request throughput

Chat completed:

```text
1.566 / 0.372 = 4.21× the RAG request rate
1.566 / 0.218 = 7.18× the summarization request rate
```

This does not mean chat uses the GPU “better” in every sense. Each chat request
simply contains much less work, so closed-loop workers recycle faster.

### Result 2: long prompts raise TTFT

Median TTFT relative to chat:

```text
RAG:           2026.88 / 216.50 = 9.36×
Summarization: 2131.41 / 216.50 = 9.84×
```

RAG and summarization have similar realized input distributions, so their
median TTFT is similar. Their output distributions differ much more strongly,
which primarily appears in E2E latency.

### Result 3: long generation dominates E2E

Summarization median E2E was:

```text
17.293 / 8.676 = 1.99× RAG
17.293 / 1.810 = 9.55× chat
```

RAG and summarization start with similar prefill work. Summarization then
performs many more sequential decode steps.

### Result 4: “tokens per second” needs a label

RAG had the highest total-token throughput at 1,900.44 tok/s, but the lowest
output-token throughput at 84.87 tok/s.

Why? RAG supplied many input tokens. Prefill processes those tokens in parallel,
so the server counted a large volume of input work each second. That did not
translate into faster response delivery.

This is a common benchmark-reporting failure:

```text
“The server achieved 1,900 tokens/s.”
```

Without saying whether those are input, output, or combined tokens, the number
is ambiguous and can mislead capacity decisions.

### Result 5: workload shape affects memory

Peak VRAM:

- Chat: 18,060 MiB
- RAG: 21,024 MiB
- Summarization: 21,024 MiB

The model weights are unchanged. The increase is consistent with longer active
sequences requiring more KV-cache blocks and workspace. Equal model size does
not imply equal serving memory demand.

### Operational interpretation

If your service resembles:

- **chat**: prioritize low TTFT, request overhead, and short-burst efficiency;
- **RAG**: control retrieved-context size and monitor TTFT tails;
- **summarization**: output length and E2E latency dominate user waiting time;
- **mixed traffic**: one aggregate mean cannot represent all user experiences.

The full empirical analysis is in
[`e003_application_workload_shapes_validation.md`](../outputs/e003_application_workload_shapes_validation.md).

---

## 7. Why E003 Alone Is Not Enough

E003 changes both mean and variance across profiles. Chat, RAG, and
summarization perform different total amounts of work. If their performance
differs, several explanations are possible:

- different mean input length,
- different mean output length,
- different variance,
- different category proportions,
- interactions among all of the above.

E004 asks a sharper causal question:

> If mean input and output work are held constant, does variance alone change
> the performance distribution?

This is experimental control. E003 establishes ecological variety; E004 isolates
one property of that variety.

---

## 8. E004 — Same Mean, Different Variance

### Workload construction

| Property | Fixed | Variable |
| :--- | :--- | :--- |
| Input values | 4096 | 1024 / 4096 / 7168 |
| Input weights | 1.0 | 0.2 / 0.6 / 0.2 |
| Output caps | 512 | 128 / 512 / 896 |
| Output weights | 1.0 | 0.2 / 0.6 / 0.2 |

The configurations are:

- [`e004_fixed.yaml`](../configs/e004_fixed.yaml)
- [`e004_variable.yaml`](../configs/e004_variable.yaml)

### Proving the means match

For a discrete distribution:

```text
μ = Σ p_j x_j
```

Variable input mean:

```text
μ_input = 0.2(1024) + 0.6(4096) + 0.2(7168)
        = 204.8 + 2457.6 + 1433.6
        = 4096
```

Variable output-cap mean:

```text
μ_output = 0.2(128) + 0.6(512) + 0.2(896)
         = 25.6 + 307.2 + 179.2
         = 512
```

The fixed profile has those exact values for every request.

### Variance and coefficient of variation

Population variance is:

```text
σ² = Σ p_j (x_j - μ)²
```

Standard deviation is `σ = sqrt(σ²)`. Coefficient of variation is:

```text
CV = σ / μ
```

CV is dimensionless, so it helps compare relative dispersion between input and
output distributions with different scales.

For E004:

- fixed input/output CV = 0,
- variable input CV = 0.4743,
- variable output CV = 0.4743.

The means are identical; the variance is the independent variable.

### Why stratified sampling was added

With ordinary IID sampling, 50 random draws may not produce exactly the desired
20/60/20 proportions. A mean mismatch would reintroduce a confounder.

E004 uses deterministic stratified sampling. For 50 requests:

```text
50 × [0.2, 0.6, 0.2] = [10, 30, 10]
```

The values are expanded to those exact counts and deterministically shuffled.
Input and output plans are shuffled independently.

Both canonical profiles therefore realized:

- mean input = 4,096 with zero sampling error,
- mean output cap = 512 with zero sampling error,
- total actual input = 204,800 tokens,
- total actual output = 25,600 tokens.

Warm-up and measured plans are stratified separately so removing warm-up
requests cannot distort the measured distribution.

---

## 9. Reading the E004 Results

Canonical run: [`E004_20260924_065119_c3c70a46`](../runs/E004_20260924_065119_c3c70a46/)

| Profile | Req/s | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 0.2816 | 2,582.03 ms | 3,368.47 ms | 3,423.06 ms | 13.789 s | 13.890 s | 13.910 s |
| Variable | 0.2823 | 1,640.65 ms | 3,848.66 ms | 3,979.81 ms | 13.436 s | 24.526 s | 25.543 s |

All 100 measured requests succeeded.

### The headline result

Throughput was essentially identical, but the latency distributions were not.

E2E tail increase:

```text
P95 increase = (24.526 / 13.890 - 1) × 100 = 76.6%
P99 increase = (25.543 / 13.910 - 1) × 100 = 83.6%
```

TTFT tail increase:

```text
P95 increase = (3848.66 / 3368.47 - 1) × 100 = 14.3%
P99 increase = (3979.81 / 3423.06 - 1) × 100 = 16.3%
```

### Why did the variable median improve?

Variable TTFT P50 was 36.5% lower than fixed. This is not contradictory.

Twenty percent of variable prompts contain only 1,024 tokens, so they can reach
their first token quickly. Those short requests pull the middle of the observed
distribution downward. Simultaneously, the 7,168-token prompts and mixed
concurrent work stretch the upper tail.

This produces the central E004 pattern:

```text
better median + worse tail + unchanged throughput
```

If a dashboard showed only average tokens, throughput, and P50, the variable
workload could appear equal or better. Users in its tail would experience much
worse latency.

### Tail amplification ratios

Fixed E2E:

```text
P99 / P50 = 13.910 / 13.789 = 1.009
```

Variable E2E:

```text
P99 / P50 = 25.543 / 13.436 = 1.901
```

Fixed requests are uniform, so completion times cluster tightly. Variable
requests require different amounts of work and overlap in different batch
compositions, so completion times spread out.

### The throughput hypothesis was rejected

E004 predicted that irregular request sizes might reduce batching efficiency.
Observed request throughput was:

```text
fixed    = 0.2816 req/s
variable = 0.2823 req/s
```

The 0.25% difference is not evidence of a meaningful improvement. It shows that
at concurrency 4, on this hardware, with equal total token work, heterogeneity
did not materially change aggregate runtime.

This negative result is valuable. Systems experiments should update beliefs,
not force every hypothesis to be confirmed.

---

## 10. Length-Bucket Analysis

Aggregate percentiles tell you that a tail exists. Buckets help explain where it
appears.

### Input-length buckets

| Input target | N | TTFT P50 | TTFT P95 | TTFT P99 |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 10 | 353.73 ms | 1,942.87 ms | 1,981.20 ms |
| 4,096 | 30 | 1,392.38 ms | 3,627.34 ms | 3,928.92 ms |
| 7,168 | 10 | 1,798.69 ms | 3,933.24 ms | 3,978.45 ms |

Median TTFT increases with prompt length, as expected from prefill work.

The short bucket is especially instructive:

```text
short-input TTFT P95 / P50 = 1942.87 / 353.73 = 5.49×
```

A 1,024-token prompt can still wait behind or overlap with more expensive work.
Its own token length does not fully determine its observed TTFT.

### Output-length buckets

| Output cap | N | TPOT P50 | E2E P50 | E2E P95 | E2E P99 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10 | 20.93 ms | 4.915 s | 6.318 s | 6.611 s |
| 512 | 30 | 23.65 ms | 13.436 s | 15.387 s | 15.678 s |
| 896 | 10 | 24.51 ms | 24.459 s | 25.618 s | 26.292 s |

Median E2E ratio:

```text
24.459 / 4.915 = 4.98×
```

TPOT changed much less than E2E. The long-output requests are slow primarily
because they perform more sequential decode steps, not because each step became
five times slower.

### Head-of-line interference: what can we claim?

When chunked prefill is disabled, a long prefill can occupy substantial GPU
work before other requests receive first tokens. Long decodes also remain active
for more iterations, changing the batch seen by shorter requests.

The observed short-request tail is consistent with head-of-line and scheduling
interference. It is not causal proof because:

- requests overlap dynamically,
- input and output lengths are independently assigned,
- only one deterministic order was tested,
- no scheduler trace identifies exactly which request blocked another.

A stronger causal experiment would replay the same short requests:

1. alone,
2. mixed only with short requests,
3. mixed with long-prefill requests,
4. mixed with long-decode requests,

while capturing scheduler-level queue and batch traces.

The full E004 analysis is in
[`e004_fixed_vs_variable_validation.md`](../outputs/e004_fixed_vs_variable_validation.md).

---

## 11. Code Walkthrough: Configuration to Evidence

The data path is:

```text
YAML profile
   ↓ load and validate
WorkloadConfig + TokenLengthDistribution
   ↓ deterministic request-plan generation
RequestSpec[]
   ↓ exact-token prompt generation
vLLM streaming requests
   ↓ monotonic timestamps + tokenizer counts
RequestMeasurement[]
   ↓ aggregate and bucket analysis
JSON/JSONL summaries + plots
```

### 11.1 Distribution model

[`workloads/spec.py`](../src/inference_os/workloads/spec.py) defines:

- `RequestSpec(target_input_tokens, max_output_tokens)`,
- `TokenLengthDistribution(values, weights)`,
- `WorkloadConfig`,
- IID and stratified generation.

Validation rejects empty distributions, non-positive token lengths, negative or
non-finite weights, and unsupported sampling modes.

### 11.2 Request plan generation

`generate_request_specs` accepts a request count, seed, input distribution,
output distribution, and sampling mode.

- IID mode preserves E003's original interleaved random draws.
- Stratified mode computes category counts, expands them, and shuffles them.

The seed affects order, not E004 canonical category counts.

### 11.3 Benchmark engine

[`runner/engine.py`](../src/inference_os/runner/engine.py):

1. builds the complete plan before measurement,
2. generates exact-token synthetic prompts,
3. assigns each request its output cap,
4. streams results from vLLM,
5. counts actual output tokens with the tokenizer,
6. samples GPU telemetry,
7. persists the full run.

Generating the plan before timing prevents client-side randomness and prompt
construction from contaminating request latency.

### 11.4 Request timing

[`metrics/request.py`](../src/inference_os/metrics/request.py) stores monotonic
nanosecond timestamps and derives TTFT, TPOT, and E2E. Monotonic time is required
because wall-clock time can jump due to clock synchronization.

### 11.5 Aggregate summaries

[`metrics/summary.py`](../src/inference_os/metrics/summary.py) calculates sample
standard deviation, percentiles, token totals, throughput, and error rate.

### 11.6 Bucket summaries

[`metrics/buckets.py`](../src/inference_os/metrics/buckets.py) joins persisted
request measurements with workload records by request ID. It groups requests by
exact target input length or maximum output length and computes metrics for each
bucket.

Joining by request ID is safer than relying only on file order.

### 11.7 Persistence

[`results/persistence.py`](../src/inference_os/results/persistence.py) writes:

- `config.json`: requested experiment configuration,
- `environment.json`: hardware, software, Git metadata,
- `summary.json`: aggregate performance and workload diagnostics,
- `workload.jsonl`: exact planned shape for every request,
- `requests.jsonl`: raw measured outcome for every request,
- `telemetry.jsonl`: time-series GPU observations.

For E004, workload summaries additionally record configured distributions,
population standard deviation, CV, realized means, and sampling error.

### 11.8 Experiment orchestration

- [`run_e003.py`](../experiments/E003-application-workload-shapes/run_e003.py)
  validates common controls and compares three application-shaped profiles.
- [`run_e004.py`](../experiments/E004-fixed-vs-variable/run_e004.py) additionally
  requires exactly one fixed and one variable profile, matched configured means,
  and stratified sampling.

Both runners mark a profile successful only when every measured request
succeeds. This prevents a comparison with silent timeout loss from being labeled
`SUCCESS`.

### 11.9 Plot generation

[`reports/plots.py`](../src/inference_os/reports/plots.py) generates aggregate,
distribution, bucket, throughput, and GPU figures directly from structured
summaries. Plots are derived artifacts; JSON/JSONL files remain the evidence.

---

## 12. How to Audit a Run

Never begin with the PNGs. Use this order:

1. **Check provenance**
   - source commit,
   - branch and dirty state,
   - model and served-model identity,
   - Python, PyTorch, CUDA, vLLM, driver, and GPU.
2. **Check experiment status**
   - profile count,
   - successful request count,
   - error types and error rate.
3. **Check workload realization**
   - requested distributions,
   - realized counts and means,
   - sampling error,
   - actual output tokens versus caps.
4. **Check controls**
   - concurrency,
   - warm-up count,
   - caching/chunked-prefill settings,
   - temperature and seed.
5. **Read aggregate metrics**
   - throughput,
   - P50/P95/P99,
   - GPU utilization and memory.
6. **Inspect buckets and raw requests**
   - bucket sample sizes,
   - outliers,
   - relationship between shape and latency.
7. **Use plots last**
   - confirm they accurately summarize the structured data.

The canonical artifacts are:

- [E003 raw run](../runs/E003_20260924_051307_29213e50/)
- [E004 raw run](../runs/E004_20260924_065119_c3c70a46/)

---

## 13. The ReadTimeout Lesson

The first E003 canonical attempt recorded `ReadTimeout` failures for long
profiles. The server had not necessarily failed. The HTTP client's short default
read-inactivity timeout could expire while a request waited in the queue or
performed a long prefill before emitting its first streamed bytes.

The correction was to configure a 300-second request-activity timeout and to
make experiment success require zero failed measured requests.

The systems lesson is broader:

> A client timeout is part of the benchmark definition.

If the timeout is too short, the client censors the slow tail. If it is
unbounded, genuine hangs can consume the run indefinitely. Record it, choose it
deliberately, and always report failures beside latency.

Do not interpret a timeout automatically as GPU OOM, server crash, or overload.
Inspect the error type, server logs, request timing, and GPU telemetry.

---

## 14. Common Interpretation Mistakes

### Mistake 1: “RAG is fastest because total tok/s is highest”

RAG processed input tokens fastest in E003. It had worse request latency and the
lowest output-token throughput. “Fastest” needs a metric and user objective.

### Mistake 2: “The average request is 4K/512, so fixed and variable are equal”

They are equal in aggregate work, not in completion-time distribution. E004 is
the counterexample.

### Mistake 3: “Variable P50 improved, so variability helped”

The distribution gained short requests and long requests. P50 improved while
P95/P99 degraded severely. Choose the percentile tied to the product SLO.

### Mistake 4: “P99 from 10 observations is a reliable production P99”

It is not. It is a descriptive statistic from a small deterministic bucket.
Production tail estimation requires many more observations and repeated runs.

### Mistake 5: “A maximum output cap is an actual output length”

The model may stop early. Use tokenizer-observed output counts.

### Mistake 6: “High GPU utilization means good user performance”

Both E004 profiles were near 99% GPU utilization. One had an E2E P99 83.6%
higher. Utilization measures activity, not service quality.

### Mistake 7: “Closed-loop throughput is server capacity”

Closed-loop clients slow their arrival rate when the server slows. Sustainable
offered load requires open-loop arrivals and queue/SLO measurements.

### Mistake 8: “Bucket correlation proves scheduler causality”

Buckets expose patterns. Proving a scheduling mechanism requires stronger
instrumentation or an intervention that isolates that mechanism.

---

## 15. Applied Decision Framework

When evaluating an inference workload, ask these questions in order.

### Workload

- What are the input and actual output distributions?
- Are input and output lengths correlated?
- Are there distinct application classes?
- Are prompts or prefixes reused?
- What is the arrival process?

### User objective

- Is first-token responsiveness most important?
- Is full completion latency important?
- Which percentile defines acceptable service?
- Are failures included in the SLO?

### Capacity

- Is demand measured in requests/s, input tok/s, output tok/s, or all three?
- Does the workload mix change throughout the day?
- Does one long-request class harm another class's tail?

### Optimization direction

| Symptom | Likely axis to investigate |
| :--- | :--- |
| High TTFT for long prompts | Prefill, context size, chunked prefill, queueing |
| High E2E for long outputs | Decode steps, output caps, memory bandwidth |
| Good P50 but poor P99 | Heterogeneity, queueing, scheduling interference |
| High total tok/s but low req/s | Large input workload; inspect metric composition |
| Timeouts with healthy server | Client timeout versus queue/prefill inactivity |
| High GPU utilization and poor SLO | Saturation or inefficient workload mix |

---

## 16. Preparing for E005

E003 answers “which shapes behave differently?” E004 answers “does variance
matter when means match?” Neither applies an external arrival rate.

E005 must introduce:

```text
scheduled arrival time → actual dispatch → queue delay → service → completion
```

### New concepts

- **Offered load**: requests scheduled per second, independent of completions.
- **Achieved throughput**: requests actually completed per second.
- **Queue delay**: dispatch/service start minus scheduled arrival.
- **Saturation knee**: the load where throughput gains diminish and queue/latency
  begin rising rapidly.
- **SLO**: a concrete objective such as `TTFT P95 ≤ 2 s` and error rate ≤ 1%.
- **Goodput**: completed requests per second that satisfy the SLO.

### Why heterogeneity matters even more near saturation

Below saturation, spare capacity can absorb bursts and large requests. Near
saturation, long requests consume service capacity while new arrivals continue.
Queues grow, and variance can amplify waiting time. Queueing theory predicts
that service-time variability matters, not only mean service time.

E004 provides the empirical reason to carry full request-shape distributions
into E005 rather than replacing them with one average request.

---

## 17. Interview-Ready Explanations

### Explain E003 in 30 seconds

> We held model, GPU, vLLM settings, and concurrency constant and changed only
> synthetic input/output token distributions representing chat, RAG, and
> summarization. Chat delivered the most requests per second, long-input profiles
> had roughly 9–10× median TTFT, and summarization had the worst E2E latency.
> RAG had the highest total-token throughput because input tokens dominated,
> showing why input and output throughput must be reported separately.

### Explain E004 in 30 seconds

> We compared a fixed 4K/512 workload with a variable workload whose input and
> output means and total token work matched exactly. Throughput was virtually
> identical, but variable E2E P95 and P99 rose about 77% and 84%. The median did
> not reveal the problem. This demonstrates that average sequence length predicts
> aggregate work better than it predicts user-facing tail latency.

### Explain why this is not a capacity result

> Both experiments use a closed-loop worker pool. When latency rises, clients
> wait and automatically reduce their send rate. To measure sustainable capacity,
> we need open-loop arrivals, explicit SLOs, queue delay, and goodput across an
> offered-load sweep.

---

## 18. Exercises

Try answering before reading the solutions.

### Exercise 1: Metric choice

A RAG server reports 2,000 total tok/s and 90 output tok/s. A chat server reports
800 total tok/s and 130 output tok/s. Which is faster?

### Exercise 2: Matched mean

Verify that `[1024, 4096, 7168]` with weights `[0.2, 0.6, 0.2]` has mean 4,096.
Why would a 50-request IID sample still potentially have a different mean?

### Exercise 3: Tail change

Calculate the percentage increase from fixed E2E P99 13.910 s to variable E2E
P99 25.543 s.

### Exercise 4: Decode reasoning

Why does the 896-output bucket have about 5× the median E2E of the 128-output
bucket even though median TPOT differs by only about 17%?

### Exercise 5: Causality

Does the 1,024-input bucket's large TTFT tail prove that a 7,168-token request
blocked it? What additional evidence would you want?

### Exercise 6: Experiment validity

Suppose the fixed profile ran with prefix caching enabled while the variable
profile ran with it disabled. Could you attribute a latency difference to
variance?

### Exercise 7: Timeout diagnosis

A request produces no bytes for six seconds during a long prefill and the client
has a five-second read timeout. Is this necessarily a server failure?

### Exercise 8: E005 design

Why can E004's 0.282 req/s not be declared the RTX 3090's sustainable request
capacity for this workload?

---

## 19. Exercise Solutions

### Solution 1

The question is underspecified. RAG processes more combined token work, while
chat delivers more generated tokens. Compare request throughput and latency for
user responsiveness, output tok/s for generation delivery, and input tok/s for
prefill capacity. Never call an unlabeled combined tok/s value simply “faster.”

### Solution 2

```text
0.2(1024) + 0.6(4096) + 0.2(7168) = 4096
```

IID sampling guarantees the distribution only in expectation. Finite random
samples can contain too many short or long values. Stratified sampling fixes the
category counts for E004's canonical comparison.

### Solution 3

```text
(25.543 / 13.910 - 1) × 100 = 83.6%
```

### Solution 4

E2E includes the number of sequential decode iterations. The 896-token request
performs roughly seven times as many capped output steps as the 128-token
request. TPOT measures cost per step; output length controls how many steps are
performed.

### Solution 5

No. The result is consistent with interference but does not identify the
specific concurrent requests or scheduler decisions. Useful evidence would
include request overlap timelines, scheduler queue traces, batch membership,
and controlled replays with and without long-prefill neighbors.

### Solution 6

No. Cache configuration would be a confounder. Both profiles must use identical
serving controls so variance remains the independent variable.

### Solution 7

No. The server may still be queueing or computing prefill. A read-inactivity
timeout measures absence of network bytes, not server death. Check server logs,
GPU telemetry, and use a documented timeout suitable for the workload.

### Solution 8

It is achieved throughput under closed-loop concurrency 4. The arrival rate
falls automatically when requests slow down. Sustainable capacity requires an
open-loop offered-load sweep and an SLO definition.

---

## 20. Glossary

**Application-shaped workload**  
A synthetic token distribution intended to resemble a class of application
requests without claiming to reproduce a production trace.

**Bucket analysis**  
Grouping requests by a discrete property, such as input length, and calculating
metrics within each group.

**Closed-loop load**  
A client model where a worker waits for completion before submitting replacement
work.

**Coefficient of variation (CV)**  
Standard deviation divided by mean; a scale-independent dispersion measure.

**Configured distribution**  
The values and probabilities declared before sampling.

**E2E latency**  
Time from request start until completion.

**Goodput**  
The rate of successful requests that also satisfy an SLO.

**Head-of-line interference**  
Delay experienced by work waiting behind or sharing constrained service with
more expensive work.

**Heterogeneous workload**  
A workload containing requests with different shapes or service demands.

**IID sampling**  
Independent draws from the same probability distribution.

**Maximum output tokens**  
A generation cap, not a promise that the model will emit that many tokens.

**Open-loop load**  
A client model where arrivals follow an external schedule regardless of prior
completion.

**Realized workload**  
The exact ordered request plan produced from a configured distribution and seed.

**Saturation knee**  
The load region where additional offered work stops producing proportional
throughput and causes queueing or latency to rise sharply.

**SLO**  
A measurable service-level objective, usually defined by latency, success rate,
and a target percentile.

**Stratified sampling**  
Sampling that controls category counts to reduce finite-sample composition error.

**Throughput**  
Completed work per unit time; always specify whether work means requests, input
tokens, output tokens, or combined tokens.

**TPOT**  
Average time per generated token after the first output token.

**TTFT**  
Time from request start until the first output token is observed.

---

## 21. Final Mental Model

Keep these five statements:

1. **Request shape determines where work occurs.** Input length drives prefill
   and TTFT; output length drives decode steps and E2E.
2. **Throughput needs a unit and token direction.** Requests/s, input tok/s, and
   output tok/s answer different capacity questions.
3. **The mean predicts aggregate work better than experience.** Two mean-matched
   workloads can have nearly identical throughput and radically different P99.
4. **Heterogeneity is a scheduling problem as well as a size problem.** A
   request's latency depends on its own shape and concurrent neighbors.
5. **Closed-loop experiments are not capacity tests.** E005 must add external
   arrivals, queues, SLOs, and goodput.

In one sentence:

> E003 teaches that application token shapes change inference behavior; E004
> teaches that the distribution around the average determines who suffers in
> the tail.
