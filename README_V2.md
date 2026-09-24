# inference_os V2 — Application-Shaped Workloads and Serving Capacity

V2 evolves `inference_os` from controlled single-shape experiments into a
framework for studying heterogeneous, application-shaped inference workloads
and the serving capacity required to handle them.

V1 established trustworthy request-level measurement and studied prompt length,
output length, and concurrency independently. V2 builds on that foundation. It
does not rewrite V1, replace its experiment model, or turn the project into an
application framework.

The central progression is:

```text
V1: measurement → isolated scaling → concurrency
V2: workload distributions → heterogeneity → load → capacity → economics
```

---

# V2 Objective

The engineering goal of V2 is to make `inference_os` capable of answering:

> How do the request shapes produced by different application categories affect
> inference performance, tail latency, sustainable load, and serving cost when
> the model and serving stack remain unchanged?

V2 models the inference requests produced by applications. It does not build the
applications themselves.

For example:

- a chat-like workload may contain mostly short prompts and short responses,
- a RAG-like workload may contain long prompts and moderate responses,
- a summarization-like workload may contain long prompts and long responses.

These names describe synthetic workload shapes. They do not claim to reproduce
any specific production application or traffic trace.

---

# Why V2 Exists

V1 deliberately varied one dimension at a time. That was necessary to isolate
and explain the effects of prefill, decode, and concurrency.

Real serving workloads are heterogeneous:

- requests do not all contain the same number of input tokens,
- requests do not all request the same number of output tokens,
- long and short requests can execute concurrently,
- averages can hide expensive requests and tail-latency problems,
- traffic can continue arriving even when the server is already busy.

A server that performs well on a uniform workload may behave differently when
request lengths vary. V2 introduces this variability in controlled stages so
that its effects remain measurable and explainable.

---

# What Success Looks Like

A successful V2 should make it possible to:

1. define deterministic input- and output-length distributions,
2. generate an auditable request plan from a configuration and random seed,
3. run fixed-size V1 workloads through the same workload mechanism,
4. preserve target and actual token counts separately,
5. compare synthetic application-shaped profiles,
6. explain why heterogeneous workloads affect tail latency and throughput,
7. generate open-loop request arrivals at a controlled offered rate,
8. identify the saturation point of a serving configuration,
9. evaluate throughput and goodput against explicit latency objectives,
10. derive narrowly scoped, documented serving-cost estimates.

V2 is successful when these capabilities are reproducible and supported by
experiments, not merely when new configuration options exist.

---

# V2 Conceptual Model

An application-shaped workload is characterized by four properties:

1. input token-length distribution,
2. requested maximum output token-length distribution,
3. request load pattern,
4. prompt-reuse characteristics recorded as workload metadata.

The minimum request-level abstraction is:

```python
@dataclass(frozen=True)
class RequestSpec:
    target_input_tokens: int
    max_output_tokens: int
```

`max_output_tokens` is a request limit, not a guarantee. A model may terminate
early. The benchmark must therefore keep the requested maximum separate from
the actual tokenizer-observed output count.

The first distribution model may sample input and output lengths independently.
This assumption must be documented. Correlated input/output shapes should only
be added when an experiment demonstrates that independent sampling is
insufficient.

---

# Example Workload Configuration

The exact schema should follow the existing configuration system. Conceptually,
a profile may look like:

```yaml
experiment_id: E003_RAG
model: Qwen/Qwen2.5-7B-Instruct
base_url: http://localhost:18000
num_requests: 500
warmup_requests: 5
seed: 42

workload:
  name: rag_like
  input_tokens:
    values: [2048, 4096, 8192]
    weights: [0.20, 0.55, 0.25]
  max_output_tokens:
    values: [128, 256, 512]
    weights: [0.45, 0.40, 0.15]
  prompt_reuse:
    mode: none

load:
  mode: closed_loop
  concurrency: 4
```

For the initial implementation:

- weights may be normalized internally and need not sum to exactly `1.0`,
- all token lengths must be positive,
- values and weights must have matching lengths,
- only `prompt_reuse.mode: none` is supported,
- unsupported modes must fail validation rather than being silently ignored.

---

# Workload Reproducibility Requirements

## 1. Deterministic Request Plans

The same workload configuration and seed must produce the same ordered sequence
of `RequestSpec` values:

```python
generate(config, seed=42) == generate(config, seed=42)
```

Changing the seed should normally produce a different sequence.

## 2. Realized Workload Persistence

Every benchmark run must persist the exact plan that was executed:

```text
runs/<run-id>/workload.jsonl
```

Example records:

```json
{"request_id":"warmup-1","is_warmup":true,"target_input_tokens":2048,"max_output_tokens":128}
{"request_id":"req-1","is_warmup":false,"target_input_tokens":4096,"max_output_tokens":256}
```

Request identifiers must match the corresponding records in `requests.jsonl`.
Warm-up requests are part of the executed workload and must also be recorded.

## 3. Target and Actual Values Remain Separate

`workload.jsonl` records what the benchmark requested. `requests.jsonl` records
what the tokenizer and server actually produced.

The framework must never silently substitute a requested token count for an
observed measurement.

## 4. Prompt Preparation Must Not Contaminate Measurements

Request-plan sampling, synthetic prompt construction, and tokenization must
happen before measured execution begins. Client-side prompt preparation must
not be included in request latency or interfere with dispatch timing.

## 5. No-Reuse Workloads Must Use Distinct Content

When `prompt_reuse.mode` is `none`, prompts should be deterministic but distinct,
for example by deriving each prompt seed from the run seed and request index.

Caching remains disabled and is not evaluated in V2.

---

# Backward Compatibility with V1

Existing E000–E002 configurations must continue to work unchanged.

A fixed-size V1 workload should be treated as a one-point distribution:

```text
prompt_tokens: 512       → values: [512], weights: [1.0]
max_output_tokens: 128   → values: [128], weights: [1.0]
```

This allows fixed and variable workloads to use the same execution and
persistence path without duplicating the benchmark runner.

V2 must not invalidate existing V1 reports or reinterpret their stored metrics.

---

# V2 Experiment Plan

## E003 — Application Workload Shapes

### Question

> How does application request shape affect inference performance when the
> underlying model and serving stack are unchanged?

Compare three synthetic profiles:

- `chat_like`,
- `rag_like`,
- `summarization_like`.

Hold constant:

- model,
- model revision,
- serving backend and server configuration,
- GPU and precision,
- concurrency,
- request count,
- warm-up policy,
- sampling parameters.

Study:

- TTFT P50/P95,
- TPOT P50/P95,
- E2E latency P50/P95,
- request throughput,
- actual input-token throughput,
- actual output-token throughput,
- total token throughput,
- GPU utilization,
- GPU memory,
- error rate.

### Engineering Capability Introduced

E003 requires:

- deterministic request-spec generation,
- weighted token-length distributions,
- request-specific prompt generation,
- request-specific output limits,
- realized-workload persistence,
- named configuration-driven profiles,
- workload distribution summaries.

### What E003 Does Not Prove

E003 does not prove that the synthetic profiles reproduce real production
traffic. It isolates the performance effects of qualitatively different token
shapes.

---

## E004 — Fixed versus Variable Workloads

### Question

> Why can workloads with similar average token counts have different tail
> latency and throughput?

Compare:

1. a fixed workload,
2. a heterogeneous workload with approximately the same mean input and output
   lengths.

The independent variable is workload variance. Mean request shape should remain
approximately controlled.

Study:

- P50, P95, and P99 latency,
- throughput,
- per-length-bucket latency,
- head-of-line interference,
- GPU utilization and memory,
- differences between aggregate means and distribution tails.

### Engineering Capability Introduced

E004 requires:

- comparison of workload distributions,
- target and realized distribution summaries,
- per-shape or per-length-bucket analysis,
- explicit reporting of sampling variance.

### Systems Question

> Why is average sequence length insufficient to predict serving performance?

---

## E005 — Open-Loop Load, Saturation, and SLO Goodput

### Question

> What request rate can an application-shaped workload sustain before latency
> objectives are violated?

V1 and the first V2 experiments use closed-loop load: a fixed number of workers
wait for requests to finish before sending replacements. This is useful for
controlled concurrency but self-throttles when the server slows down.

E005 introduces open-loop load. Requests arrive according to an external
schedule regardless of whether earlier requests have completed.

Start with deterministic constant-rate arrivals. Add Poisson arrivals only if a
specific experiment requires production-like arrival variability.

Study:

- offered request rate,
- achieved request throughput,
- concurrent requests over time,
- TTFT and E2E P50/P95/P99,
- timeout and error rates,
- queue growth,
- saturation point,
- SLO compliance,
- goodput.

Example SLOs:

```text
TTFT P95 <= 1 second
E2E P95 <= 5 seconds
error rate <= 1%
```

Goodput is defined as:

```text
completed requests satisfying the configured SLO / measured second
```

### Saturation Definition

Saturation is the region where additional offered traffic no longer produces a
corresponding increase in completed throughput and instead causes queues and
latency to grow.

Saturation must not be inferred from GPU utilization alone. It is an end-to-end
property of the serving system under a defined workload.

### Engineering Capability Introduced

E005 requires:

- duration-based runs,
- scheduled request arrivals,
- offered-load measurement,
- request dispatch timestamps,
- in-flight request tracking,
- timeouts and cancellation,
- configurable SLO evaluation,
- throughput-versus-goodput plots.

---

## E006 — Serving Economics

### Question

> How does workload shape affect the estimated cost of serving the same model at
> a measured sustainable load?

Derive narrowly scoped metrics such as:

- GPU-seconds per completed request,
- GPU-seconds per SLO-compliant request,
- estimated GPU cost per request,
- estimated GPU cost per 1,000 requests,
- estimated cost per one million output tokens.

All cost assumptions must be explicit configuration or report metadata:

- currency,
- GPU price per hour,
- billing granularity,
- whether idle time is included,
- workload and load point used,
- SLO definition.

These values are estimates for the measured configuration. E006 is not a general
capacity planner or cloud-pricing engine.

---

# V2 Metrics

V2 retains all V1 metrics and adds, when required by the experiments:

- target input-token distribution,
- target maximum-output-token distribution,
- actual input-token distribution,
- actual output-token distribution,
- input-token throughput,
- total token throughput,
- offered request rate,
- achieved request rate,
- in-flight request count over time,
- SLO compliance rate,
- goodput,
- GPU-seconds per completed request,
- estimated cost metrics.

Metric definitions must specify their numerator, denominator, measurement window,
eligible request set, and behavior when requests fail.

---

# Explicitly Out of Scope for V2

Do not add the following as part of V2:

- real chat application logic,
- a RAG retrieval pipeline,
- embedding models,
- vector databases,
- document ingestion,
- production conversation or summarization datasets,
- prefix or prompt caching experiments,
- speculative decoding,
- quantization comparisons,
- multiple serving backends,
- multi-GPU or distributed inference,
- Kubernetes or cloud provisioning,
- web dashboards,
- hosted services,
- databases for result storage,
- a plugin architecture,
- custom CUDA or Triton kernels,
- automatic capacity recommendations,
- general cloud-cost optimization.

These may become future experiments only when the measurement framework and a
specific research question justify them.

---

# V2 Engineering Principles

## 1. Extend V1; Do Not Rewrite It

Existing configurations, experiments, result artifacts, and metrics remain valid.
V2 should introduce the smallest extension needed for each new experiment.

## 2. Workload Plans Are First-Class Data

A seed is not a substitute for the realized request plan. Persist exactly what
was scheduled so a run can be audited independently of sampler implementation.

## 3. Workload Shape Is Not Application Semantics

Profile names are convenient labels for token distributions. They do not imply
that the benchmark executes the corresponding application or measures its
end-to-end quality.

## 4. Controlled Experiments Still Apply

E003 varies profile shape. E004 varies heterogeneity. E005 varies offered load.
Do not combine these independent variables until their individual effects are
understood.

## 5. Tail Behavior Matters

Means alone are insufficient for heterogeneous workloads. V2 reports percentiles
and explains which request shapes contribute to tail latency.

## 6. Capacity Is Defined Relative to a Workload and SLO

There is no single context-free capacity number. A capacity result is valid only
for its model, hardware, backend configuration, workload distribution, arrival
process, and latency objective.

## 7. Cost Results Must Expose Assumptions

Never report cost estimates without the price, utilization/load point, time
window, currency, and workload used to derive them.

---

# Planned Repository Additions

The exact names should follow the existing code during implementation, but V2 is
expected to require additions conceptually similar to:

```text
src/inference_os/
├── workloads/
│   ├── spec.py             # RequestSpec and distribution configuration
│   └── sampler.py          # Deterministic request-plan generation
├── runner/
│   └── load.py             # Open-loop scheduling, added only for E005
├── metrics/
│   ├── buckets.py          # Per-length workload summaries (added in E004)
│   └── slo.py              # SLO and goodput metrics, added only for E005
└── results/
    └── persistence.py      # workload.jsonl integration

configs/
├── e003_chat_like.yaml
├── e003_rag_like.yaml
├── e003_summarization_like.yaml
├── e004_fixed.yaml
├── e004_variable.yaml
└── e005_open_loop.yaml

experiments/
├── E003-application-workload-shapes/
├── E004-fixed-vs-variable/
├── E005-open-loop-saturation/
└── E006-serving-economics/
```

This is a planning structure, not a requirement to create every file in advance.
Directories should be added only when their experiment begins.

---

# V2 Milestones

## Milestone V2.0 — Workload Foundation

- define `RequestSpec`,
- define validated weighted token distributions,
- generate deterministic request plans,
- preserve fixed-size V1 behavior,
- generate distinct deterministic synthetic prompts for no-reuse profiles,
- persist `workload.jsonl`,
- keep target and actual counts separate,
- add unit and integration tests,
- document the configuration schema.

Do not implement E003 reports until this foundation is clean and tested.

## Milestone V2.1 — E003 Application Shapes

- add three documented synthetic profiles,
- run a pilot on the target GPU,
- run the canonical comparison,
- add input and total token throughput,
- produce plots and the E003 report.

## Milestone V2.2 — E004 Heterogeneity

- construct fixed and variable workloads with similar means,
- add distribution and length-bucket summaries,
- quantify tail-latency differences,
- produce the E004 report.

## Milestone V2.3 — E005 Sustainable Load

- add constant-rate open-loop scheduling,
- add duration-based execution and overload safeguards,
- define SLO configuration and goodput,
- identify the saturation knee,
- produce the E005 report.

## Milestone V2.4 — E006 Economics

- configure explicit GPU-hour cost assumptions,
- calculate benchmark-normalized cost metrics,
- relate cost to sustainable SLO-compliant capacity,
- document limitations and produce the E006 report.

---

# Current V2 Milestone

The current V2 milestone is:

## Milestone V2.2 — E004 Heterogeneity

Milestones V2.0 and V2.1 are complete. E003 preserved deterministic request
plans and compared chat-like, RAG-like, and summarization-like token shapes on
the canonical RTX 3090 environment. All 150 measured requests succeeded, and
the raw measurements, telemetry, plots, and validation report are preserved in
the repository.

E003 found that long-input profiles increased median TTFT by roughly 9–10×
relative to chat, while the long-output summarization profile produced the
highest E2E latency. It also demonstrated why input, output, and total-token
throughput must be reported separately.

The E004 implementation is now ready with mean-matched fixed and heterogeneous
profiles, deterministic stratified sampling, configured-versus-realized
distribution diagnostics, exact length-bucket summaries, P99 reporting, and
comparison plots. The next operational step is to run its pilot and canonical
GPU comparison, then preserve and analyze the resulting artifacts. Open-loop
scheduling, SLO evaluation, and cost metrics remain deferred until their
planned experiments.

---

# V2 Definition of Done

V2 is complete when:

1. existing E000–E002 tests and configurations remain valid,
2. deterministic application-shaped request plans can be generated and audited,
3. E003 explains performance differences among the three synthetic profiles,
4. E004 demonstrates how heterogeneity affects tail latency despite similar
   average request sizes,
5. E005 identifies sustainable offered load under a documented SLO,
6. E006 reports transparent, reproducible serving-cost estimates,
7. raw measurements, realized workload plans, environment metadata, summaries,
   plots, and reports are preserved for every canonical experiment,
8. every conclusion states its hardware, model, backend, workload, load model,
   and validity limits.

In one sentence:

> V2 makes `inference_os` capable of explaining how application-shaped workload
> variability affects inference performance, sustainable serving capacity, and
> estimated cost—without building the applications themselves.
