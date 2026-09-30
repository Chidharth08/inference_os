# inference_os V4 — Workload-Aware Prefix Caching

V4 evolves `inference_os` from modeling request-length distributions and serving
load into a framework for studying how repeated prompt content affects LLM
inference performance.

V1 established trustworthy request-level measurement and isolated prompt
length, output length, and concurrency. V2 introduced application-shaped token
distributions, workload heterogeneity, open-loop load, SLO goodput, and serving
economics. V4 builds on those foundations by making prompt identity and
relationships between requests explicit.

V3 is intentionally skipped. Version numbers describe project milestones, while
experiment identifiers remain continuous. Because V2 already contains E006 —
Serving Economics, V4 contains E007 through E009.

The central progression is:

```text
V1: measurement → isolated scaling → concurrency
V2: workload distributions → heterogeneity → load → capacity → economics
V4: prompt identity → prefix reuse → cache verification → application benefit
```

---

# V4 Objective

The engineering goal of V4 is to make `inference_os` capable of answering:

> When does automatic prefix caching reduce inference cost, how does its benefit
> change with the amount of reusable prompt content, and which application-shaped
> request patterns benefit in practice?

V4 studies the requests produced by applications. It does not build chat, RAG,
summarization, or agent applications.

The optimization under study is automatic prefix caching in vLLM. When later
requests begin with the same token sequence as an earlier request, reusable KV
cache blocks may allow the server to skip computation for part of the repeated
prefill. The expected direct effect is therefore lower prefill work and lower
time to first token, not faster autoregressive decode.

V4 must demonstrate observed cache reuse rather than infer it from a server flag
or from intended prompt construction.

---

# Why V4 Exists

V2 showed that token-length distributions affect latency, throughput, and
serving capacity. However, token counts alone do not fully describe an inference
workload.

Two requests with the same input length can behave differently when:

- one repeats a long prefix already processed by the server,
- one differs within the first cacheable block,
- one places shared content before its unique content,
- one places unique content before otherwise repeated content,
- one extends a previous conversation or agent trajectory,
- one reuses only a short instruction before a large unique document.

This means that workload characterization must eventually include token identity
and relationships between requests, not only independent request lengths.

V4 introduces those properties in controlled stages:

1. validate the caching mechanism and its measurements,
2. vary the reusable fraction of a fixed-length prompt,
3. apply reuse patterns to synthetic application-shaped workloads.

---

# What Success Looks Like

A successful V4 should make it possible to:

1. generate deterministic prompts with exact shared prefixes and unique suffixes,
2. generate prompts that are deliberately unique from their first cacheable
   block,
3. preserve prompt-reuse relationships in the realized workload plan,
4. separate model/GPU warm-up from prefix-cache population,
5. distinguish compulsory cache misses from steady-state cache hits,
6. verify the running server's prefix-caching configuration,
7. collect cache query, hit, and cached-token observations for a measured window,
8. compare caching enabled and disabled under identical request plans,
9. relate observed cached-token fraction to TTFT and throughput changes,
10. model chat-like, RAG-like, summarization-like, and agent-like prefix reuse,
11. evaluate selected application-shaped cases under open-loop load and an SLO,
12. preserve enough metadata to reproduce cache state and server configuration.

V4 is successful when the experiments demonstrate these capabilities with raw,
auditable evidence. Adding `enable_prefix_caching: true` to a configuration is
not sufficient.

---

# V4 Conceptual Model

## Prompt Structure

The minimum V4 prompt model is:

```text
[shared prefix][unique suffix]
```

The total input length is:

```text
total_input_tokens = shared_prefix_tokens + unique_suffix_tokens
```

The configured shared fraction is:

```text
configured_shared_fraction = shared_prefix_tokens / total_input_tokens
```

This is an intended workload property. It is not proof that the server reused
the same number of tokens. Cache block boundaries, server state, eviction, and
backend behavior can make the observed reusable fraction different.

## Prefix Identity

Prefix caching depends on exact token identity and order. Similar text is not an
adequate substitute for identical token sequences.

The framework must construct prompts at the token level or verify the final
tokenized result so that:

- requests in the same reuse group have the intended common token prefix,
- unique-control requests do not accidentally share a complete cacheable block,
- suffixes remain deterministic and distinct,
- decoding and re-encoding do not silently change the intended boundary.

V4 studies prefix reuse, not arbitrary repeated substrings. Content that appears
after the first differing token is not assumed reusable merely because it also
appears in another prompt.

## Request Relationships

V2 request plans could be understood as independent token shapes. V4 request
plans may contain relationships such as:

```text
reuse group: document-1
request 1:   [document-1][question-A]
request 2:   [document-1][question-B]
request 3:   [document-1][question-C]
```

or a growing trajectory:

```text
turn 1: [system][tools][state-A][event-1]
turn 2: [system][tools][state-A][result-1][event-2]
turn 3: [system][tools][state-A][result-1][event-2][result-2][event-3]
```

The realized workload must preserve enough metadata to reconstruct these
relationships without requiring the report reader to infer them from request
order.

## Cache State

Cache state is part of the experimental condition.

At minimum, V4 distinguishes:

```text
model/GPU warm:       one-time execution effects have been exercised
target prefix cold:   the measured prefix has not yet populated the cache
target prefix warm:   a prior request has made the prefix eligible for reuse
```

A warm server is not necessarily a warm target prefix. Conversely, a normal
benchmark warm-up can accidentally populate the target prefix and turn a
supposed first miss into a hit.

---

# Conceptual Workload Configuration

The exact schema should follow the existing configuration system and may evolve
as E007 is implemented. Conceptually, a controlled reuse workload may look like:

```yaml
experiment_id: E007
model: Qwen/Qwen2.5-7B-Instruct
base_url: http://localhost:18000
num_requests: 50
warmup_requests: 5
concurrency: 1
seed: 42
temperature: 0.0
enable_prefix_caching: true

workload:
  name: repeated_prefix
  input_tokens:
    values: [4096]
    weights: [1.0]
  max_output_tokens:
    values: [64]
    weights: [1.0]
  prompt_reuse:
    mode: shared_prefix
    shared_prefix_tokens: 3072
    reuse_group_count: 1
```

The corresponding no-reuse control may use:

```yaml
prompt_reuse:
  mode: none
  require_unique_first_cacheable_block: true
```

The configuration describes the intended workload. The stored run must also
contain the realized token counts, reuse relationships, and observed cache
metrics.

---

# Measurement Validity Requirements

## 1. Server Configuration Must Be Verified

`enable_prefix_caching` in a benchmark configuration does not by itself change
an already running external vLLM server.

Every run must capture and verify the effective server configuration, including
at least:

- vLLM version,
- prefix-caching enabled or disabled,
- cache block or prefix-match granularity when exposed,
- KV-cache dtype,
- GPU-memory utilization setting,
- chunked-prefill setting,
- model name and revision,
- complete server command or equivalent structured configuration.

If the observed server state disagrees with the experiment condition, the run
must fail validation or be marked invalid. It must not be silently reported as
a cache-ON or cache-OFF result.

## 2. Model Warm-Up and Cache Warm-Up Must Be Separate

Model/GPU warm-up prompts must not share the target experiment prefix. They
should exercise the serving path while remaining disjoint from measured reuse
groups.

After model warm-up:

1. the first request for a target prefix is recorded as a compulsory miss,
2. later requests are recorded as expected steady-state hits,
3. observed cache metrics determine whether those expectations were met.

Warm-up requests and cache-populating requests must be persisted and labeled.

## 3. Cache Metrics Must Use Window Deltas

Server metrics may be cumulative across requests and benchmark runs. V4 must
capture cache counters immediately before and after each measured condition and
derive deltas for that condition.

For the pinned vLLM version, the report must document the exact meaning and units
of every cache metric used. A token-based observed hit fraction is conceptually:

```text
observed_prefix_cache_hit_fraction =
    delta(prefix_cache_hit_tokens) / delta(prefix_cache_query_tokens)
```

The implementation must use the actual metric names and semantics exposed by
the tested version. Division-by-zero behavior and unavailable metrics must be
explicit.

## 4. Independent Conditions Must Not Share Hidden State

Independent experimental points must start from a known cache state. Server
restart is the default isolation mechanism unless a verified cache-reset
mechanism is implemented.

Cache-OFF and cache-ON runs must otherwise use equivalent server settings.
Execution order should be alternated or randomized across repetitions so that
thermal, background, or time-order effects do not consistently favor one
condition.

## 5. Intended and Observed Reuse Must Remain Separate

The workload plan records intended values such as:

- configured shared-prefix tokens,
- configured shared fraction,
- reuse group,
- trajectory or conversation identifier,
- turn index,
- expected cache state.

Server observations record values such as:

- cache query tokens,
- cache hit tokens,
- cached prompt tokens,
- KV-cache utilization,
- preemptions when available.

Reports must not substitute intended reuse for observed reuse.

## 6. Prompt Preparation Must Not Contaminate Timings

Prompt construction, tokenization, prefix verification, hashing, and request-plan
generation must happen before measured request dispatch. These client-side
operations are workload preparation, not inference latency.

## 7. Canonical Results Require Repetition

Pilot runs may use one repetition. Canonical comparisons should use at least
three independent repetitions per condition when GPU budget permits.

Reports must show per-repetition results or uncertainty rather than treating all
requests from one server lifetime as independent replications.

---

# Backward Compatibility with V1 and V2

Existing E000–E006 configurations, reports, and stored runs must remain valid.

V4 extends the V2 `prompt_reuse` abstraction. It must preserve:

```yaml
prompt_reuse:
  mode: none
```

for existing profiles without changing their generated prompts or request-plan
order.

Existing `workload.jsonl`, `requests.jsonl`, `summary.json`, telemetry, SLO, and
economics artifacts must remain readable. New reuse and cache fields should be
additive or versioned rather than retroactively required from old runs.

V4 must not reinterpret the repeated fixed prompt used by legacy V1 experiments
as canonical V4 cache evidence. V4 requires explicit reuse metadata and observed
cache measurements.

---

# V4 Experiment Plan

## E007 — Prefix Caching Mechanism Validation

### Question

> Does verified prefix reuse reduce prefill latency, and can the framework
> distinguish a cache hit from merely enabling caching?

E007 uses a 2×2 controlled design:

| Prompt pattern | Cache OFF | Cache ON |
|---|---:|---:|
| Unique prefixes | Yes | Yes |
| Repeated prefix plus unique suffix | Yes | Yes |

The unique-prefix conditions are negative controls. If cache ON materially
improves unique-prefix traffic without observed reuse, the result requires
investigation rather than being attributed to prefix caching.

The initial canonical shape is approximately:

```text
total prompt:      4096 tokens
shared prefix:     3072 tokens
unique suffix:     1024 tokens
maximum output:      64 tokens
concurrency:           1
```

Short output reduces the chance that decode time hides a prefill improvement.

E007 separates:

- disjoint model/GPU warm-up,
- the first compulsory miss for the target prefix,
- steady-state requests expected to hit the cache.

### Independent Variables

- prefix caching disabled versus enabled,
- unique versus repeated-prefix prompt construction,
- first target request versus subsequent target requests.

### Controlled Variables

- model and model revision,
- hardware and precision,
- total input length,
- maximum output length,
- concurrency,
- sampling parameters,
- request count,
- server settings other than prefix caching,
- warm-up and measurement protocol.

### Metrics

- TTFT P50/P95 and per-request TTFT,
- E2E latency,
- TPOT as a decode sanity check,
- request throughput,
- output-token throughput,
- cache query and hit token deltas,
- cached prompt-token deltas when available,
- KV-cache utilization,
- GPU utilization and memory,
- error rate.

### Hypotheses

1. Repeated-prefix/cache-ON requests after the compulsory miss will produce
   observed cache hits and lower TTFT than repeated-prefix/cache-OFF requests.
2. Unique-prefix/cache-ON traffic will produce negligible cached-token reuse and
   no material TTFT improvement over unique-prefix/cache-OFF traffic.
3. TPOT will remain broadly similar because prefix caching avoids repeated
   prefill computation rather than decode computation.
4. E2E improvement will be smaller than TTFT improvement when generation time
   remains unchanged.

### Engineering Capability Introduced

E007 requires:

- deterministic shared-prefix and unique-prefix generation,
- token-level prefix verification,
- explicit cache-state phases,
- effective server-configuration verification,
- before/after server-metric snapshots,
- cold-miss and steady-state summaries,
- paired four-condition reporting.

### What E007 Does Not Prove

E007 does not establish production benefit, cache capacity, eviction behavior,
or performance under heterogeneous load. It validates the mechanism and the
measurement method under one controlled prompt shape.

---

## E008 — Prefix Reuse Sensitivity

### Question

> How does prefix-caching benefit change as the reusable fraction of a fixed-size
> prompt increases?

E008 holds total prompt length constant and sweeps the configured shared-prefix
fraction:

```text
0%, 25%, 50%, 75%, 90%
```

For a 4096-token prompt, exact token targets may be adjusted to valid cacheable
boundaries. The report must preserve both the requested fraction and the actual
tokenized/shared boundary.

The primary relationship is:

```text
observed cached-token fraction
        ↓
TTFT reduction
```

### Independent Variable

- configured shared-prefix fraction.

### Controlled Variables

- total prompt length,
- output-token limit,
- reuse group count,
- request order,
- request count,
- concurrency,
- model and server configuration,
- warm-up and cache-state protocol.

### Metrics

- configured shared-prefix tokens and fraction,
- actual common-prefix tokens,
- observed prefix-cache hit fraction,
- TTFT P50/P95,
- absolute and relative TTFT change versus cache OFF,
- E2E latency,
- TPOT,
- request and output-token throughput,
- GPU utilization and memory,
- KV-cache utilization.

### Hypotheses

1. Observed cached-token fraction will generally increase with configured shared
   fraction, subject to cache granularity.
2. TTFT will decrease as more prefill work is reused.
3. Small reusable prefixes may produce negligible end-to-end benefit even when
   they register cache hits.
4. TPOT will remain broadly stable in the controlled low-concurrency experiment.

### Engineering Capability Introduced

E008 requires:

- a shared-prefix fraction sweep,
- exact common-prefix validation,
- configured-versus-observed reuse reporting,
- cache-aware comparison plots,
- relative improvement metrics with explicit baselines.

### What E008 Does Not Prove

E008 does not characterize cache eviction, reuse distance, working-set capacity,
or popularity distributions. It uses a small resident working set to isolate
shared-prefix size.

Working-set sweeps, round-robin versus Zipf popularity, eviction curves, and
reuse-distance studies are optional follow-up experiments. They are not part of
V4's definition of done unless E009 reveals unexplained cache-pressure behavior.

---

## E009 — Application-Shaped Prefix Reuse

### Question

> Which application-shaped request relationships benefit from prefix caching,
> and can prompt structure change latency, throughput, or SLO goodput even when
> token lengths remain similar?

E009 applies explicit prompt identity and ordering to four synthetic profiles:

- `chat_like`,
- `rag_like`,
- `summarization_like`,
- `agent_like`.

These names describe request patterns. E009 does not execute the corresponding
applications or measure application quality.

### Chat-Like

The chat-like profile contains:

```text
shared system prompt
+ growing conversation history
+ current user turn
```

Later turns in one conversation contain the exact tokenized history of earlier
turns, including role markers and prior assistant content. Multiple conversation
IDs may be interleaved, but ordering within one conversation remains valid.

The important reuse property is a modest stable system prefix plus a growing
conversation-specific prefix.

### RAG-Like

The RAG-like profile compares two prompt layouts with similar content and token
counts.

Cache-friendly:

```text
[system][shared document][unique question]
```

Cache-hostile:

```text
[system][unique question][same document]
```

The first layout keeps the large repeated document before the first unique
tokens. The second diverges early, so the later matching document is not treated
as a reusable prefix.

This comparison asks whether prompt ordering itself can influence inference
performance.

### Summarization-Like

The summarization-like profile contains:

```text
[shared summarization instruction][unique long document]
```

Only a small portion of the prompt repeats. This is the application-shaped
negative control: a cache hit on a short prefix may correspond to a low cached-
token fraction and negligible TTFT improvement.

### Agent-Like

The agent-like profile models the shape of repeated model calls produced by an
agent loop. It does not implement tools, planning, browsing, or multi-agent
coordination.

Conceptually, each trajectory contains:

```text
stable system instructions:  approximately 1500 tokens
stable tool schemas:         approximately 2500 tokens
growing trajectory/state:    approximately 1000–3000 tokens
current tool result/event:   approximately 200–1000 tokens
maximum output:              approximately 50–300 tokens
calls per trajectory:        approximately 3–6
```

Sequential calls grow from the prior exact trajectory:

```text
call 1: [system][tools][state-A][event-1]
call 2: [system][tools][state-A][result-1][event-2]
call 3: [system][tools][state-A][result-1][event-2][result-2][event-3]
```

The defining property is a large stable system/tool prefix followed by a growing
trajectory and a small unique tail. Calls within a trajectory are sequential;
multiple independent trajectories may be interleaved to create concurrency.

Conversation and trajectory content should be generated deterministically before
measurement. E009 does not make a measured request depend on nondeterministic
live output from a preceding request.

### Experimental Phases

E009 proceeds in two stages:

1. compare all application-shaped profiles at a fixed controlled concurrency,
2. apply open-loop load and SLO evaluation to a small set of representative
   cache-friendly and cache-hostile cases.

The second stage reuses V2's open-loop scheduler and SLO machinery. It should not
repeat a large saturation matrix for every profile. Load points should be chosen
from prior capacity evidence and a pilot.

### Metrics

- actual input and output-token distributions,
- intended and actual common-prefix lengths,
- observed prefix-cache hit fraction,
- cached prompt-token fraction when available,
- TTFT/E2E/TPOT P50/P95/P99,
- request and token throughput,
- offered rate and achieved rate for selected open-loop cases,
- in-flight requests and dispatch delay,
- error, timeout, and drop rates,
- individual SLO compliance and aggregate SLO pass/fail,
- SLO goodput,
- GPU utilization, memory, and KV-cache utilization.

### Hypotheses

1. RAG-friendly ordering will reuse more prompt tokens and reduce TTFT relative
   to RAG-hostile ordering with similar token counts.
2. Summarization-like traffic will obtain little benefit because most expensive
   prompt tokens belong to unique documents.
3. Agent-like sequential calls will benefit from both the stable system/tool
   prefix and the growing trajectory prefix.
4. Chat-like benefit will grow across turns within a conversation but depend on
   exact history preservation.
5. Selected cache-friendly patterns will improve throughput or SLO goodput under
   load, but the improvement will remain specific to the tested workload and
   serving configuration.

### Engineering Capability Introduced

E009 requires:

- request relationship and reuse-group metadata,
- conversation and trajectory identifiers,
- deterministic growing-prefix construction,
- prompt-layout variants,
- per-profile cache and latency summaries,
- reuse-aware open-loop workload execution,
- cache-aware SLO and goodput comparison.

### What E009 Does Not Prove

E009 does not measure retrieval quality, summarization quality, tool correctness,
agent success, conversation quality, or end-user task completion. It does not
claim that the synthetic profiles reproduce a production traffic trace.

---

# V4 Metrics

V4 retains all V1 and V2 metrics and adds, when available and required:

- configured shared-prefix tokens,
- configured shared-prefix fraction,
- actual common-prefix tokens,
- prefix reuse-group identifier,
- conversation or trajectory identifier,
- turn/call index,
- expected cache state,
- prefix-cache query-token delta,
- prefix-cache hit-token delta,
- observed prefix-cache hit fraction,
- cached prompt-token delta,
- KV-cache utilization,
- compulsory-miss latency,
- steady-state-hit latency,
- absolute latency change,
- relative latency change against an explicit baseline.

Every derived metric must define:

- numerator,
- denominator,
- units,
- measurement window,
- eligible requests,
- server/version-specific source metric,
- behavior when the denominator is zero or the source is unavailable.

Input-token throughput in existing summaries counts request input tokens. Under
prefix caching, it represents effective workload throughput, not necessarily the
number of prompt tokens physically recomputed by the model. Reports must keep
effective input throughput separate from observed cached or recomputed prompt
work.

---

# Required Run Artifacts

In addition to existing V1/V2 artifacts, a canonical V4 run should preserve
conceptually similar files:

```text
runs/<run-id>/
├── config.json
├── environment.json
├── workload.jsonl
├── requests.jsonl
├── telemetry.jsonl
├── summary.json
├── server_config.json
├── cache_metrics_before.txt
├── cache_metrics_after.txt
└── cache_metrics_delta.json
```

The exact filenames may follow implementation conventions. Raw server snapshots
should be retained so derived cache metrics can be audited when metric names or
semantics change between vLLM versions.

For each request, `workload.jsonl` should eventually be capable of preserving:

```json
{
  "request_id": "req-2",
  "is_warmup": false,
  "target_input_tokens": 4096,
  "max_output_tokens": 64,
  "prompt_reuse_mode": "shared_prefix",
  "reuse_group_id": "prefix-1",
  "configured_shared_prefix_tokens": 3072,
  "actual_common_prefix_tokens": 3072,
  "trajectory_id": null,
  "turn_index": null,
  "expected_cache_state": "warm"
}
```

This schema is conceptual. Implementation should add only the fields required by
the active experiment.

---

# Explicitly Out of Scope for V4

Do not add the following as part of V4:

- a production chat application,
- a RAG retrieval or ingestion pipeline,
- embedding models or vector databases,
- real tool execution,
- an agent framework,
- multi-agent orchestration,
- planner, browser, research, or tool-agent variants,
- application-quality or task-success evaluation,
- production traffic datasets unless separately justified,
- exhaustive reuse-distance sweeps,
- working-set capacity characterization,
- round-robin versus Zipf popularity studies,
- cache-eviction curve characterization,
- distributed or cross-instance prefix caching,
- multi-GPU cache coordination,
- speculative decoding,
- quantization comparisons,
- serving-backend comparisons,
- general automatic prompt optimization,
- automatic configuration recommendation,
- web dashboards,
- hosted services,
- Kubernetes or cloud orchestration,
- custom CUDA or Triton kernels.

Cache locality and eviction experiments may become an optional follow-up only if
V4 observations motivate a specific hypothesis.

---

# V4 Engineering Principles

## 1. Extend V2; Do Not Rewrite It

Reuse the deterministic workload, request runner, open-loop load, SLO, telemetry,
and persistence infrastructure. Add the smallest representation of prompt reuse
needed by each experiment.

## 2. Cache ON Is Not Evidence of a Cache Hit

Every caching conclusion must be supported by observed cache metrics and verified
prompt identity. Configuration intent alone is insufficient.

## 3. Prompt Identity Is First-Class Workload Data

Token counts, request order, reuse groups, and exact prefix relationships must be
auditable. A seed alone is not a substitute for the realized plan.

## 4. Cache State Is an Experimental Variable

Cold and warm cache behavior must not be mixed without labeling. Warm-up policy,
cache population, condition isolation, and server lifetime belong in the report.

## 5. Controlled Experiments Still Apply

E007 varies cache enablement and prefix identity. E008 varies shared-prefix
fraction. E009 varies application-shaped reuse patterns. Do not introduce cache
pressure as an additional independent variable without a motivated follow-up.

## 6. Application Labels Describe Request Patterns

`chat_like`, `rag_like`, `summarization_like`, and `agent_like` describe synthetic
model-request relationships. They do not imply that the framework executes or
evaluates those applications.

## 7. Report Mechanism Before Benefit

First establish that the intended cache hits occurred. Then interpret TTFT,
throughput, or goodput changes. A faster result without mechanism evidence is an
observation, not a prefix-caching conclusion.

## 8. Do Not Generalize Beyond the Tested Environment

Cache behavior depends on model architecture, vLLM version, block granularity,
cache configuration, GPU memory, prompt layout, concurrency, and request order.
Every conclusion must retain those boundaries.

---

# Planned Repository Additions

The exact names should follow the existing code during implementation. V4 is
expected to require additions conceptually similar to:

```text
src/inference_os/
├── workloads/
│   ├── spec.py                 # Extended prompt-reuse configuration and metadata
│   └── prefix.py               # Deterministic shared-prefix prompt construction
├── telemetry/
│   └── vllm_metrics.py         # Server metric snapshots and counter deltas
├── metrics/
│   └── cache.py                # Validated cache-derived metrics
└── reports/
    └── plots.py                # Cache fraction and latency comparison plots

configs/
├── e007_prefix_cache_off_unique.yaml
├── e007_prefix_cache_on_unique.yaml
├── e007_prefix_cache_off_repeated.yaml
├── e007_prefix_cache_on_repeated.yaml
├── e008_prefix_fraction.yaml
├── e009_chat_like_cache.yaml
├── e009_rag_like_cache.yaml
├── e009_summarization_like_cache.yaml
└── e009_agent_like_cache.yaml

experiments/
├── E007-prefix-caching-baseline/
├── E008-prefix-reuse-sensitivity/
└── E009-application-prefix-reuse/
```

This is a planning structure, not a requirement to create every file in advance.
Directories and abstractions should be added only when the active experiment
requires them.

---

# V4 Milestones

## Milestone V4.0 — Prefix-Reuse Foundation

- extend prompt-reuse configuration without breaking V2 profiles,
- define reuse-group and request-relationship metadata,
- generate deterministic exact shared prefixes and unique suffixes,
- verify token-level common-prefix lengths,
- persist reuse metadata in the realized workload,
- capture and validate effective vLLM cache configuration,
- capture raw server metrics and derive measured-window deltas,
- add unit and offline integration tests.

Do not run the canonical E007 comparison until the four conditions and cache
metric deltas can be validated offline.

## Milestone V4.1 — E007 Mechanism Validation

- implement the 2×2 unique/repeated × cache-OFF/cache-ON comparison,
- separate disjoint model warm-up, compulsory miss, and steady-state hits,
- run a pilot on the target GPU,
- repeat canonical conditions with isolated server state,
- produce plots and the E007 report.

## Milestone V4.2 — E008 Reuse Sensitivity

- sweep shared-prefix fraction while holding total length constant,
- preserve configured and actual prefix boundaries,
- relate observed cache-hit fraction to TTFT improvement,
- report granularity and measurement limitations,
- produce plots and the E008 report.

## Milestone V4.3 — E009 Application-Shaped Reuse

- implement deterministic chat-like growing histories,
- implement RAG-friendly and RAG-hostile prompt ordering,
- implement the summarization-like low-reuse control,
- implement deterministic sequential agent-like trajectories,
- compare all profiles at controlled concurrency,
- apply open-loop SLO measurement to selected representative cases,
- produce plots and the E009 report.

---

# Current V4 Milestone

The current V4 milestone is:

## V4.1 — E007 GPU Pilot Pending

V1 and V2 are complete. The V4.0 prefix-reuse foundation and the local E007
implementation are complete. E007 now has deterministic token-verified prompt
plans, disjoint model warm-up, server-configuration checks, measured-window
cache-counter deltas, four condition configurations, comparison plots, and
offline tests.

No E007 performance result is claimed yet. The next task is to run the four
pilot conditions on an isolated target GPU, verify the live vLLM metric semantics
and cache behavior, and only then proceed to canonical E007 measurements.

---

# V4 Definition of Done

V4 is complete when:

1. existing E000–E006 tests, configurations, reports, and runs remain valid,
2. exact shared-prefix and unique-prefix workloads can be generated and audited,
3. model/GPU warm-up is separated from target-prefix cache population,
4. effective server cache configuration is verified for every canonical run,
5. measured-window cache counter deltas are preserved with their raw snapshots,
6. E007 demonstrates whether verified steady-state hits reduce TTFT while unique
   controls do not show unexplained benefit,
7. E008 explains how benefit changes with observed cached-token fraction,
8. E009 compares chat-like, RAG-like, summarization-like, and agent-like request
   relationships without building those applications,
9. E009 demonstrates the effect of cache-friendly versus cache-hostile prompt
   ordering,
10. selected application-shaped cases are evaluated for throughput or SLO
    goodput under open-loop load,
11. raw measurements, workload relationships, server/cache observations,
    environment metadata, summaries, plots, and reports are preserved for every
    canonical experiment,
12. every conclusion states its model, hardware, backend version, cache settings,
    prompt structure, load model, and validity limits,
13. cache-pressure and eviction characterization remain outside the required
    scope unless a specific V4 result motivates them.

In one sentence:

> V4 makes `inference_os` capable of explaining when exact prompt-prefix reuse
> improves inference latency and serving performance across controlled and
> application-shaped workloads—without building the applications themselves.
