# inference_os — Study Guide for E007, E008, and E009

## Prefix Caching: Mechanism, Reuse Sensitivity, and Application-Shaped Workloads

## Table of Contents

1. [Where E007–E009 Fit](#1-where-e007e009-fit)
2. [The Core Idea: Reuse Prefill Work](#2-the-core-idea-reuse-prefill-work)
3. [KV Cache and Automatic Prefix Caching](#3-kv-cache-and-automatic-prefix-caching)
4. [The Rules of Prefix Reuse](#4-the-rules-of-prefix-reuse)
5. [Metrics and Equations](#5-metrics-and-equations)
6. [Experimental Validity](#6-experimental-validity)
7. [E007 — Does Prefix Caching Actually Work?](#7-e007--does-prefix-caching-actually-work)
8. [Reading the E007 Results](#8-reading-the-e007-results)
9. [E008 — How Benefit Changes with Reusable Fraction](#9-e008--how-benefit-changes-with-reusable-fraction)
10. [Reading the E008 Results](#10-reading-the-e008-results)
11. [E009 — Application-Shaped Prefix Reuse](#11-e009--application-shaped-prefix-reuse)
12. [Reading the E009 Controlled Results](#12-reading-the-e009-controlled-results)
13. [E009 Under Open-Loop Load](#13-e009-under-open-loop-load)
14. [The Complete V4 Story](#14-the-complete-v4-story)
15. [Code Walkthrough: Configuration to Evidence](#15-code-walkthrough-configuration-to-evidence)
16. [Artifact Walkthrough and Run Auditing](#16-artifact-walkthrough-and-run-auditing)
17. [Applied Design Guidance](#17-applied-design-guidance)
18. [Operating the Experiments](#18-operating-the-experiments)
19. [Common Interpretation Mistakes](#19-common-interpretation-mistakes)
20. [Limitations and Follow-Up Experiments](#20-limitations-and-follow-up-experiments)
21. [Interview-Ready Explanations](#21-interview-ready-explanations)
22. [Exercises](#22-exercises)
23. [Exercise Solutions](#23-exercise-solutions)
24. [Glossary](#24-glossary)
25. [Final Mental Model](#25-final-mental-model)

---

## 1. Where E007–E009 Fit

The project progression is:

```text
V1 / E000–E002
trustworthy measurement → prefill/decode → closed-loop concurrency

V2 / E003–E006
application-shaped lengths → heterogeneity → open-loop load → economics

V4 / E007–E009
prompt identity → verified prefix reuse → application-shaped caching
```

V2 showed that request lengths and arrival patterns matter. V4 adds another
dimension: **the identity and order of the tokens**. Two 4,096-token prompts
can have identical lengths but very different serving cost if one begins with a
previously processed prefix and the other diverges immediately.

V3 was intentionally skipped as a project milestone name. Experiment numbering
remains continuous, so V4 begins with E007.

The three experiments form a scientific progression:

| Experiment | Main question | Role |
| :--- | :--- | :--- |
| E007 | Does verified prefix reuse reduce latency? | Validate mechanism and measurement |
| E008 | How does benefit scale with reusable fraction? | Build a controlled response curve |
| E009 | Which realistic request relationships benefit? | Apply the mechanism to workload shapes |

Do not collapse them into one claim. E007 establishes that the mechanism is
real, E008 characterizes one important variable, and E009 tests whether the
lesson survives more realistic prompt relationships and load.

---

## 2. The Core Idea: Reuse Prefill Work

An autoregressive LLM request has two broad phases:

```text
prompt tokens ──► prefill ──► first generated token ──► decode ──► remaining tokens
```

During **prefill**, the model processes all prompt tokens and creates attention
keys and values for every transformer layer. During **decode**, it generates
one new token at a time while reading the existing keys and values.

Without prefix caching, two requests that begin with the same 3,000 tokens each
perform the same prefill computation for those 3,000 tokens.

```text
Request A: [same 3,000 tokens][unique A]
Request B: [same 3,000 tokens][unique B]
                           ▲
             repeated prefill work without caching
```

With automatic prefix caching, the server can retain the KV tensors produced
for cacheable blocks of Request A. Request B can reuse those tensors and prefill
only its noncached suffix.

```text
Request A: [compute shared prefix][compute suffix A]
Request B: [reuse shared prefix  ][compute suffix B]
```

The expected direct effect is:

```text
less repeated prefill work
        ↓
lower time to first token (TTFT)
        ↓
possibly lower end-to-end latency and higher throughput
```

Prefix caching does **not** skip generation of the new answer. It therefore
should not materially improve time per output token (TPOT) for otherwise
equivalent requests.

This gives V4 a useful diagnostic pattern:

- TTFT should improve when verified hit tokens increase.
- TPOT should remain broadly stable.
- E2E should improve by less than TTFT when decode remains significant.
- Cache ON without hits should not create a large benefit.

---

## 3. KV Cache and Automatic Prefix Caching

### 3.1 What the KV cache stores

For each processed token and attention layer, the model produces a key vector
and a value vector. Attention for later tokens needs those vectors. Keeping them
in GPU memory avoids recomputing the entire history during each decode step.

The ordinary per-request KV cache answers:

> How can later tokens in this sequence attend to earlier tokens without
> rerunning the earlier sequence?

Automatic prefix caching extends the idea across requests:

> If a later request has an identical leading token sequence, can its prefill
> reuse KV blocks already created by an earlier request?

The cache stores model-internal KV tensors associated with token-prefix blocks.
It does not mean the server is replaying an old generated response, and it does
not eliminate decode for the new request.

### 3.2 Block granularity

The tested vLLM configuration used a 16-token cache block size. Reuse is
accounted for in complete cacheable blocks. A conceptual calculation is:

```text
cacheable_prefix_tokens = floor(common_prefix_tokens / block_size) × block_size
```

If two prompts share 1,030 leading tokens and the block size is 16, at most
1,024 tokens form complete blocks. The six remaining shared tokens belong to an
incomplete block and are not counted as a complete cached block.

This is why E008's requested 90% of 4,096 tokens was resolved from 3,686.4 to
3,680 tokens:

```text
floor(3,686.4 / 16) × 16 = 3,680
3,680 / 4,096 = 89.84375%
```

### 3.3 Compulsory miss

The first request for a prefix cannot reuse a block that has not yet been
created. It is a **compulsory miss**. It populates the cache for later requests.

For `N` equal-length requests with one shared cacheable prefix `P` and total
prompt length `T`, the ideal measured-window hit fraction is:

```text
hit_fraction = ((N - 1) × P) / (N × T)
```

For E007:

```text
N = 30
P = 3,072
T = 4,096

hit_fraction = (29 × 3,072) / (30 × 4,096)
             = 89,088 / 122,880
             = 0.725 = 72.5%
```

The steady-state reusable fraction is 75%, but the measured-window fraction is
72.5% because the window includes the first miss.

### 3.4 Cache residency and eviction

A matching prefix is useful only if its blocks are still resident. Cache
pressure, many competing prefixes, reuse distance, scheduler behavior, and
eviction policy can reduce observed reuse. E007–E009 intentionally use small,
resident working sets to isolate other variables. They do not characterize
eviction behavior.

---

## 4. The Rules of Prefix Reuse

### Rule 1: Tokens must match exactly

Semantic similarity is irrelevant. The cache operates on token identity.

```text
"Summarize this document:"  ≠  "Please summarize this document:"
```

Even if the meanings are almost identical, their token sequences can differ.

### Rule 2: Order matters

Prefix caching reuses a **leading** sequence. A repeated document after a
different question is not rescued by the fact that it appears later.

```text
Friendly: [system][same document][different question]
Hostile:  [system][different question][same document]
```

The friendly layout has a long common prefix. The hostile layout diverges at
the question, so the later matching document is not part of the common prefix.

### Rule 3: Formatting is content

Role markers, whitespace, serialization order, tool-schema ordering, JSON field
ordering, and template versions all affect the token sequence. A stable system
message with nondeterministically ordered tool definitions may destroy reuse.

### Rule 4: A server flag is not evidence of reuse

`--enable-prefix-caching` makes reuse possible. It does not prove that any
request hit the cache. Evidence requires observed cache counters and an audited
request plan.

### Rule 5: Intended reuse and observed reuse are different quantities

The workload says what should be reusable. Server metrics say what was reused.
Both are needed:

```text
intended shared tokens
        +
token-level prompt verification
        +
observed server hit counters
        +
latency movement
        =
credible mechanism claim
```

### Rule 6: Reuse benefit depends on what remains

A 75% cached fraction can greatly reduce prefill but may produce a smaller E2E
change if the request generates many tokens. User-visible benefit depends on
the mix of prefill, queueing, and decode.

---

## 5. Metrics and Equations

### 5.1 TTFT

```text
TTFT = first_token_time - request_start_time
```

TTFT includes server queueing plus prefill and first-token work. In the
controlled concurrency-1 experiments, queueing is minimized, making TTFT a
useful proxy for prefill changes. Under open-loop load, queueing can dominate.

### 5.2 TPOT

For a request that returns `O` output tokens:

```text
TPOT = (E2E - TTFT) / (O - 1)
```

The first output token is represented by TTFT; TPOT describes intervals for the
remaining output tokens. Prefix caching should mainly change TTFT, not TPOT.

### 5.3 End-to-end latency

```text
E2E = completion_time - request_start_time
```

A useful approximation is:

```text
E2E ≈ TTFT + (output_tokens - 1) × TPOT
```

Therefore a 70% TTFT reduction does not imply a 70% E2E reduction.

### 5.4 Cache hit fraction

The observed token-based fraction is:

```text
observed_hit_fraction = Δ(prefix_cache_hit_tokens)
                        ──────────────────────────
                        Δ(prefix_cache_query_tokens)
```

The deltas matter because Prometheus counters are cumulative over the server
lifetime.

### 5.5 Relative change

The reports use the cache-OFF condition as baseline:

```text
relative_change_percent = 100 × (cache_on - cache_off) / cache_off
```

A negative latency change is an improvement. A positive throughput change is
an improvement.

### 5.6 Throughput and goodput

```text
throughput = successful completions / measured wall-clock duration

goodput = SLO-compliant requests / measured wall-clock duration
```

Throughput answers “how much finished?” Goodput answers “how much finished
within the required service quality?” A run can have zero errors and zero
goodput if every response is too slow.

### 5.7 P50 versus P95

- P50 describes the median request.
- P95 exposes tail behavior.
- A cache may improve typical prefill while queue buildup still damages P95.
- With only 24 or 30 requests, percentiles are useful descriptions of this run,
  not precise estimates of a large production population.

---

## 6. Experimental Validity

Prefix-cache experiments are unusually easy to contaminate. V4 treats cache
state as part of the experimental condition.

### 6.1 Model warm-up is not target-prefix warm-up

Warm-up should exercise the model and GPU without populating the prefix used in
measurement. E007–E009 generate token-distinct warm-up prompts. This preserves
the first measured request as a real compulsory miss.

### 6.2 Fresh server per condition

Each condition used a fresh vLLM process. This prevents a prefix from an earlier
condition from remaining resident and prevents cumulative metric counters from
being confused with the current condition.

### 6.3 Effective configuration verification

The external vLLM server, not the YAML file, controls whether caching is active.
The runner reads the live `cache_config_info` metric and verifies:

- prefix caching enabled or disabled as expected,
- cache block size,
- relevant cache configuration labels,
- pristine cache counters for cache-ON controlled runs.

A config/server mismatch invalidates the condition.

### 6.4 Measured-window snapshots

The sequence is:

```text
server preflight snapshot
        ↓
disjoint warm-up requests
        ↓
"before" metric snapshot via after_warmup_hook
        ↓
measured requests
        ↓
"after" metric snapshot
        ↓
counter delta and mechanism checks
```

This excludes warm-up traffic from measured cache counters.

### 6.5 Exact token-level prompt construction

The generator does not trust character counts. It constructs and verifies
token IDs, hashes prompts and reusable prefixes, detects decode/encode boundary
changes, and rejects accidental cacheable overlap in unique controls.

### 6.6 Paired controls

Every positive cache condition has a cache-OFF counterpart with the same model,
hardware, prompt plan, lengths, sampling, and chunked-prefill setting. Negative
controls separate “cache enabled” from “cache actually used.”

### 6.7 Why chunked prefill was disabled

Chunked prefill can change scheduling and TTFT. Disabling it reduces the number
of mechanisms changing simultaneously, making prefix-cache attribution clearer.

---

## 7. E007 — Does Prefix Caching Actually Work?

### Research question

> Does caching improve a repeated-prefix workload, and can the benchmark prove
> that the improvement comes from prefix reuse?

### 7.1 The 2×2 design

| Prompt identity | Cache OFF | Cache ON |
| :--- | :---: | :---: |
| Unique from the first cacheable block | Yes | Yes |
| Shared 3,072-token prefix + unique 1,024-token suffix | Yes | Yes |

All prompts contain 4,096 input tokens, have a maximum output of 64 tokens, and
run sequentially at concurrency 1. Each canonical condition has three disjoint
warm-ups and 30 measured requests.

This design isolates two independent variables:

1. whether the server permits prefix caching;
2. whether requests actually share a cacheable prefix.

### 7.2 Why all four cells are necessary

`shared + ON` alone would show a speedup, but not prove its cause. The other
cells answer alternative explanations:

- `unique + OFF` establishes the ordinary unique-prompt baseline.
- `unique + ON` tests overhead or unexplained benefit from enablement alone.
- `shared + OFF` proves prompt content itself is not inherently cheaper.
- `shared + ON` is the treatment condition.

The strongest causal pattern is an interaction:

```text
cache ON + no reusable prefix  → no hit, no meaningful benefit
cache ON + reusable prefix     → verified hit, large TTFT benefit
```

### 7.3 Hypotheses

1. Shared/cache-ON requests after the compulsory miss will hit the cache and
   reduce TTFT.
2. Unique/cache-ON requests will have zero reusable blocks and no material
   benefit.
3. TPOT will remain stable.
4. E2E improvement will be smaller than TTFT improvement.

---

## 8. Reading the E007 Results

### 8.1 Canonical results

| Prompt pattern | Cache | TTFT P50 | TTFT P95 | E2E P50 | TPOT P50 | Requests/s | Hit fraction |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unique | OFF | 869.60 ms | 939.11 ms | 2,124.76 ms | 19.85 ms | 0.470 | N/A |
| Unique | ON | 874.86 ms | 941.30 ms | 2,134.75 ms | 19.93 ms | 0.467 | 0.0% |
| Shared | OFF | 872.51 ms | 936.84 ms | 2,128.55 ms | 19.86 ms | 0.469 | N/A |
| Shared | ON | **275.57 ms** | **360.92 ms** | **1,524.73 ms** | 19.87 ms | **0.643** | **72.5%** |

### 8.2 The negative control

Unique cache OFF versus ON changed TTFT P50 by only +0.6% and produced zero hit
tokens. Enabling the feature is not a general speed switch. The workload must
contain reusable leading tokens.

### 8.3 Exact mechanism agreement

The shared/cache-ON condition observed:

```text
query tokens = 30 × 4,096 = 122,880
hit tokens   = 29 × 3,072 = 89,088
hit fraction = 89,088 / 122,880 = 72.5%
```

The observed count exactly equals the planned count. The first request had
860.84 ms TTFT; the 29 warm requests had 275.35 ms median TTFT. The transition
from cold to warm aligns with the server counters.

### 8.4 What moved and what did not

- TTFT P50 fell 68.4% for verified reuse.
- E2E P50 fell less because decode still occurred.
- TPOT stayed around 19.9 ms/token in every condition.
- Sequential request throughput rose 37.3% because each request completed
  sooner.

The throughput figure is not a production capacity result. It is a
concurrency-1 closed-loop consequence of lower request time.

### 8.5 E007 conclusion

E007 validates the mechanism and the measurement system. It does not yet say
how benefit changes with prefix size or how real request relationships behave.
Those become E008 and E009.

---

## 9. E008 — How Benefit Changes with Reusable Fraction

### Research question

> With total prompt length fixed, how does performance change as a larger
> fraction of the prompt becomes reusable?

### 9.1 Controlled sweep

E008 holds these constant:

- 4,096 total input tokens;
- maximum 64 output tokens;
- concurrency 1;
- 30 measured requests and three disjoint warm-ups;
- one reuse group;
- model, GPU, dtype, seed, temperature, and server settings.

It varies requested shared-prefix fraction:

```text
0%, 25%, 50%, 75%, 90%
```

Every fraction has cache OFF and cache ON, producing ten conditions.

### 9.2 Why cache OFF exists at every fraction

Different deterministic token content could theoretically change runtime. A
paired OFF baseline at every point shows that the underlying 4,096-token
prompts remain stable when reuse is unavailable. Indeed, OFF TTFT medians stayed
between about 870 and 895 ms.

### 9.3 Requested versus resolved fraction

The requested value describes experiment intent. The resolved value respects
the 16-token cache-block boundary.

| Requested | Resolved tokens | Resolved fraction |
| ---: | ---: | ---: |
| 0% | 0 | 0.00% |
| 25% | 1,024 | 25.00% |
| 50% | 2,048 | 50.00% |
| 75% | 3,072 | 75.00% |
| 90% | 3,680 | 89.84% |

Both requested and resolved values must be retained. Reporting “90%” alone
would hide the actual token boundary used by the server.

---

## 10. Reading the E008 Results

### 10.1 Cache evidence

| Requested fraction | Expected hits | Observed hits | Window hit fraction | Steady-state fraction |
| ---: | ---: | ---: | ---: | ---: |
| 0% | 0 | 0 | 0.00% | 0.00% |
| 25% | 29,696 | 29,696 | 24.17% | 25.00% |
| 50% | 59,392 | 59,392 | 48.33% | 50.00% |
| 75% | 89,088 | 89,088 | 72.50% | 75.00% |
| 90% | 106,720 | 106,720 | 86.85% | 89.84% |

The measured-window fraction is lower than steady state because each condition
includes one compulsory miss. Exact agreement at all five points jointly
validates prompt generation, block rounding, ordering, metric windows, and
server behavior.

### 10.2 TTFT response curve

| Shared fraction | OFF P50 | ON P50 | Change | OFF P95 | ON P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 872.87 ms | 874.06 ms | +0.14% | 907.84 ms | 882.03 ms |
| 25% | 870.30 ms | 693.11 ms | -20.36% | 898.57 ms | 747.08 ms |
| 50% | 874.57 ms | 508.98 ms | -41.80% | 918.33 ms | 531.72 ms |
| 75% | 894.96 ms | 279.10 ms | -68.81% | 918.27 ms | 351.38 ms |
| 89.84% | 877.21 ms | 179.93 ms | -79.49% | 886.00 ms | 188.22 ms |

The relationship is strong and monotonic in this controlled environment: more
verified reusable prefill leads to lower TTFT.

It is not perfectly proportional. Fixed overheads remain, the uncached suffix
still needs prefill, and first-token execution cannot fall to zero. A 89.84%
steady-state cached fraction produced a 79.49% median TTFT reduction, not an
89.84% reduction.

### 10.3 E2E, TPOT, and throughput

| Shared fraction | E2E change | TPOT OFF | TPOT ON | Throughput change |
| ---: | ---: | ---: | ---: | ---: |
| 0% | +0.35% | 19.70 ms | 19.78 ms | approximately 0% |
| 25% | -7.51% | 19.73 ms | 19.92 ms | +8.15% |
| 50% | -18.29% | 19.76 ms | 19.56 ms | +21.32% |
| 75% | -26.74% | 19.76 ms | 19.74 ms | +36.03% |
| 89.84% | -32.97% | 19.79 ms | 19.75 ms | +46.62% |

TPOT stability is essential evidence. If TPOT had improved by 70%, the result
would not match the proposed prefill-only mechanism and would require further
investigation.

### 10.4 Practical threshold lesson

E008 does not define one universal “minimum useful prefix.” It shows a smooth
tradeoff in this environment. Whether 25% reuse is worthwhile depends on:

- absolute prompt length;
- output length;
- latency objective;
- lookup and memory overhead;
- frequency and reuse distance;
- cache pressure;
- cost of engineering a stable prefix.

Use observed tokens and end-user metrics, not a universal fraction.

---

## 11. E009 — Application-Shaped Prefix Reuse

### Research question

> Which request relationships benefit from prefix caching, and can prompt
> ordering change latency and SLO behavior even when content and lengths are
> comparable?

E009 models the shape of model requests. It does not implement real chat, RAG,
summarization, or agent products and does not measure answer quality.

### 11.1 Chat-like

```text
[stable system prompt][growing conversation history][current turn]
```

- Four conversations × six turns = 24 requests.
- Input lengths grow from 1,024 to 3,584 tokens.
- Stable system prefix is 512 tokens.
- Later turns preserve the exact earlier history as a prefix.
- Maximum output is 64 tokens.

The first conversation begins cold. The first request of the other three
conversations can reuse the common 512-token system prefix. Later turns can
reuse increasingly large conversation-specific prefixes.

### 11.2 RAG-friendly

```text
[256-token system][3,072-token shared document][changing question]
```

- Total input is 4,096 tokens.
- The stable leading region is 3,328 tokens.
- Maximum output is 64 tokens.
- One document relationship spans all 24 requests.

Expected hit tokens:

```text
23 × 3,328 = 76,544
```

### 11.3 RAG-hostile

```text
[256-token system][changing question][same 3,072-token document]
```

The content, document, total input length, and output cap match RAG-friendly.
Only order changes. The question causes early divergence, leaving only the
256-token system prompt cacheable.

```text
23 × 256 = 5,888 expected hit tokens
```

This paired layout is one of the strongest experiments in V4 because it changes
the causal property of interest while keeping major alternatives fixed.

### 11.4 Summarization-like

```text
[256-token shared instruction][unique long document]
```

- Total input is 4,096 tokens.
- Each request has a different document.
- Only the short instruction prefix is reusable.
- Expected hits are also `23 × 256 = 5,888` tokens.

This profile demonstrates that a valid cache hit can still be too small to
matter much.

### 11.5 Agent-like

```text
[stable system instructions]
[stable tool schemas]
[growing trajectory state]
[current event]
```

- Four trajectories × six sequential calls = 24 requests.
- Stable system/tool prefix is 4,000 tokens.
- Inputs grow from 5,504 to 8,064 tokens.
- Output caps grow from 64 to 256 tokens.
- Later calls preserve earlier state as an exact prefix.

The relevant property is not the word “agent.” It is repeated model inference
with a large stable leading prefix and a relatively small changing tail.

The first trajectory begins cold. Initial calls for the other trajectories can
reuse the 4,000-token system/tool prefix; later calls reuse growing
trajectory-specific state. The planned total is 142,560 hit tokens.

### 11.6 Why sequence sampling was added

Chat and agent requests must occur in a defined progression. IID sampling could
send a later turn before its earlier state or change the planned relationship.
`sampling_mode: sequence` cycles through exact configured values in order,
making the realized plan deterministic and auditable.

---

## 12. Reading the E009 Controlled Results

All ten controlled conditions completed 24 requests successfully. Cache-OFF
runs recorded zero hit tokens. Every cache-ON total exactly matched its
token-level plan.

| Profile | Hit fraction | TTFT P50 OFF | TTFT P50 ON | TTFT change | E2E change | Throughput change |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat-like | 76.85% | 579.49 ms | 177.99 ms | -69.29% | -19.06% | +24.53% |
| RAG-friendly | 77.86% | 1,065.40 ms | 273.67 ms | -74.31% | -31.31% | +43.03% |
| RAG-hostile | 5.99% | 1,070.64 ms | 1,020.98 ms | -4.64% | -1.71% | +1.13% |
| Summarization-like | 5.99% | 1,071.48 ms | 1,023.44 ms | -4.48% | -2.20% | +2.42% |
| Agent-like | 87.56% | 1,828.07 ms | 279.12 ms | -84.73% | -27.67% | +34.58% |

### 12.1 Chat-like interpretation

Chat-like traffic gains reuse as each conversation grows. Its 69.29% TTFT
reduction shows why stable system prompts and exact conversation serialization
can be valuable. The smaller 19.06% E2E reduction reminds us that generation
still takes time.

### 12.2 RAG ordering is the central applied result

Friendly and hostile RAG contain comparable content and identical total token
lengths. Friendly observed 76,544 hit tokens and reduced TTFT by 74.31%.
Hostile observed only 5,888 hit tokens and reduced TTFT by 4.64%.

```text
same document + same total length + different order
                         ↓
              radically different reuse
```

Repeated content is not enough. Repeated content must be in the stable prefix.

### 12.3 Summarization interpretation

The shared instruction produced real hits, but only 5.99% of query tokens were
hits. TTFT improved by 4.48%. A short common instruction in front of a unique
long document is a weak caching opportunity.

### 12.4 Agent-like interpretation

Agent-like had the largest reusable fraction and the largest TTFT improvement:
84.73%. This supports a practical strategy: keep stable system instructions and
tool schemas first, serialize them deterministically, and append changing state.

Its E2E improvement was 27.67%, much smaller than its TTFT improvement, because
agent calls used output limits up to 256 tokens. Prefix caching avoids repeated
prefill, not the longer decode.

### 12.5 Why cross-profile latency should be compared carefully

Profiles have different input lengths and output caps. The strongest causal
comparisons are cache OFF versus ON within one profile, and friendly versus
hostile RAG where content and total length are deliberately matched. Do not
interpret agent-like's absolute latency as directly comparable to the shorter
chat workload without accounting for shape.

---

## 13. E009 Under Open-Loop Load

Controlled concurrency-1 runs isolate mechanism. They do not show how saved
prefill affects queue buildup. E009 therefore reruns the two RAG layouts at a
deterministic offered rate of 1 request/s.

### 13.1 SLO

```text
TTFT P95 ≤ 1 second
E2E P95  ≤ 3 seconds
error rate ≤ 1%
```

The four conditions each offered 24 requests. All requests were dispatched and
completed; there were zero drops, timeouts, and failures.

### 13.2 Results

| Layout | Cache | Throughput | Goodput | TTFT P95 | E2E P95 | Max in flight | SLO |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Friendly | OFF | 0.780 req/s | 0.000 req/s | 5.704 s | 15.913 s | 15 | Fail |
| Friendly | ON | 0.962 req/s | 0.922 req/s | 0.398 s | 2.419 s | 4 | **Pass** |
| Hostile | OFF | 0.788 req/s | 0.000 req/s | 5.451 s | 15.647 s | 15 | Fail |
| Hostile | ON | 0.848 req/s | 0.000 req/s | 2.988 s | 14.297 s | 14 | Fail |

### 13.3 Queueing interpretation

At 1 request/s, cache-OFF service time was too slow to keep up. Requests arrived
faster than they drained, so in-flight work accumulated and tail latency grew.

Friendly cache ON reduced per-request prefill enough to keep the system near the
offered rate. In-flight work peaked at four instead of 15, and the complete SLO
passed.

Hostile cache ON saved only the short system prefix. It improved throughput and
TTFT somewhat, but not enough to prevent accumulation. It peaked at 14 in-flight
requests and failed both latency objectives.

This is a nonlinear systems lesson:

```text
small service-time reduction below capacity  → modest latency change
enough reduction to cross the load boundary → queue collapse and large SLO gain
```

### 13.4 Why throughput is below the offered rate

The measured duration includes draining the outstanding requests after the last
arrival. When the system falls behind, `24 / total duration` can be below the
1 request/s offered rate even though every request eventually completes.

### 13.5 Why zero errors does not mean success

All four runs had zero errors, but three had zero goodput under the defined SLO.
Reliability and service quality are different axes.

---

## 14. The Complete V4 Story

The evidence chain is:

```text
E007
Cache ON alone does nothing.
Verified repeated prefix → exact hits → lower TTFT.
        ↓
E008
Larger verified reusable fraction → progressively lower TTFT.
        ↓
E009 controlled
Real request relationships expose very different reusable fractions.
Prompt order can create or destroy the opportunity.
        ↓
E009 open loop
Enough reuse can change queueing behavior and turn an SLO failure into a pass.
```

The central conclusion is not “prefix caching is always good.” It is:

> Prefix caching is valuable when repeated requests share a sufficiently large,
> exact, resident leading token sequence. The practical effect depends on prompt
> structure, decode work, arrival rate, and whether the saved prefill moves the
> server across an operating boundary.

---

## 15. Code Walkthrough: Configuration to Evidence

### 15.1 Workload schema

[`src/inference_os/workloads/spec.py`](../src/inference_os/workloads/spec.py)
defines `PromptReuseConfig` with modes:

```text
none
unique_prefix
shared_prefix
application
```

Important fields include:

- `shared_prefix_tokens`;
- `reuse_group_id`;
- `cache_block_size_tokens`;
- `application_profile`;
- `prompt_layout`;
- `relationship_group_count`;
- `stable_prefix_tokens`;
- `document_tokens`.

Validation prevents nonsensical combinations, such as an application profile
outside application mode or RAG without document tokens.

`TokenLengthDistribution.sample_many()` supports `sequence` in addition to
randomized modes. Sequence mode is necessary for deterministic conversational
and trajectory progression.

### 15.2 Controlled prefix planning

[`src/inference_os/workloads/prefix.py`](../src/inference_os/workloads/prefix.py)
builds E007/E008 prompts.

`PreparedPrompt` stores both text and evidence metadata:

- exact token count;
- prompt hash;
- reuse mode and group;
- configured shared tokens;
- actual reusable prefix tokens;
- expected cache state;
- cacheable reusable tokens.

`prepare_reuse_prompt_plan()`:

1. creates a deterministic shared prefix when requested;
2. generates unique suffixes or block-distinct controls;
3. tokenizes the final prompt;
4. verifies exact lengths and common prefixes;
5. rejects a unique prompt sharing a full cacheable first block;
6. builds disjoint warm-up and measured plans.

`common_prefix_length()` compares token sequences directly. Token hashes make
identity auditable without relying on visual inspection of long prompts.

### 15.3 Application planning

[`src/inference_os/workloads/application.py`](../src/inference_os/workloads/application.py)
creates E009 relationships before measurement.

- `_growing_plan()` handles chat and agent trajectories.
- `_rag_plan()` creates matched friendly and hostile layouts.
- `_shared_instruction_plan()` creates summarization-like traffic.
- `_make_prepared()` attaches relationship ID, sequence index, hashes, actual
  common prefix, and expected state.

The planner verifies that every later prompt retains at least the configured
stable prefix. No measured request depends on nondeterministic model output;
the complete plan exists before dispatch.

### 15.4 Benchmark engines

[`src/inference_os/runner/engine.py`](../src/inference_os/runner/engine.py)
selects ordinary, controlled-reuse, or application planning for closed-loop
runs.

[`src/inference_os/runner/load_engine.py`](../src/inference_os/runner/load_engine.py)
does the same for open-loop execution and carries prompt metadata into saved
artifacts.

[`src/inference_os/runner/load.py`](../src/inference_os/runner/load.py) accepts an
`after_warmup_hook`. E007–E009 use the hook to capture the exact start of the
measured cache-counter window.

### 15.5 vLLM metrics

[`src/inference_os/telemetry/vllm_metrics.py`](../src/inference_os/telemetry/vllm_metrics.py):

1. fetches the `/metrics` endpoint;
2. parses needed Prometheus samples;
3. reads active cache configuration labels;
4. computes before/after counter deltas;
5. derives observed hit fraction;
6. persists raw snapshots and structured values.

Supporting more than one metric name handles naming differences such as
`vllm:prefix_cache_queries` and a `_total` variant while preserving the exact
raw exposition for audit.

### 15.6 Experiment orchestrators

- [`run_e007.py`](../experiments/E007-prefix-caching-baseline/run_e007.py)
  validates four conditions and summarizes the 2×2 comparison.
- [`run_e008.py`](../experiments/E008-prefix-reuse-sensitivity/run_e008.py)
  resolves fraction boundaries, validates ten conditions, checks monotonic hit
  fractions, and produces the sensitivity summary.
- [`run_e009.py`](../experiments/E009-application-prefix-reuse/run_e009.py)
  builds five application profiles, validates ten controlled and four
  open-loop conditions, and produces separate comparisons.

Each script fails or marks a condition invalid when the mechanism evidence does
not match expectations. A successful HTTP response alone is insufficient.

### 15.7 Reports and plots

[`src/inference_os/reports/plots.py`](../src/inference_os/reports/plots.py)
generates:

- E007 latency and cache-observation plots;
- E008 TTFT-versus-fraction, reuse-versus-benefit, and E2E/TPOT plots;
- E009 controlled latency, reuse/benefit, sequence-prefix, load throughput/
  goodput, and open-loop latency plots.

Plots are derived from validated summaries, not manually entered numbers.

---

## 16. Artifact Walkthrough and Run Auditing

Each condition directory contains:

| Artifact | What to verify |
| :--- | :--- |
| `config.json` | Intended model, lengths, cache flag, profile, seed |
| `environment.json` | Commit, GPU, driver, Python, PyTorch, vLLM |
| `workload.jsonl` | Exact plan, hashes, relationship IDs, sequence, cache state |
| `requests.jsonl` | Per-request timings, token counts, success/error |
| `summary.json` | Aggregate latency, throughput, GPU, workload statistics |
| `telemetry.jsonl` | GPU utilization and memory samples |
| `cache_metrics_preflight.prom` | Server state before traffic |
| `cache_metrics_before.prom` | Counter snapshot after warm-up |
| `cache_metrics_after.prom` | Counter snapshot after measurement |
| `cache_metrics_delta.json` | Query/hit/cached-token deltas |
| `server_cache_config.json` | Effective live vLLM cache settings |
| `e007/e008/e009_condition.json` | Mechanism checks and experiment verdict |
| `in_flight.jsonl` | Open-loop arrival/in-flight trace where applicable |

### Audit procedure

1. **Identify the condition.** Check experiment ID, prompt mode/profile, cache
   flag, and pilot versus canonical request count.
2. **Verify environment.** Confirm all compared conditions use the intended
   model, GPU type, vLLM version, and source commit.
3. **Verify server state.** Compare `server_cache_config.json` to the condition.
4. **Inspect request plan.** Confirm lengths, reuse groups, hashes, sequence
   indices, and expected cache states.
5. **Check completeness.** Count successes, failures, dropped arrivals, and
   timeouts.
6. **Recompute cache delta.** Subtract before counters from after counters.
7. **Recompute expected hits.** Use block-rounded common prefixes and request
   order.
8. **Compare expected and observed.** Controlled cache-ON totals should match;
   cache-OFF should report zero hits.
9. **Inspect cold versus warm latency.** The first compulsory miss should not be
   mistaken for steady state.
10. **Recompute report metrics.** Verify relative changes and SLO verdicts from
    raw summaries.

### Published result locations

- [E007 report](../outputs/e007_prefix_caching_baseline/e007_report.md)
- [E008 report](../outputs/e008_prefix_reuse_sensitivity/e008_report.md)
- [E009 report](../outputs/e009_application_prefix_reuse/e009_report.md)
- Raw and comparison runs are under [`runs/`](../runs/).

---

## 17. Applied Design Guidance

### 17.1 Prompt layout

Prefer:

```text
[stable instructions][stable schemas/context][changing state/query]
```

Avoid, when semantics permit:

```text
[stable instructions][changing query][large repeated context]
```

Correctness has priority. Do not reorder a prompt if it changes model behavior
or violates the model's chat template. When several correct layouts exist,
place stable content first.

### 17.2 RAG systems

If many questions reuse one document or corpus slice, a cache-friendly template
places stable retrieved content before the per-request question. Batch or route
requests with the same leading context to the same cache domain when the serving
architecture permits.

But retrieval often returns different documents or orderings. Normalize only
where it preserves relevance and answer quality. Cache performance cannot
justify degrading retrieval quality.

### 17.3 Chat systems

- Keep system instructions stable.
- Use a deterministic chat template.
- Append turns rather than rewriting old history.
- Be aware that summarizing or truncating history creates a new prefix and may
  cause a cold request.
- Version system prompts deliberately; a one-token change can invalidate old
  prefixes.

### 17.4 Agent systems

- Put stable system instructions and tool schemas first.
- Serialize tool schemas in deterministic order.
- Keep schema formatting stable across calls.
- Append observations and actions when possible.
- Treat tool-set changes as cache-key changes.
- Measure the number of model calls per task: a moderate per-call saving can
  compound across a long trajectory.

E009's `agent_like` result applies to request shape, not agent quality. It does
not prove that an agent finishes tasks faster or more accurately end to end.

### 17.5 Summarization systems

A short shared instruction before a unique long document offers little reuse.
Possible strategies include repeated document analysis, hierarchical workflows
with stable chunks, or caching at other layers, but those are different
experiments. Do not expect automatic prefix caching to transform one-off unique
documents.

### 17.6 Routing and affinity

In a multi-replica deployment, matching requests must reach a worker that owns
the relevant cached blocks unless the serving system supports shared or
distributed prefix caches. Cache-aware routing may improve hit rate, but can
also create load imbalance. The correct objective is SLO goodput, not hit rate
alone.

### 17.7 Cache budget and admission

Prefix blocks consume memory that could otherwise support active sequences.
Under pressure, a cache policy must decide which prefixes remain valuable.
Useful signals can include prefix size, reuse frequency, recency, tenant, and
saved compute. E007–E009 do not test these policies.

### 17.8 Decide with an economic model

The applied decision should combine:

```text
reuse probability
× reusable prefill work
× calls per user task
× traffic rate
× SLO value
− cache memory/opportunity cost
− engineering complexity
```

A high hit fraction is not automatically the best business outcome, and a low
hit fraction may still matter if the prefix is extremely expensive or the SLO
boundary is close.

---

## 18. Operating the Experiments

### 18.1 Server modes

Cache OFF uses:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 18000 \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill
```

Cache ON replaces the cache flag:

```bash
--enable-prefix-caching
```

### 18.2 Isolation workflow

For every condition:

1. start a fresh server with the intended mode;
2. wait for readiness;
3. run exactly one condition;
4. record the printed artifact directory;
5. stop the server;
6. restart before the next condition.

Running several cache-ON conditions in one server lifetime can violate pristine
counter and hidden-state assumptions.

### 18.3 Pilot versus canonical

A pilot validates compatibility, tokenizer behavior, endpoints, metrics, and
artifact creation at low cost. It is not substituted for canonical sample size.

Canonical summaries reject missing, duplicate, or unexpected conditions. This
prevents accidentally mixing a pilot with a canonical run or omitting a paired
baseline.

### 18.4 Practical failure diagnosis

If a run fails:

- **Tokenizer round-trip error:** the decoded text did not re-encode to the
  planned token sequence; fix prompt material rather than ignoring it.
- **Server config mismatch:** restart vLLM with the correct cache flags.
- **Nonpristine counters:** another request reached the server; restart it.
- **No hits in a cache-ON reuse run:** inspect token hashes, prefix length,
  block alignment, ordering, and cache residency.
- **Hits differ from plan:** check hidden traffic, overlapping open-loop
  population timing, eviction, and metric semantics.
- **Good hits but no TTFT movement:** the prefix may be too short, queueing or
  decode may dominate, or another serving mechanism may mask the effect.

---

## 19. Common Interpretation Mistakes

### Mistake 1: “Caching is enabled, so the workload is cached”

Enablement is a capability. Only observed hit counters establish reuse.

### Mistake 2: “Repeated text anywhere in the prompt is reusable”

Only the common leading token sequence qualifies for prefix reuse.

### Mistake 3: “A 75% shared prefix means a 75% measured hit fraction”

The first request is a compulsory miss. With 30 requests, E007 observes 72.5%
across the full window and 75% in steady state.

### Mistake 4: “A 75% hit fraction means TTFT must fall 75%”

Fixed overhead and uncached computation remain. E008 observed a 68.81% TTFT
reduction at 75% configured reuse.

### Mistake 5: “Prefix caching speeds up decode”

It avoids repeated prefill. Stable TPOT across E007/E008 is evidence of that
boundary.

### Mistake 6: “Higher closed-loop requests/s proves higher capacity”

Concurrency-1 throughput reflects shorter elapsed request time. Capacity needs
open-loop offered-load tests.

### Mistake 7: “All requests succeeded, so the SLO passed”

E009 had zero errors in every load condition, yet three of four conditions had
zero SLO goodput.

### Mistake 8: “RAG benefits from caching”

Some RAG request layouts benefit. E009 friendly and hostile layouts had the
same document and total length but radically different results.

### Mistake 9: “Agent-like proves agents are faster”

It proves that a synthetic large-stable-prefix, growing-state request pattern
benefited. It does not measure tool execution, reasoning quality, or task
completion.

### Mistake 10: “The profile with the highest hit fraction is always best”

Output length, absolute latency, queueing, cache memory, and application quality
also matter.

### Mistake 11: “Requests in one run are independent replications”

They share one server lifetime and deliberately share cache state. Canonical
uncertainty requires independent repetitions, not treating related requests as
independent experiments.

### Mistake 12: “A dirty environment means uncommitted source changed”

On the ephemeral host, generated run artifacts made the repository dirty. The
recorded commit identifies the source. Auditing should still confirm no source
edits were made.

---

## 20. Limitations and Follow-Up Experiments

E007–E009 use one model family, one RTX 3090, one vLLM version, one cache-block
size, and one canonical repetition per condition. Most controlled runs use
concurrency 1 and resident working sets.

They do not establish:

- eviction behavior;
- cache capacity or working-set limits;
- long reuse distances;
- multi-tenant interference;
- distributed or cross-replica caching;
- alternative block sizes or cache dtypes;
- multiple models and GPUs;
- response or application quality;
- production arrival distributions;
- confidence intervals from independent repetitions.

High-value follow-ups include:

1. **Reuse distance sweep:** insert unrelated prefixes between repeated uses.
2. **Working-set sweep:** increase the number and total size of active prefixes
   until eviction appears.
3. **Popularity distribution:** compare uniform, round-robin, and Zipf reuse.
4. **Multi-tenant isolation:** mix unrelated users and test routing policies.
5. **Load sweep:** repeat friendly/hostile application cases across several
   offered rates to locate saturation boundaries.
6. **Independent repetitions:** quantify uncertainty and time-order effects.
7. **Model/hardware replication:** test whether response curves transfer.
8. **Quality-aware prompt layouts:** verify that cache-friendly reordering does
   not harm answer quality.
9. **Cache-memory tradeoff:** measure active-sequence capacity, preemptions, and
   goodput as prefix residency consumes KV memory.
10. **End-to-end agent tasks:** combine inference savings with tool latency,
    number of calls, and task success.

---

## 21. Interview-Ready Explanations

### “What is automatic prefix caching?”

It reuses model KV tensors from previously processed leading token blocks when
a later request begins with the exact same tokens. It avoids repeated prefill
for those blocks, primarily reducing time to first token. It does not reuse the
answer or eliminate decode for new output tokens.

### “How did you prove the speedup came from caching?”

E007 used a 2×2 design: unique versus shared prefixes, each with cache OFF and
ON. Cache ON produced no hits and no benefit for unique prompts. Shared/cache ON
produced exactly the planned hit-token count and a 68.4% median TTFT reduction,
while TPOT stayed unchanged. Live server configuration and measured-window
Prometheus deltas were also verified.

### “Why include cache OFF for every E008 prefix fraction?”

It controls for prompt content and time-order effects. The cache-OFF medians
stayed near 870–895 ms across fractions, showing that the large cache-ON trend
came from verified reuse rather than different deterministic token content.

### “Why is observed hit fraction below configured fraction?”

The first request is a compulsory miss, and only complete cache blocks count.
For 30 requests sharing 3,072 of 4,096 tokens, the window fraction is
`29×3072 / 30×4096 = 72.5%`, while steady state is 75%.

### “What was the most important E009 finding?”

Prompt ordering determines whether repeated content is a reusable prefix. With
the same document and total length, `[system][document][question]` reduced TTFT
by 74.31%, while `[system][question][document]` improved only 4.64% because the
changing question caused early divergence.

### “Why did friendly cache ON pass the load SLO?”

Its long cached prefix reduced service time enough to prevent a large backlog at
1 request/s. Maximum in-flight work fell from 15 to four, TTFT P95 fell to 0.398
seconds, and E2E P95 fell to 2.419 seconds. This crossed the selected SLO
boundary. Hostile caching saved too little prefill to prevent accumulation.

### “Why didn't E2E improve as much as TTFT?”

E2E contains both prefill/first-token time and decode time. Prefix caching
reduces the former, but every new output token still has to be generated. This
is especially visible in agent-like calls with larger output caps.

### “How would you apply this in production?”

I would measure real common-prefix distributions, stabilize templates and tool
serialization, place stable content before changing content when semantically
safe, use cache-aware routing carefully, and evaluate observed hits together
with TTFT, SLO goodput, memory pressure, and application quality. I would not
optimize hit rate in isolation.

---

## 22. Exercises

### Exercise 1: Compulsory miss

Twenty requests each contain 2,048 input tokens and share a block-aligned
1,024-token prefix. Assume no eviction. What measured-window hit fraction is
expected?

### Exercise 2: Block rounding

Two prompts share 1,003 leading tokens and the cache block size is 16. How many
complete prefix tokens are cacheable?

### Exercise 3: Mechanism diagnosis

A cache-ON run reports a 60% intended shared fraction, zero observed hit tokens,
and a 30% TTFT reduction. Can the reduction be attributed to prefix caching?
What should you investigate?

### Exercise 4: Latency decomposition

A request's TTFT falls from 1,000 ms to 250 ms. It generates 65 tokens with a
TPOT of 20 ms. Approximate E2E before and after caching and the E2E percentage
change.

### Exercise 5: Prompt ordering

Which layout is more cache-friendly for repeated questions about one document,
and why?

```text
A: [system][document][question]
B: [system][question][document]
```

### Exercise 6: Audit

Name at least six artifacts needed to audit a prefix-cache condition and state
what each proves.

### Exercise 7: Load interpretation

A load run offers 1 request/s, completes every request without errors, achieves
0.8 requests/s over the measured-and-drain duration, and has zero goodput. Is
the server reliable? Is it meeting the service objective?

### Exercise 8: Agent design

An agent randomizes tool-schema order on every model call. What effect can that
have on prefix caching, and how would you fix it without changing the tools?

### Exercise 9: Experiment extension

Design a minimal experiment to study cache eviction. Identify the independent
variable, controls, and primary measurements.

### Exercise 10: Claim boundary

Can E009 support the claim “prefix caching improves agent task success”? Explain
what it does support and what additional measurement would be required.

---

## 23. Exercise Solutions

### Solution 1

The first request misses; 19 hit the 1,024-token prefix:

```text
(19 × 1,024) / (20 × 2,048) = 0.475 = 47.5%
```

Steady-state reusable fraction is 50%, but the full window is 47.5%.

### Solution 2

```text
floor(1,003 / 16) × 16 = 62 × 16 = 992 tokens
```

### Solution 3

No. Zero observed hits contradict the proposed mechanism. Investigate whether
the live server actually enabled caching, token sequences matched, prefixes
were block aligned and resident, metric names/units were correct, the window was
captured correctly, and another uncontrolled difference affected latency.

### Solution 4

There are 64 post-first-token intervals:

```text
decode time = 64 × 20 ms = 1,280 ms
before E2E ≈ 1,000 + 1,280 = 2,280 ms
after E2E  ≈   250 + 1,280 = 1,530 ms
change = (1,530 - 2,280) / 2,280 ≈ -32.9%
```

TTFT improves 75%, but E2E improves only about 32.9%.

### Solution 5

Layout A. Requests remain identical through the system prompt and document, so
only the final question diverges. Layout B diverges at the question before the
repeated document; later matching tokens do not restore the broken prefix.

### Solution 6

One valid answer:

- `config.json`: intended condition;
- `environment.json`: software/hardware/commit;
- `workload.jsonl`: token relationships and expected states;
- `requests.jsonl`: per-request performance and success;
- `server_cache_config.json`: effective server setting;
- before/after `.prom` files: raw cumulative counters;
- `cache_metrics_delta.json`: condition-window hits and queries;
- condition JSON: mechanism verdict;
- `in_flight.jsonl`: arrival and backlog behavior for open-loop runs.

### Solution 7

It is reliable in the narrow sense that requests complete without errors. It is
not meeting the service objective: work arrives faster than it drains over the
window and no request satisfies the complete SLO, so goodput is zero.

### Solution 8

Random ordering changes early tokens and can fragment or eliminate the shared
tool prefix. Sort schemas by a stable key, use deterministic JSON
serialization, keep whitespace/template versions fixed, and verify token hashes
across calls.

### Solution 9

Keep model, prompt length, cached prefix size, request count, and server settings
fixed. Vary the number or total bytes of competing prefixes inserted between
two uses of a target prefix. Measure target hit tokens, TTFT, KV-cache usage,
preemptions, and the point where the target changes from hit to miss. Restart
the server for each independent working-set point.

### Solution 10

No. E009 measures synthetic inference request performance, not task quality. It
supports the claim that an agent-shaped request sequence with a large stable
prefix had verified cache reuse and much lower TTFT. A task-success claim needs
real agent trajectories, tool outcomes, quality evaluation, total task time,
and success-rate measurements.

---

## 24. Glossary

**Automatic prefix caching (APC):** Reuse of KV blocks for an identical leading
token sequence across requests.

**Block size:** Number of tokens in the cache allocation/matching granularity
used by the tested server configuration.

**Cache hit token:** A prompt token whose required KV state is served from an
eligible cached prefix block according to the server metric.

**Cache query token:** A prompt token considered by the server's prefix-cache
accounting during prefill.

**Cache residency:** Whether previously created prefix blocks remain available
in cache memory.

**Common prefix:** The longest identical leading token sequence shared by two
prompts.

**Compulsory miss:** The first access to a prefix before its blocks have been
populated.

**Controlled run:** A low-concurrency experiment designed to isolate a causal
mechanism rather than production load behavior.

**E2E latency:** Time from request start until the response finishes.

**Eviction:** Removal of cached blocks to reclaim memory.

**Goodput:** Rate of requests that complete while satisfying the defined SLO.

**Hit fraction:** Hit tokens divided by queried prompt tokens over a defined
counter window.

**In-flight request:** A dispatched request that has not yet completed. This is
not necessarily the same as server queue depth.

**KV cache:** Stored attention key and value tensors for processed tokens.

**Measured window:** The exact interval between before and after counter
snapshots used for condition metrics.

**Open-loop load:** Requests arrive according to an external schedule rather
than waiting for earlier completions.

**Prefill:** Parallel processing of input tokens to build model state and
produce the first output token.

**Prompt identity:** Exact token values and order, not semantic similarity.

**Relationship group:** Conversation, document, trajectory, or other identifier
linking prompts that intentionally reuse content.

**Reuse distance:** Amount of intervening work between two uses of the same
prefix.

**SLO:** Service-level objective defining acceptable latency and error behavior.

**Steady state:** Requests after the target prefix has been populated and before
it is evicted.

**Throughput:** Rate of successful request completions, regardless of whether
they meet latency objectives.

**TPOT:** Average time per output token after the first generated token.

**TTFT:** Time from request start to receipt of the first generated token.

**Unique-prefix control:** Prompts deliberately made distinct within the first
cacheable block so that reuse should be zero.

**Working set:** Collection of prefixes competing for cache residency during an
interval.

---

## 25. Final Mental Model

Remember V4 as five layers:

```text
1. Identity
   Are the leading tokens exactly the same and in the same order?

2. Eligibility
   Do they form complete cacheable blocks, and has an earlier request populated them?

3. Residency
   Are those blocks still present on the server handling this request?

4. Mechanism
   Do observed server counters show the expected cache hits?

5. Outcome
   Did TTFT, E2E, throughput, queueing, or SLO goodput improve?
```

Then remember the experiment sequence:

```text
E007: prove the mechanism with controls
E008: measure the reuse-size response curve
E009: apply it to prompt relationships and load
```

And the practical rule:

> Put stable, frequently reused, semantically valid content first; keep its
> tokenization deterministic; verify real cache hits; and judge success by
> user-facing latency and SLO goodput, not by the cache flag or hit rate alone.
