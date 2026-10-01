# E009 Application-Shaped Prefix Reuse — Results

## Executive Summary

E009 completed successfully. All 336 canonical requests across ten controlled
and four open-loop conditions completed without errors. Every request-plan and
server-configuration check passed. Cache-OFF conditions recorded zero cache
hits, while every controlled cache-ON condition exactly matched its planned
token-level hit count.

The controlled results show that application labels alone do not predict cache
benefit; prompt ordering and the size of the stable leading region do. The
agent-like profile had the largest observed hit fraction (87.56%) and reduced
TTFT P50 by 84.73%. Chat-like and cache-friendly RAG profiles also benefited
strongly, reducing TTFT P50 by 69.29% and 74.31%. Cache-hostile RAG and
summarization-like prompts exposed only a short reusable prefix (5.99% observed
hit fraction), so their TTFT improvements were limited to 4.64% and 4.48%.

At an offered rate of 1 request/s, only cache-friendly RAG with caching enabled
met the complete latency and error SLO. Its TTFT P95 was 0.398 seconds, E2E P95
was 2.419 seconds, and goodput was 0.922 requests/s. Moving the same document
behind the changing question prevented most reuse: cache-hostile RAG with
caching enabled had TTFT P95 of 2.988 seconds and E2E P95 of 14.297 seconds and
did not meet the SLO.

## Environment and Protocol

| Item | Value |
| :--- | :--- |
| Source commit | `0677295a7e1b4a5dc7fe8e0a56017fca6ee790eb` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| GPU | 1× NVIDIA GeForce RTX 3090, 24,576 MiB |
| Driver | 580.142 |
| PyTorch / vLLM | 2.13.0+cu132 / 0.30.0 |
| Python | 3.12.14 |
| Cache block size | 16 tokens |
| Controlled conditions | 10: five profiles paired cache OFF and ON |
| Controlled sample | 3 disjoint warm-ups + 24 measured requests per condition |
| Open-loop conditions | 4: two RAG layouts paired cache OFF and ON |
| Open-loop load | 24 requests offered at 1 request/s per condition |
| Sampling | Temperature 0.0, seed 42 |
| Chunked prefill | Disabled |

Every condition used a fresh vLLM process. Cache-enabled controlled runs
required pristine cache counters, and measured cache deltas excluded warm-up
traffic. The environment reports a dirty source tree because result artifacts
were created on the ephemeral host; the recorded source commit is the pushed
E009 implementation.

## Controlled Results

| Profile | Cache hit fraction | TTFT P50 OFF | TTFT P50 ON | TTFT change | E2E P50 change | Throughput change |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chat-like | 76.85% | 579.49 ms | 177.99 ms | -69.29% | -19.06% | +24.53% |
| RAG-friendly | 77.86% | 1,065.40 ms | 273.67 ms | -74.31% | -31.31% | +43.03% |
| RAG-hostile | 5.99% | 1,070.64 ms | 1,020.98 ms | -4.64% | -1.71% | +1.13% |
| Summarization-like | 5.99% | 1,071.48 ms | 1,023.44 ms | -4.48% | -2.20% | +2.42% |
| Agent-like | 87.56% | 1,828.07 ms | 279.12 ms | -84.73% | -27.67% | +34.58% |

The cache-ON hit totals were 42,496 tokens for chat-like, 76,544 for
RAG-friendly, 5,888 for RAG-hostile, 5,888 for summarization-like, and 142,560
for agent-like. Each total exactly matched the corresponding request plan.

Chat-like and agent-like reuse increases as conversation or trajectory state
grows. Their strong results demonstrate the value of repeated sequential calls
that retain a large stable leading context. Agent-like E2E improvement is
smaller than its TTFT improvement because its longer generated outputs still
require decode work.

The RAG comparison isolates ordering. Both layouts contain the same 256-token
system prompt, 3,072-token document, question content, 4,096-token total input,
and 64-token output limit. RAG-friendly places the document before the changing
question and exposes a 3,328-token stable prefix. RAG-hostile places the
question first, leaving only the 256-token system prefix reusable. The 74.31%
versus 4.64% TTFT reductions therefore support prompt ordering—not document
identity alone—as the causal explanation.

The summarization-like profile behaves similarly to RAG-hostile because only
its 256-token instruction prefix is shared; each document diverges immediately
afterward.

## Open-Loop Results

The SLO required TTFT P95 at or below 1 second, E2E P95 at or below 3 seconds,
and error rate at or below 1%.

| RAG layout | Cache | Throughput | Goodput | TTFT P95 | E2E P95 | SLO met |
| :--- | :---: | ---: | ---: | ---: | ---: | :---: |
| Friendly | OFF | 0.780 req/s | 0.000 req/s | 5.704 s | 15.913 s | No |
| Friendly | ON | 0.962 req/s | 0.922 req/s | 0.398 s | 2.419 s | Yes |
| Hostile | OFF | 0.788 req/s | 0.000 req/s | 5.451 s | 15.647 s | No |
| Hostile | ON | 0.848 req/s | 0.000 req/s | 2.988 s | 14.297 s | No |

All four load conditions dispatched all 24 offered requests with zero drops,
timeouts, or request failures. The cache-OFF conditions accumulated up to 15
in-flight requests because service time exceeded the interarrival interval.
Friendly cache-ON reduced the maximum to four and satisfied every SLO
objective. Hostile cache-ON reached 14 in flight and remained outside both
latency objectives despite its small cached prefix.

The cache-OFF throughput values below the 1 request/s offered rate reflect the
drain period included in measured duration. Goodput is zero whenever no request
meets the complete per-request latency SLO, even though all requests eventually
finish successfully.

## Conclusions

E009 establishes four findings for the measured environment:

1. Application-shaped workloads benefit when many calls share a large stable
   prefix, as demonstrated by chat-like, RAG-friendly, and agent-like traffic.
2. The same reusable content provides little benefit when changing content is
   placed before it; prefix caching depends on leading token identity.
3. Short shared instruction prefixes produce valid cache hits but only small
   end-to-end improvements for long prompts.
4. Under the selected 1 request/s load, cache-friendly prompt ordering changed
   RAG service from an SLO failure into an SLO pass; cache-hostile ordering did
   not.

## Validity and Limitations

E009 uses one model, one RTX 3090, one vLLM version, deterministic synthetic
request relationships, resident prefixes, and one canonical repetition per
condition. It does not execute real chat, retrieval, summarization, tools, or
agents and does not measure response quality, retrieval quality, tool success,
or task completion. It also does not characterize cache eviction,
multi-tenancy, distributed serving, or alternative offered rates.

The large controlled effects, exact cache accounting, and paired RAG layout
support the qualitative conclusions. Repeated runs and more load points would
be required for confidence intervals and capacity-boundary claims.

## Artifacts

- Controlled comparison: `outputs/e009_application_prefix_reuse/e009_summary.json`
- Open-loop comparison: `outputs/e009_application_prefix_reuse/e009_load_summary.json`
- Published plots: `outputs/e009_application_prefix_reuse/plots/`
- Controlled comparison run: `runs/E009_controlled_20261001_173303_29917f34`
- Open-loop comparison run: `runs/E009_load_20261001_173304_917d3f2c`
- Five pilot and fourteen canonical condition directories: `runs/E009_20261001_*`

Every condition directory retains its configuration, environment, realized
workload, request measurements, GPU telemetry, raw Prometheus snapshots, cache
delta, verified server cache configuration, and E009 mechanism verdict.
