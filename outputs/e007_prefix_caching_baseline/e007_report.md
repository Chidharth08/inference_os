# E007 Prefix-Caching Mechanism Validation — Results

## Executive Summary

E007 completed successfully. The controlled 2×2 experiment showed that merely
enabling prefix caching did not improve requests with unique prefixes, while
requests sharing an exact 3,072-token prefix saw a 68.4% reduction in median
time to first token (TTFT). All 120 canonical requests completed successfully,
and every workload, server-configuration, and cache-mechanism check passed.

The negative control was stable: enabling caching for unique 4,096-token
prompts changed TTFT P50 from 869.60 ms to 874.86 ms (+0.6%) and produced zero
cache-hit tokens. With a shared 3,072-token prefix, enabling caching reduced
TTFT P50 from 872.51 ms to 275.57 ms (-68.4%). The cache-enabled shared-prefix
condition recorded 89,088 hit tokens from 122,880 queried tokens, an observed
hit fraction of 72.5%.

These results validate both the expected vLLM prefix-caching mechanism and the
benchmark's ability to distinguish real prefix reuse from cache enablement
alone. They motivate E008's controlled sweep over reusable-prefix fraction.

## Environment and Protocol

| Item | Value |
| :--- | :--- |
| Source commit | `2f39dc97df828372721a7d9062f2791e0f4e32c6` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| GPU | 1× NVIDIA GeForce RTX 3090, 24,576 MiB |
| Driver | 595.71.05 |
| PyTorch / vLLM | 2.13.0+cu132 / 0.30.0 |
| Prompt shape | 4,096 input tokens, maximum 64 output tokens |
| Shared-prefix shape | 3,072 shared + 1,024 unique tokens |
| Cache block size | 16 tokens |
| Concurrency | 1 |
| Canonical sample | 3 disjoint warm-ups + 30 measured requests per condition |
| Sampling | Temperature 0.0, seed 42 |
| Chunked prefill | Disabled |

Each condition used a fresh vLLM process. The runner verified the effective
server cache configuration before traffic, required pristine cache counters for
cache-enabled runs, used warm-up prompts disjoint from measured prefixes, and
calculated Prometheus counter deltas only across the measured window.

The environment marked the source tree dirty because run artifacts were being
created on the ephemeral host. The captured commit is the pushed E007
implementation used for every condition.

## Canonical Results

| Prompt pattern | Cache | TTFT P50 | TTFT P95 | E2E P50 | TPOT P50 | Requests/s | Cache-hit fraction |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unique | OFF | 869.60 ms | 939.11 ms | 2,124.76 ms | 19.85 ms | 0.470 | N/A |
| Unique | ON | 874.86 ms | 941.30 ms | 2,134.75 ms | 19.93 ms | 0.467 | 0.0% |
| Shared | OFF | 872.51 ms | 936.84 ms | 2,128.55 ms | 19.86 ms | 0.469 | N/A |
| Shared | ON | **275.57 ms** | **360.92 ms** | **1,524.73 ms** | 19.87 ms | **0.643** | **72.5%** |

The shared-prefix/cache-ON run improved request throughput by approximately
37.3% relative to its cache-OFF control (0.643 versus 0.469 requests/s). This is
a closed-loop, concurrency-1 result, so it is an elapsed-time consequence of
the reduced prefill latency rather than a production-capacity measurement.

## Mechanism Evidence

| Condition | Query tokens | Hit tokens | Observed hit fraction | Expected outcome |
| :--- | ---: | ---: | ---: | :--- |
| Unique, cache OFF | 0 | 0 | N/A | No cache accounting |
| Unique, cache ON | 122,880 | 0 | 0.0% | Negative control; no reusable prefix |
| Shared, cache OFF | 0 | 0 | N/A | Reuse unavailable |
| Shared, cache ON | 122,880 | 89,088 | 72.5% | Repeated prefix reused after compulsory miss |

The 72.5% aggregate hit fraction exactly matches the planned steady-state
behavior: the first request is a compulsory miss, followed by 29 requests that
reuse 3,072 of 4,096 tokens:

```text
(29 × 3,072) / (30 × 4,096) = 0.725
```

The first shared-prefix/cache-ON request remained cold at 860.84 ms TTFT. The
29 warm requests had a median TTFT of 275.35 ms, a 68.4% reduction relative to
the cache-OFF shared-prefix median. This agreement between planned reuse,
observed hit counters, and latency isolates prefix caching as the explanation.

## Interpretation

Three results matter together:

1. The unique-prefix control produced no cache hits and no meaningful TTFT
   benefit when caching was enabled.
2. The repeated-prefix condition produced exactly the expected measured-window
   hit-token count.
3. Only the condition with verified cache hits produced a large TTFT reduction.

TPOT remained effectively unchanged at approximately 19.9 ms/token across all
four conditions. That is the expected shape: automatic prefix caching avoids
repeated prefill computation but does not accelerate autoregressive decoding.

## Validity and Limitations

E007 validates the mechanism for one model, one RTX 3090, vLLM 0.30.0, a fixed
4,096-token prompt, a 75% reusable prefix, 16-token cache blocks, sequential
requests, and a cache large enough to avoid pressure. It does not establish the
benefit at other prefix fractions, under concurrency, with multiple reuse
groups, during eviction, or for application-shaped traffic.

Only one canonical repetition per condition was collected. The large effect,
exact counter agreement, and negative controls make the mechanism conclusion
strong, but repeated runs would be needed for uncertainty estimates or claims
about small differences.

## Artifacts

- Machine-readable comparison: `e007_summary.json`
- Latency comparison: `plots/latency_by_condition.png`
- Cache evidence: `plots/cache_observations_by_condition.png`
- Comparison run: `runs/E007_20260930_171552_25dca2e5`
- Canonical condition runs: `runs/E007_20260930_164409_6b97a6da`,
  `runs/E007_20260930_164849_598f4bb8`,
  `runs/E007_20260930_165232_747e60ec`, and
  `runs/E007_20260930_165451_5243ea36`

The four pilot runs are also retained under `runs/` for audit. Each condition
directory includes its configuration, environment, realized workload,
request-level measurements, GPU telemetry, raw Prometheus snapshots, cache
counter delta, verified server cache configuration, and mechanism verdict.
