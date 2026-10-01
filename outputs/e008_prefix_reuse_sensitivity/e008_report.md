# E008 Prefix Reuse Sensitivity — Results

## Executive Summary

E008 completed successfully. Across five paired cache-OFF/cache-ON conditions,
verified cache reuse increased monotonically with shared-prefix size and TTFT
fell correspondingly. All 300 canonical requests completed successfully, every
server and request-plan check passed, and every observed cache-hit count exactly
matched the token-level expectation.

Enabling prefix caching with unique prompts changed TTFT P50 by only +0.14%,
confirming that cache enablement alone did not improve latency. At requested
shared-prefix fractions of 25%, 50%, 75%, and 90%, TTFT P50 decreased by 20.36%,
41.80%, 68.81%, and 79.49%, respectively. The observed measured-window hit
fractions rose from 24.17% to 86.85%; steady-state fractions, which exclude the
first compulsory miss, were exactly 25%, 50%, 75%, and 89.84%.

The result supports the E008 hypothesis: under this controlled sequential
workload, prefix-caching benefit scales strongly with the fraction of prefill
work that can be reused.

## Environment and Protocol

| Item | Value |
| :--- | :--- |
| Source commit | `0b66677d3045d4865ce6ff47e726203b84b98625` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| GPU | 1× NVIDIA GeForce RTX 3090, 24,576 MiB |
| Driver | 590.48.01 |
| PyTorch / vLLM | 2.13.0+cu132 / 0.30.0 |
| Prompt shape | 4,096 input tokens, maximum 64 output tokens |
| Requested shared fractions | 0%, 25%, 50%, 75%, 90% |
| Cache block size | 16 tokens |
| Concurrency | 1 |
| Canonical sample | 3 disjoint warm-ups + 30 measured requests per condition |
| Canonical conditions | 10: paired cache OFF and ON at each fraction |
| Sampling | Temperature 0.0, seed 42 |
| Chunked prefill | Disabled |

Every condition used a fresh vLLM process. Cache-enabled runs required pristine
prefix-cache counters before traffic. Warm-up prompts were token-distinct from
the measured reuse group, and cache deltas covered only the measured window.

The environment recorded a dirty source tree because run artifacts were created
on the ephemeral host. The captured commit is the pushed E008 implementation;
no source edits were used during execution.

## Prefix Resolution and Cache Evidence

| Requested fraction | Resolved prefix | Resolved fraction | Expected hits | Observed hits | Window hit fraction | Steady-state fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 0 | 0.00% | 0 | 0 | 0.00% | 0.00% |
| 25% | 1,024 | 25.00% | 29,696 | 29,696 | 24.17% | 25.00% |
| 50% | 2,048 | 50.00% | 59,392 | 59,392 | 48.33% | 50.00% |
| 75% | 3,072 | 75.00% | 89,088 | 89,088 | 72.50% | 75.00% |
| 90% | 3,680 | 89.84% | 106,720 | 106,720 | 86.85% | 89.84% |

The requested 90% boundary equals 3,686.4 tokens and was deliberately rounded
down to 3,680, the nearest complete 16-token cache block. Both values are
retained in the raw artifacts.

The measured-window fraction is lower than the steady-state fraction because
request 1 must populate the prefix. For example, the 75% condition produced:

```text
(29 × 3,072) / (30 × 4,096) = 0.725
```

The exact agreement at all five points validates prompt construction, request
ordering, measured-window accounting, and vLLM cache behavior together.

## Latency Results

| Shared fraction | Cache-OFF TTFT P50 | Cache-ON TTFT P50 | TTFT change | Cache-OFF TTFT P95 | Cache-ON TTFT P95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 872.87 ms | 874.06 ms | +0.14% | 907.84 ms | 882.03 ms |
| 25% | 870.30 ms | 693.11 ms | -20.36% | 898.57 ms | 747.08 ms |
| 50% | 874.57 ms | 508.98 ms | -41.80% | 918.33 ms | 531.72 ms |
| 75% | 894.96 ms | 279.10 ms | -68.81% | 918.27 ms | 351.38 ms |
| 90% | 877.21 ms | 179.93 ms | -79.49% | 886.00 ms | 188.22 ms |

Cache-OFF TTFT remained between 870 and 895 ms across all prompt identities,
showing that the underlying fixed-length workload was stable. The 0% cache-ON
negative control was also effectively unchanged. Large reductions appeared only
when verified cache hits were present.

The steady-state P50 reductions were -20.36%, -41.91%, -68.83%, and -79.49% at
the four positive reuse levels. Their close agreement with whole-window P50
changes reflects the single compulsory miss among 30 measurements.

## End-to-End, Decode, and Throughput Results

| Shared fraction | E2E P50 OFF | E2E P50 ON | E2E change | TPOT P50 OFF | TPOT P50 ON | Requests/s OFF | Requests/s ON |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0% | 2,112.54 ms | 2,120.03 ms | +0.35% | 19.70 ms | 19.78 ms | 0.472 | 0.472 |
| 25% | 2,111.09 ms | 1,952.56 ms | -7.51% | 19.73 ms | 19.92 ms | 0.473 | 0.511 |
| 50% | 2,117.56 ms | 1,730.29 ms | -18.29% | 19.76 ms | 19.56 ms | 0.471 | 0.571 |
| 75% | 2,120.65 ms | 1,553.63 ms | -26.74% | 19.76 ms | 19.74 ms | 0.470 | 0.639 |
| 90% | 2,125.43 ms | 1,424.72 ms | -32.97% | 19.79 ms | 19.75 ms | 0.471 | 0.690 |

E2E benefit was smaller than TTFT benefit because each request still performed
the same decode work. TPOT stayed within 0.36 ms/token across every paired
condition, supporting the expected interpretation that prefix caching reduces
prefill rather than autoregressive decode time.

At concurrency 1, request throughput rose by 8.15%, 21.32%, 36.03%, and 46.62%
at positive reuse levels. These figures describe sequential elapsed-time benefit;
they are not production-capacity measurements under concurrent or open-loop
load.

## Conclusions

E008 establishes four findings for the measured environment:

1. Cache enablement without reusable prefixes provides no meaningful TTFT or
   throughput improvement.
2. Observed cache reuse grows monotonically with the resolved shared-prefix
   fraction.
3. TTFT reduction grows strongly with verified reusable prefill, reaching 79.5%
   at the 89.84% resolved prefix.
4. TPOT remains stable, separating prefill savings from decode performance.

The relationship is not assumed from configured prompt percentages: every point
is backed by exact token identity, expected cache-block accounting, and observed
vLLM counters.

## Validity and Limitations

E008 uses one model, one RTX 3090, one vLLM version, one 4,096-token prompt
length, a 64-token output limit, one reuse group, concurrency 1, and a resident
working set. It does not establish behavior under eviction, cache pressure,
multiple tenants, varying prompt lengths, concurrent requests, or realistic
application request relationships.

One canonical repetition was collected for each condition. The mechanism
evidence and large monotonic effects support the qualitative conclusion, but
repeated sweeps would be needed for confidence intervals or claims about small
differences.

## Artifacts

- Machine-readable comparison: `outputs/e008_prefix_reuse_sensitivity/e008_summary.json`
- Published plots: `outputs/e008_prefix_reuse_sensitivity/plots/`
- Comparison run: `runs/E008_20261001_062832_49edaab1`
- Seven pilot and ten canonical condition directories: `runs/E008_20261001_*`

Every condition directory retains its configuration, environment, realized
workload, request measurements, GPU telemetry, raw Prometheus snapshots, cache
delta, verified server cache configuration, and E008 mechanism verdict.
