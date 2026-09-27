# E005 Open-Loop Load, Saturation, and SLO Goodput — Validation Report

## Executive Summary

E005 applied deterministic open-loop traffic to the synthetic chat-like workload
at 0.5, 1, 2, 3, and 4 requests/s. Each point scheduled the same deterministic
30-request plan. All 150 measured requests completed successfully, with zero
client-overload drops, drain timeouts, or request errors.

Completed throughput tracked offered load closely at 0.5 and 1 request/s, then
increasingly diverged: it reached 1.715, 2.109, and 2.359 requests/s at offered
rates of 2, 3, and 4 requests/s. Peak in-flight work simultaneously rose from 3
to 15 requests. The declining marginal throughput gain, accumulating in-flight
work, rising E2E latency, and near-flat goodput from 3 to 4 requests/s place the
observed saturation region around and above 2 requests/s for this workload and
configuration. The five discrete points do not locate a precise saturation knee.

No tested point passed the complete configured SLO. TTFT P95 remained below its
1-second objective and the error objective passed everywhere, but E2E P95 was
already 6.560 seconds at 0.5 requests/s, above the 5-second objective. Therefore
the correct sustainable rate under the exact configured aggregate SLO is not
identified by this sweep: it is either below 0.5 requests/s or the 5-second E2E
objective is incompatible with the long-output tail of this workload on the
measured system.

## Run Identity and Environment

| Item | Value |
| :--- | :--- |
| Run ID | `E005_20260927_063839_565f514e` |
| Source commit | `1d88839f13e3c8d33b5e991e569f08a460b0a319` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| GPU | 1× NVIDIA GeForce RTX 3090, 24,576 MiB |
| Driver | 580.173.02 |
| PyTorch | 2.13.0+cu132 |
| vLLM | 0.30.0 |
| Python | 3.12.14 |
| Arrival process | Deterministic constant rate, open loop |
| Requests | 5 warm-up + 30 measured per rate; 150 measured total |
| Offered rates | 0.5, 1, 2, 3, and 4 requests/s |
| Prefix caching / chunked prefill | Disabled / disabled |
| Sampling | Temperature 0.0, seed 42, stratified workload plan |
| Request inactivity timeout | 300 seconds |
| Client safety limits | 64 in flight, 300-second drain deadline |

The environment snapshot reports a dirty tree only because `environment.txt`
and the generated run directory were untracked on the ephemeral host. The
captured source revision is the merged E005 commit on `main`; no source edits
were present during execution.

## Workload and Accounting Validation

Every load point used the same realized workload plan:

- 30 measured requests and 5 warm-ups;
- 10,752 actual input tokens;
- 2,688 actual output tokens;
- no prompt reuse;
- the E003 chat-like discrete input and output distributions.

For every point, the raw artifacts contain 30 measured request records, 30
measured workload records, 30 dispatch events, and 30 completion events. All
offered arrivals were accounted for. There were no drops, cancellations, or
missing records.

## Aggregate Results

| Offered req/s | Achieved req/s | Offered achieved | Goodput req/s | Compliance | TTFT P95 | E2E P95 | Peak in flight | Avg GPU |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5 | 0.503 | 100.6% | 0.453 | 90.0% | 338.93 ms | 6.560 s | 3 | 90.50% |
| 1.0 | 0.974 | 97.4% | 0.877 | 90.0% | 365.28 ms | 7.046 s | 4 | 98.91% |
| 2.0 | 1.715 | 85.7% | 1.543 | 90.0% | 359.70 ms | 7.831 s | 9 | 93.44% |
| 3.0 | 2.109 | 70.3% | 1.828 | 86.7% | 360.26 ms | 7.950 s | 11 | 94.69% |
| 4.0 | 2.359 | 59.0% | 1.888 | 80.0% | 382.22 ms | 8.082 s | 15 | 100.00% |

Achieved throughput uses the full measurement window, including the time needed
to drain requests after arrivals stopped. This is why it can be lower than the
configured arrival rate even though every request eventually succeeds.

## Saturation Analysis

Three signals move together as load increases:

1. **Throughput divergence.** Achieved throughput follows offered traffic below
   1 request/s, reaches only 85.7% of offered load at 2 requests/s, and only
   59.0% at 4 requests/s.
2. **Accumulating work.** Peak in-flight requests rise from 3 at 0.5 requests/s
   to 15 at 4 requests/s.
3. **Diminishing useful capacity.** Raising offered load from 3 to 4 requests/s
   increases raw throughput by only 0.251 requests/s and goodput by only 0.060
   requests/s. Compliance falls from 86.7% to 80.0%.

These observations support describing 2–4 requests/s as a saturation region,
with substantial pressure clearly present by 3 requests/s. They do not justify
a single exact knee. A denser repeat around 1–3 requests/s and longer high-rate
arrival windows would be needed to estimate one more precisely.

GPU utilization supports, but does not define, this conclusion. Average GPU
utilization was already 98.9% at 1 request/s, dipped to 93–95% at the next two
shorter points, and reached 100% at 4 requests/s. End-to-end behavior—not GPU
utilization alone—is the capacity signal.

## SLO and Goodput Analysis

The configured load-point SLO was:

- TTFT P95 at or below 1 second;
- E2E P95 at or below 5 seconds;
- error rate at or below 1%.

TTFT P95 remained between 339 and 382 ms, and error rate remained zero, so both
objectives passed at every load point. E2E P95 ranged from 6.560 to 8.082
seconds, so the aggregate E2E objective failed everywhere.

Goodput uses a per-request definition rather than the aggregate percentile:
successful requests with TTFT at most 1 second and E2E latency at most 5 seconds
per measured wall-clock second. At 0.5, 1, and 2 requests/s, 27 of 30 requests
met those individual limits. This is compatible with aggregate P95 failure:
three slow requests are enough to place the interpolated P95 above 5 seconds in
a 30-request sample.

Goodput rose through 3 requests/s but nearly plateaued at the highest point:
1.828 to 1.888 requests/s while offered load rose by a full request/s. Raw
throughput still rose because late completions count as throughput even when
they no longer meet the latency objective.

## Hypothesis Evaluation

| Hypothesis | Empirical result | Status |
| :--- | :--- | :--- |
| Open-loop traffic exposes pressure hidden by closed-loop clients. | Offered/achieved divergence and in-flight work increased with rate. | Confirmed |
| Throughput eventually grows more slowly than offered load. | Achieved/offered fell from 100.6% to 59.0%. | Confirmed |
| Latency and queue pressure grow near saturation. | E2E P95 rose 23.2%; peak in flight rose from 3 to 15. | Confirmed |
| Goodput separates useful service from raw completions. | At 4 req/s, throughput was 2.359 but goodput was 1.888 req/s. | Confirmed |
| The configured SLO has a passing point in the tested range. | E2E P95 exceeded 5 seconds at every point. | Rejected |
| GPU utilization alone locates saturation. | Utilization was near maximum before strong throughput divergence appeared. | Rejected |

## Artifacts

- Raw canonical run: [`runs/E005_20260927_063839_565f514e/`](../runs/E005_20260927_063839_565f514e/)
- Capacity and goodput: [`plots/e005/throughput_and_goodput_vs_offered_rate.png`](plots/e005/throughput_and_goodput_vs_offered_rate.png)
- Latency scaling: [`plots/e005/latency_vs_offered_rate.png`](plots/e005/latency_vs_offered_rate.png)
- Pressure and SLO health: [`plots/e005/pressure_and_slo_vs_offered_rate.png`](plots/e005/pressure_and_slo_vs_offered_rate.png)

The raw run retains all five point configurations, environment snapshots,
per-request measurements, exact workload plans, GPU telemetry, in-flight event
timelines, summaries, plots, package inventory, GPU report, and source revision.

## Limitations

1. Each point has 30 requests. P95 is descriptive and P99 is not a stable tail
   estimate at this sample size.
2. The 3 and 4 requests/s arrival windows lasted only 10 and 7.5 seconds. They
   expose transient overload but do not establish long-duration steady state.
3. This is one deterministic workload ordering and one run per rate. Repeated
   seeds are needed to quantify run-to-run variance.
4. No point passed the aggregate E2E SLO, so this sweep does not establish a
   non-zero sustainable rate under that exact SLO.
5. The server's internal queue delay is not directly observable through the
   OpenAI-compatible API. It is included in TTFT; in-flight count is an
   end-to-end pressure indicator, not a direct server queue measurement.
6. Prefix caching and chunked prefill were disabled.
7. Conclusions apply only to the recorded model, workload, hardware, backend,
   software environment, and SLO.

## Conclusion

E005 validates the open-loop measurement system and exposes a clear transition
from load tracking to accumulating pressure. The server completes every request,
but higher offered traffic produces diminishing throughput gains, increasing
in-flight work, longer E2E tails, and a goodput plateau. This is the distinction
between raw completion capacity and useful SLO-compliant capacity.

The configured 5-second E2E P95 objective is not met even at the lightest tested
load. That is an important result rather than an execution error: sustainable
capacity is defined jointly by workload and SLO, and an objective can be tighter
than the workload's baseline service-time tail.
