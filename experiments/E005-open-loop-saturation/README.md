# E005 — Open-Loop Load, Saturation, and SLO Goodput

## Research question

What constant request rate can the synthetic chat-shaped workload sustain before
completed throughput stops tracking offered traffic or the configured latency and
error objectives fail?

E005 changes the load model, not the serving configuration. E001–E004 used
closed-loop workers: a worker sent its next request only after its current request
finished. That is useful for controlled concurrency, but a slowing server also slows
the client. E005 schedules arrivals from a clock, independently of completions, so
pressure is visible rather than hidden by client self-throttling.

## Controlled setup

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Arrival process: deterministic constant rate
- Offered rates: 0.5, 1, 2, 3, and 4 requests/s
- Requests: 30 scheduled arrivals per rate point (150 total)
- Arrival duration: `30 / offered rate` seconds per point
- Workload: the E003 synthetic chat-like token distribution
- Prefix caching and chunked prefill: disabled
- Temperature: 0
- One GPU and one vLLM server configuration for the complete sweep

Using the same 30-request plan at every rate controls workload composition while
the clock-derived duration preserves each requested arrival rate. These rates are
starting points, not universal capacity claims. If every point passes
comfortably, extend the upper end. If the lowest point is already overloaded, lower
the range. Preserve the original run and configuration when refining the sweep.

## SLO and goodput definitions

The default load-point objectives are:

- TTFT P95 at or below 1 second;
- E2E latency P95 at or below 5 seconds;
- total error/drop rate at or below 1%.

A single request contributes to goodput only when it succeeds, has a first-token
measurement, TTFT is at most 1 second, and E2E latency is at most 5 seconds.

```text
goodput = per-request SLO-compliant completions / measured wall-clock second
```

The aggregate P95 checks answer whether the load point passes the stated service
objectives. The per-request check supplies an additive numerator for goodput. Failed,
dropped, and drain-timed-out requests count as non-compliant.

`measured wall-clock second` starts when the arrival schedule starts and ends after
the last dispatched request completes or the drain deadline cancels remaining work.
It therefore includes backlog drain time. Each arrival window is `30 / rate` seconds
and remains recorded separately so it is not confused with the observation window.

## Overload safety and queue interpretation

`max_in_flight` is a safety boundary, not a closed-loop worker count. Scheduled
arrivals continue at the configured rate. If 64 dispatched requests are already in
flight, the next arrival is recorded immediately as `client_overload`; it is never
silently delayed or omitted. After arrivals stop, `max_drain_seconds` bounds how long
the client waits before cancelling remaining requests as `drain_timeout`.

The client records:

- scheduled arrival time;
- actual dispatch time and client dispatch delay;
- request start, first-token, and completion times;
- every dispatch, completion, and overload drop in the in-flight timeline.

The vLLM server's internal queue time is not directly observable through the OpenAI
HTTP API. It is included in TTFT. Rising TTFT and rising in-flight counts together are
evidence of accumulating service pressure; `dispatch_delay_seconds` instead diagnoses
client-side scheduling lag.

## Run

From the repository root, with vLLM already listening on port 18000:

```bash
pip install -e .

python experiments/E005-open-loop-saturation/run_e005.py --pilot \
  --base-url http://localhost:18000
```

If the pilot accounts for every arrival and the server remains healthy:

```bash
python experiments/E005-open-loop-saturation/run_e005.py \
  --base-url http://localhost:18000
```

Use `--output-dir PATH` to choose the artifact root.

## Artifacts

The top-level E005 directory contains:

- `e005_summary.json`: all load points and the highest observed SLO-passing rate;
- `plots/throughput_and_goodput_vs_offered_rate.png`;
- `plots/latency_vs_offered_rate.png`;
- `plots/pressure_and_slo_vs_offered_rate.png`.

Each rate has a normal benchmark run directory with `config.json`,
`environment.json`, `summary.json`, `requests.jsonl`, `workload.jsonl`, and GPU
`telemetry.jsonl`, plus `in_flight.jsonl` for the load timeline. This preserves the
exact workload plan and every offered arrival for later auditing.

## Interpretation

Look for a region where offered rate rises but completed throughput flattens, while
TTFT/E2E tails and peak in-flight requests rise. That joint behavior is saturation.
GPU utilization alone is not enough. Goodput may peak before raw throughput because
late completions still add throughput but no longer deliver service within the SLO.

`highest_observed_slo_passing_rate` means only “the highest tested point that passed.”
It is bounded by this discrete sweep, workload, SLO, hardware, software, and server
configuration. E005 deliberately does not turn it into an automatic production
capacity recommendation.

## Empirical Results (1× NVIDIA GeForce RTX 3090)

- Run ID: `E005_20260927_063839_565f514e`
- Source commit: `1d88839f13e3c8d33b5e991e569f08a460b0a319`
- Backend: vLLM 0.30.0, BF16
- Workload: 30 measured requests at each of five rates; 150/150 successful
- Errors, overload drops, and drain timeouts: zero

| Offered req/s | Achieved req/s | Goodput req/s | TTFT P95 | E2E P95 | Peak in flight | SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.5 | 0.503 | 0.453 | 338.93 ms | 6.560 s | 3 | Fail |
| 1.0 | 0.974 | 0.877 | 365.28 ms | 7.046 s | 4 | Fail |
| 2.0 | 1.715 | 1.543 | 359.70 ms | 7.831 s | 9 | Fail |
| 3.0 | 2.109 | 1.828 | 360.26 ms | 7.950 s | 11 | Fail |
| 4.0 | 2.359 | 1.888 | 382.22 ms | 8.082 s | 15 | Fail |

All requests completed, but achieved throughput increasingly diverged from
offered load above 1 request/s while in-flight pressure grew. Goodput nearly
plateaued between 3 and 4 requests/s. Together these observations place the
observed saturation region around and above 2 requests/s, without locating an
exact knee.

The complete SLO failed at every point only because E2E P95 exceeded its
5-second limit. TTFT P95 stayed below 1 second and error rate stayed at zero.
Therefore this sweep does not identify a passing sustainable rate under the
exact configured SLO; the rate may be below 0.5 requests/s, or the E2E objective
may be too strict for this workload's long-output tail.

See the [complete validation report](../../outputs/e005_open_loop_validation.md),
[canonical raw run](../../runs/E005_20260927_063839_565f514e/), and
[publication plots](../../outputs/plots/e005/).
