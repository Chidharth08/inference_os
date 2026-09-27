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
