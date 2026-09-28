# inference_os — Study Guide for E005 and E006

## Open-Loop Saturation, SLO Goodput, and Serving Economics

This guide connects the systems concepts behind E005 and E006 to the actual
configuration, code, artifacts, and measurements in this repository. It assumes
you understand the earlier progression: E001 separated prefill from decode,
E002 studied closed-loop concurrency, E003 introduced application-shaped
workloads, and E004 showed why workload variance matters.

By the end, you should be able to:

- distinguish closed-loop concurrency from open-loop offered load;
- explain scheduled arrival, dispatch delay, in-flight pressure, and drain time;
- identify saturation without relying on GPU utilization alone;
- distinguish throughput, goodput, individual compliance, and aggregate SLO pass;
- explain why 90% compliant requests can coexist with a failed P95 objective;
- audit every scheduled E005 request from configuration to persisted evidence;
- calculate GPU-seconds and several normalized serving-cost metrics;
- separate measured-window cost from the total cloud bill;
- explain why the cheapest load point is not necessarily sustainable;
- state what these experiments prove and what they do not prove.

---

## Table of Contents

1. [Where E005 and E006 Fit](#1-where-e005-and-e006-fit)
2. [The Load-Model Shift](#2-the-load-model-shift)
3. [The Open-Loop Request Timeline](#3-the-open-loop-request-timeline)
4. [Metrics That Must Stay Separate](#4-metrics-that-must-stay-separate)
5. [What Saturation Means](#5-what-saturation-means)
6. [Understanding the E005 SLO](#6-understanding-the-e005-slo)
7. [E005 Experiment Design](#7-e005-experiment-design)
8. [Reading the E005 Results](#8-reading-the-e005-results)
9. [Why Every Load Point Failed the Aggregate SLO](#9-why-every-load-point-failed-the-aggregate-slo)
10. [E005 Code and Artifact Walkthrough](#10-e005-code-and-artifact-walkthrough)
11. [How to Audit E005](#11-how-to-audit-e005)
12. [The E006 Cost Model](#12-the-e006-cost-model)
13. [Measurement Boundaries and Billing](#13-measurement-boundaries-and-billing)
14. [Worked E006 Calculations](#14-worked-e006-calculations)
15. [Reading the E006 Results](#15-reading-the-e006-results)
16. [Benchmark Cost Versus Actual Session Spend](#16-benchmark-cost-versus-actual-session-spend)
17. [E006 Code and Artifact Walkthrough](#17-e006-code-and-artifact-walkthrough)
18. [Common Interpretation Mistakes](#18-common-interpretation-mistakes)
19. [Applied Decision Framework](#19-applied-decision-framework)
20. [Interview-Ready Explanations](#20-interview-ready-explanations)
21. [Exercises](#21-exercises)
22. [Exercise Solutions](#22-exercise-solutions)
23. [Glossary](#23-glossary)
24. [Final Mental Model](#24-final-mental-model)

---

## 1. Where E005 and E006 Fit

| Experiment | Independent variable | Main question |
| :--- | :--- | :--- |
| E001 | Input/output length | How do prefill and decode scale? |
| E002 | Closed-loop concurrency | How does batching trade latency for throughput? |
| E003 | Application-shaped workload | How do realistic request shapes behave? |
| E004 | Length variance at a controlled mean | Why is the average request insufficient? |
| E005 | Open-loop offered request rate | Where does service saturate, and what load meets the SLO? |
| E006 | Price and billing assumptions | What do E005's measurements cost? |

E005 changes the question from “what happens with N continuously active
clients?” to “what happens when users arrive at R requests per second?” E006
then places an economic model over those preserved measurements. It does not
rerun inference and does not manufacture new performance data.

```text
request shape + arrival rate
            ↓
client scheduling and server service
            ↓
latency, throughput, goodput, pressure
            ↓
SLO-qualified operating region
            ↓
cost inside that operating region
```

The arrows matter. Cost is a downstream interpretation of measured service
behavior. A cheap result is useful only if it meets the required quality.

---

## 2. The Load-Model Shift

### Closed loop

In a closed-loop test, each worker waits for its current request to finish before
sending its next request:

```text
send → wait → complete → send again
```

Concurrency is controlled, but the arrival rate is an outcome. If the server
slows down, workers also send less frequently. The client therefore
self-throttles. This is valuable for studying concurrency, as in E002–E004, but
it can conceal overload behavior.

### Open loop

In E005, request arrivals are scheduled from a clock independently of previous
completions. For constant offered rate `r`, request `i` is scheduled at:

```text
t_i = i / r
```

At 2 requests/s the intended schedule is 0.0, 0.5, 1.0, 1.5 seconds, and so on.
If one request is still executing, later requests are still due. That allows
work to accumulate and exposes service pressure.

Open loop does not mean unlimited load. E005 has `max_in_flight` and a drain
deadline as safety boundaries. Importantly, `max_in_flight: 64` is not a worker
count or desired concurrency. It is the point at which the client records a
`client_overload` drop instead of allowing unbounded growth.

---

## 3. The Open-Loop Request Timeline

For each request, keep these moments distinct:

```text
scheduled arrival
      ↓  client dispatch delay
actual dispatch / request start
      ↓  server admission, queueing, prefill
first token
      ↓  decode and network streaming
completion
```

The corresponding measurements are:

```text
dispatch delay = dispatch time - scheduled arrival time
TTFT           = first-token time - request start time
E2E latency    = completion time - request start time
```

`dispatch_delay_seconds` diagnoses whether the load generator itself fell
behind its schedule. It is not vLLM's internal queue duration.

The OpenAI-compatible HTTP response does not directly expose every server-side
queue transition. Server queueing is therefore embedded in TTFT. Rising TTFT
together with rising in-flight pressure is evidence of congestion, but neither
field alone is a direct server-queue trace.

“In flight” also is not identical to “queued.” It includes every dispatched
request that has not finished: some may be queued, some prefilling, and some
decoding. It is best understood as a pressure measure.

After the final scheduled arrival, unfinished requests are allowed to drain.
The measured window ends only when they finish or the drain deadline cancels
them. This prevents throughput from being artificially inflated by counting
arrivals while ignoring completion work.

---

## 4. Metrics That Must Stay Separate

### Offered load

The external demand requested from the load generator:

```text
offered rate = scheduled arrivals / intended arrival-window seconds
```

With 30 requests at 0.5 req/s, the intended arrival window is 60 seconds. At
4 req/s, it is only 7.5 seconds.

### Achieved throughput

Successful completions divided by the full measured wall-clock duration:

```text
throughput = successful completions / measured duration
```

The measured duration includes the arrival window and backlog drain. It can be
longer than `requests / offered_rate`, so achieved throughput can be lower than
offered load even when every request eventually succeeds.

### Token throughput

Requests have different prompt and output lengths. Request throughput tells you
how many jobs complete; token throughput tells you how much model work completes.
Never assume one substitutes for the other.

### Individual SLO compliance

An E005 request is individually compliant only if it:

1. succeeds;
2. has a valid first-token measurement;
3. has TTFT at or below 1 second; and
4. has E2E latency at or below 5 seconds.

### Goodput

```text
goodput = individually compliant requests / measured duration
```

Throughput counts successful but slow completions. Goodput excludes them. It is
therefore a quality-qualified delivery rate.

### Aggregate load-point SLO

A complete rate point passes only when all three are true:

```text
TTFT P95       ≤ 1 second
E2E P95        ≤ 5 seconds
error/drop rate ≤ 1%
```

Individual compliance supplies an additive count for goodput. Aggregate
percentiles decide whether the entire tested operating point passes. They answer
related but different questions.

---

## 5. What Saturation Means

Saturation is the region in which adding offered work no longer produces
proportional useful output and instead creates waiting, longer tails, or drops.
It is not merely “GPU utilization reached 100%.”

Strong evidence combines several signals:

- offered rate rises faster than achieved throughput;
- in-flight requests accumulate;
- TTFT or E2E tail latency rises;
- goodput flattens or falls before raw throughput;
- overload drops or drain timeouts may eventually appear.

The canonical run shows this transition without any request errors. At the high
end, the server still completed all requests, but not at the rate or latency the
client requested. Zero errors does not imply absence of saturation.

The “knee” is the transition between the efficient region and sharply declining
marginal returns. E005 samples only five rates with one run and 30 requests per
rate. It supports an observed saturation region around and above 2 req/s, not a
precise universal knee.

---

## 6. Understanding the E005 SLO

The SLO thresholds are experimental policy, not laws of inference serving:

- TTFT P95 ≤ 1 second;
- E2E P95 ≤ 5 seconds;
- total error/drop rate ≤ 1%.

The thresholds should come from product requirements in a real system. A chat
product may care strongly about TTFT, while an offline summarization pipeline may
accept slower responses in exchange for lower unit cost.

### Why 90% individual compliance can fail P95

At 30 requests, 90% compliance means 27 passed and 3 missed at least one
per-request threshold. P95 asks for the value below which roughly 95% of samples
fall. With three slow requests, the tail can exceed the threshold even though a
large majority passed.

This is not a contradiction:

```text
individual compliance: How many requests met every per-request limit?
aggregate SLO:         Did the load point's tail and error metrics meet policy?
```

Small samples also make percentiles coarse. One or two observations can change
P95 materially, which is why production capacity studies normally repeat points
and use longer windows.

---

## 7. E005 Experiment Design

The canonical configuration is in
[`configs/e005_open_loop.yaml`](../configs/e005_open_loop.yaml).

| Control | Canonical value | Why it matters |
| :--- | ---: | :--- |
| Model | Qwen2.5-7B-Instruct | Keeps the model fixed across points |
| Offered rates | 0.5, 1, 2, 3, 4 req/s | Sweeps low load into pressure |
| Measured requests | 30 per rate | Same workload plan size at every point |
| Warm-ups | 5 per rate | Removes initial cold behavior from measurement |
| Arrival process | Constant/deterministic | Isolates rate before adding burst randomness |
| `max_in_flight` | 64 | Bounds client-side overload risk |
| Temperature | 0 | Reduces generation variability |
| Prefix caching | Disabled | Avoids cache-dependent speedups |
| Chunked prefill | Disabled | Holds server behavior controlled |

The workload reuses the synthetic chat-shaped distribution from E003. Holding
the request plan constant while changing only arrival timing makes cross-rate
comparison more defensible.

There are 150 measured requests in total. Intended arrival windows are:

| Offered rate | Requests | Intended arrival window |
| ---: | ---: | ---: |
| 0.5 req/s | 30 | 60 s |
| 1.0 req/s | 30 | 30 s |
| 2.0 req/s | 30 | 15 s |
| 3.0 req/s | 30 | 10 s |
| 4.0 req/s | 30 | 7.5 s |

The shorter high-rate windows are a limitation. They make those points more
sensitive to transient conditions and drain time.

---

## 8. Reading the E005 Results

The canonical results are documented in
[`outputs/e005_open_loop_validation.md`](../outputs/e005_open_loop_validation.md)
and preserved under
[`runs/E005_20260927_063839_565f514e/`](../runs/E005_20260927_063839_565f514e/).

| Offered req/s | Achieved req/s | Goodput req/s | Individual compliance | TTFT P95 | E2E P95 | Peak in-flight | Aggregate SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.5 | 0.503 | 0.453 | 90.0% | 0.339 s | 6.560 s | 3 | Fail |
| 1.0 | 0.974 | 0.877 | 90.0% | 0.365 s | 7.046 s | 4 | Fail |
| 2.0 | 1.715 | 1.543 | 90.0% | 0.360 s | 7.831 s | 9 | Fail |
| 3.0 | 2.109 | 1.828 | 86.7% | 0.360 s | 7.950 s | 11 | Fail |
| 4.0 | 2.359 | 1.888 | 80.0% | 0.382 s | 8.082 s | 15 | Fail |

All 150 requests completed successfully, with zero recorded errors, overload
drops, or drain timeouts. The important result is therefore performance
degradation, not reliability failure.

### Offered-to-achieved tracking

Approximate achieved/offered ratios are:

| Offered rate | Achieved/offered |
| ---: | ---: |
| 0.5 | 100.6% |
| 1.0 | 97.4% |
| 2.0 | 85.7% |
| 3.0 | 70.3% |
| 4.0 | 59.0% |

The slight value above 100% at 0.5 req/s is a timing-boundary effect, not free
capacity. At higher rates, achieved throughput increasingly fails to track demand.

### Diminishing returns

Moving from 3 to 4 offered req/s increases offered load by 33.3%, but achieved
throughput rises only about 11.9%, and goodput rises only about 3.3%. Meanwhile,
peak in-flight reaches 15 and individual compliance falls to 80%.

This is the central saturation result: the extra demand mostly creates pressure
and latency rather than proportional useful service.

### Tail behavior

TTFT P95 remains comfortably below one second. E2E P95 rises from 6.560 to
8.082 seconds, about a 23.2% increase. For this workload and SLO, decode-length
completion time is the binding quality problem, not first-token responsiveness.

---

## 9. Why Every Load Point Failed the Aggregate SLO

Each point passed the TTFT threshold and had a 0% error/drop rate. Each failed
because E2E P95 exceeded five seconds—even at 0.5 req/s.

That leads to an important diagnosis: this sweep did not bracket a passing
capacity point for the complete SLO. The failure at low load suggests that the
five-second E2E target is incompatible with part of this workload/model/output
configuration even before saturation becomes severe.

Therefore, do not report “the sustainable rate is zero” as if hardware capacity
alone were the answer. Better conclusions are:

- no tested point passed the configured aggregate SLO;
- the SLO, workload, output limits, or serving configuration should be revisited;
- a lower-rate sweep alone may not help if long generations exceed five seconds
  even without queueing;
- capacity under a different, product-justified SLO requires a new analysis.

Never relax an SLO solely to make the benchmark pass. First decide whether the
threshold truly reflects the user experience the product needs.

---

## 10. E005 Code and Artifact Walkthrough

The implementation path is:

```text
e005_open_loop.yaml
        ↓
configuration validation
        ↓
deterministic workload plan
        ↓
clock-scheduled open-loop runner
        ↓
request + in-flight evidence
        ↓
throughput, goodput, percentile, SLO summaries
        ↓
E005 aggregate JSON and plots
```

Key files:

- [`src/inference_os/config.py`](../src/inference_os/config.py) defines and
  validates arrival rates, limits, drain policy, and SLO thresholds.
- [`src/inference_os/runner/load.py`](../src/inference_os/runner/load.py) holds
  open-loop load-domain structures.
- [`src/inference_os/runner/load_engine.py`](../src/inference_os/runner/load_engine.py)
  schedules arrivals independently of completions and records pressure events.
- [`src/inference_os/metrics/request.py`](../src/inference_os/metrics/request.py)
  derives request-level measurements.
- [`src/inference_os/metrics/slo.py`](../src/inference_os/metrics/slo.py)
  evaluates individual compliance and aggregate SLO conditions.
- [`src/inference_os/results/persistence.py`](../src/inference_os/results/persistence.py)
  persists normal run artifacts and in-flight evidence.
- [`experiments/E005-open-loop-saturation/run_e005.py`](../experiments/E005-open-loop-saturation/run_e005.py)
  coordinates the multi-rate experiment and plots.

Each rate directory retains configuration, environment metadata, workload plan,
request records, summary, telemetry, and `in_flight.jsonl`. The top-level E005
summary connects those rate points into one experiment.

---

## 11. How to Audit E005

An audit should answer five questions.

### 1. Was the intended experiment run?

Inspect the saved config, model, endpoint, rate list, request count, seed, SLO,
and server controls. Do not trust a directory name as proof.

### 2. Was every offered arrival accounted for?

At every rate:

```text
scheduled arrivals = successes + request errors + overload drops + drain timeouts
```

For the canonical run, all 30 arrivals at each point became successful requests.

### 3. Did the client keep up?

Inspect dispatch-delay distributions. Large delays can make a nominal 4 req/s
test an accidental lower-rate test.

### 4. Did work accumulate and drain?

Use the in-flight timeline, peak in-flight count, arrival window, and measured
duration together. A growing backlog followed by drain is expected evidence of
pressure, not necessarily a client bug.

### 5. Can every aggregate claim be rebuilt?

Recalculate success count, percentile metrics, compliant count, throughput,
goodput, and error/drop rate from request-level artifacts. Generated summaries
are convenient indexes; raw records remain the evidence.

---

## 12. The E006 Cost Model

E006 is a deterministic offline transformation of E005. Its assumptions live in
[`configs/e006_economics.yaml`](../configs/e006_economics.yaml).

Canonical assumptions:

- GPU price: USD 0.190 per GPU-hour;
- GPU count: 1;
- billing granularity: 1 second;
- approximate observed session credit decrease: USD 0.50.

First convert hourly price:

```text
price per GPU-second = 0.190 / 3600
                     = USD 0.0000527778
```

For measured duration `d`, billing granularity `g`, and GPU count `n`:

```text
billable duration = ceil(d / g) × g
measured GPU-seconds = d × n
billed GPU-seconds   = billable duration × n
estimated point cost = billed GPU-seconds × price per GPU-second
```

Normalized metrics include:

```text
USD / 1,000 completed requests
USD / 1,000 individually compliant requests
USD / 1,000,000 input tokens
USD / 1,000,000 output tokens
USD / 1,000,000 total tokens
GPU-seconds / completed request
GPU-seconds / compliant request
```

Failed or slow requests still consume time and therefore cost. They simply do
not enter successful or compliant denominators. A zero denominator produces
`null`, never a misleading zero-cost result.

---

## 13. Measurement Boundaries and Billing

The numerator and denominator must cover the same boundary. E006 prices each
E005 measured point from schedule start through backlog drain. That includes:

- scheduled gaps between arrivals;
- inference execution;
- queueing embedded in service time;
- completion of remaining work after the final arrival.

It excludes costs that E005 did not attribute per point:

- instance provisioning and shutdown time;
- container or dependency setup;
- model download and loading;
- warm-up requests;
- idle time between rate points;
- result packaging and download;
- persistent storage and bandwidth charges.

E006 therefore reports benchmark-normalized compute estimates—not the invoice
for operating the entire rented instance.

Billing rounding is applied to each point before costs are summed. With a
one-second granularity, five individually rounded points can cost slightly more
than rounding their combined duration once. The choice models point-level
accounting and is explicit in the output.

---

## 14. Worked E006 Calculations

### Example A: 0.5 req/s

The measured window was about 59.64 seconds. With one-second billing:

```text
billable seconds = ceil(59.64) = 60
point cost = 60 × (0.190 / 3600)
           = USD 0.00316667
```

All 30 requests completed and 27 were individually compliant:

```text
USD / 1,000 completed = (0.00316667 / 30) × 1000
                      ≈ USD 0.1056

USD / 1,000 compliant = (0.00316667 / 27) × 1000
                      ≈ USD 0.1173
```

The compliant metric is higher because the same point cost is divided by fewer
useful outcomes.

### Example B: 4 req/s

The measured window was about 12.715 seconds:

```text
billable seconds = ceil(12.715) = 13
point cost = 13 × (0.190 / 3600)
           = USD 0.00068611
```

All 30 requests completed, while 24 were individually compliant:

```text
USD / 1,000 completed ≈ USD 0.0229
USD / 1,000 compliant ≈ USD 0.0286
```

This point is cheaper per completion because a short, highly utilized window
amortizes GPU time over more work. It is not automatically better: compliance
fell to 80%, and the aggregate SLO failed.

---

## 15. Reading the E006 Results

The generated evidence is under
[`outputs/e006_serving_economics/`](../outputs/e006_serving_economics/).

| Offered req/s | Point cost | USD/1k completed | USD/1k compliant | USD/1M output tokens | Aggregate SLO |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 0.5 | 0.003167 | 0.1056 | 0.1173 | 1.1781 | Fail |
| 1.0 | 0.001636 | 0.0545 | 0.0606 | 0.6087 | Fail |
| 2.0 | 0.000950 | 0.0317 | 0.0352 | 0.3534 | Fail |
| 3.0 | 0.000792 | 0.0264 | 0.0304 | 0.2945 | Fail |
| 4.0 | 0.000686 | 0.0229 | 0.0286 | 0.2552 | Fail |

Unit cost falls as offered load rises because batching and utilization improve.
At the same time, individual compliance falls from 90% to 80%, and every point
fails the aggregate E2E SLO.

This gives two different optima:

```text
cheapest observed completion: 4 req/s
cheapest aggregate-SLO-passing completion: unavailable
```

The second is the decision-relevant production metric, but it cannot be reported
because there is no passing point. E006 correctly preserves that absence rather
than labeling the cheapest failing point “sustainable.”

The experiment also compares rates for one workload. It does not prove that this
model or provider is cheaper than another. Cross-model economics would require
equivalent workload, output quality, hardware accounting, and SLO policy.

---

## 16. Benchmark Cost Versus Actual Session Spend

The five E005 measured windows total 134.879 seconds. Rounding every point to a
whole billed second produces 137 billed GPU-seconds:

```text
normalized benchmark cost = 137 × (0.190 / 3600)
                          = USD 0.00723056
```

The approximate Vast.ai credit decrease was USD 0.50:

```text
unattributed spend = 0.50 - 0.00723056
                   = USD 0.49276944

session / measured-window ratio ≈ 69.15×
```

This does not prove that storage alone cost USD 0.4928. The difference may
contain rental time outside measured windows, model download/loading, setup,
warm-ups, idle time, packaging, storage, bandwidth, provider rounding, and price
variation. Without an itemized bill and timestamped session phases, allocation
would be invented precision.

Do not divide USD 0.50 by 150 requests and call it benchmark cost per request.
That would mix experimental inference with unrelated session overhead. Both
numbers are useful, but they answer different questions:

- normalized cost: efficiency of the measured serving windows;
- total session spend: what the complete experimental workflow cost the user.

For a future run, record instance start/stop timestamps and itemized provider
charges, then separate setup, model load, warm-up, measurement, idle, storage,
and transfer phases.

---

## 17. E006 Code and Artifact Walkthrough

The offline path is:

```text
E005 summary + E006 price assumptions
                    ↓
validated economics calculations
                    ↓
per-rate normalized costs
                    ↓
session reconciliation
                    ↓
JSON + Markdown report + plots
```

Key files:

- [`src/inference_os/metrics/economics.py`](../src/inference_os/metrics/economics.py)
  contains reusable billing and unit-cost calculations.
- [`experiments/E006-serving-economics/run_e006.py`](../experiments/E006-serving-economics/run_e006.py)
  loads E005, applies assumptions, and writes the complete analysis.
- [`src/inference_os/reports/plots.py`](../src/inference_os/reports/plots.py)
  renders cost-per-request, cost-per-token, and cost-quality views.
- [`outputs/e006_serving_economics/e006_summary.json`](../outputs/e006_serving_economics/e006_summary.json)
  is the machine-readable derived result.
- [`outputs/e006_serving_economics/e006_report.md`](../outputs/e006_serving_economics/e006_report.md)
  is the human-readable generated report.

Because E006 is offline and deterministic, changing only the GPU-hour price does
not require another GPU run. It is a scenario analysis over the same measurements.
Changing model, hardware, server configuration, workload, or performance behavior
does require new empirical data.

---

## 18. Common Interpretation Mistakes

### “Open loop means unlimited concurrency”

No. Arrivals are independent of completions, but safety limits still bound
in-flight work and drain duration.

### “In-flight count is server queue length”

No. It combines queued and actively processed requests visible to the client.

### “Dispatch delay is server queueing”

No. It measures load-generator scheduling lag. Server waiting is included in TTFT.

### “Zero errors means the server handled 4 req/s”

All requests completed, but achieved throughput was 2.359 req/s, goodput was
1.888 req/s, and the backlog drained after arrivals. Completion is not the same
as sustaining the offered rate.

### “90% compliant means the P95 SLO passed”

No. Individual compliance is a count. Aggregate P95 is a tail statistic.

### “The cheapest point is the best deployment point”

No. The 4 req/s point is cheapest per completion but failed the aggregate SLO.

### “The benchmark should explain the full USD 0.50 charge”

No. E006 prices only preserved measured windows. The full session has a wider
boundary and unitemized components.

### “E006 proves the current Vast.ai price”

No. USD 0.190/hour is an explicit scenario assumption captured at the time. It
is not a live-price claim.

### “Thirty requests precisely locate the saturation knee”

No. They establish a useful first measurement. Precise capacity planning needs
more samples, repeated trials, narrower rate spacing, and often bursty arrivals.

---

## 19. Applied Decision Framework

When choosing a serving operating point, use this order:

1. Define a product-derived workload and SLO.
2. Verify client scheduling and request accounting.
3. Find the rate region where throughput begins to flatten and pressure rises.
4. Exclude points that fail the aggregate SLO.
5. Compare goodput and cost only among passing points.
6. Add safety headroom for bursts, failures, and workload drift.
7. Validate with longer, repeated, and burst-aware runs.
8. Reconcile benchmark estimates with itemized cloud spend.

For this canonical result, step 4 removes every point. The next scientifically
sound action is not to select 4 req/s because it is cheapest. It is to investigate
the low-load E2E baseline and the SLO/workload relationship, then rerun a sweep
that actually brackets a passing-to-failing transition.

A stronger follow-up could include:

- more requests and multiple seeds per point;
- narrower rates around 1.5–3 req/s;
- a low-load latency baseline;
- production-like Poisson or bursty arrivals;
- separate SLOs by workload class or output-length bucket;
- direct server-side queue and scheduler telemetry;
- itemized provider cost and full session phase timing.

---

## 20. Interview-Ready Explanations

### What is the difference between closed-loop and open-loop load?

In closed loop, a fixed set of workers waits for completions before sending more
requests, so a slow server automatically reduces arrivals. In open loop, arrivals
follow an external schedule independent of completions, so overload appears as
backlog, tail latency, reduced rate tracking, or drops.

### How did you identify saturation?

I combined signals rather than using GPU utilization alone. As offered load rose,
achieved throughput tracked it less closely, peak in-flight work rose from 3 to
15, E2E P95 increased, and goodput nearly flattened from 3 to 4 offered req/s.
That supports saturation around and above 2 req/s, while the coarse sweep prevents
claiming an exact knee.

### Why use goodput?

Throughput rewards every successful completion, including responses too slow for
the product. Goodput counts only completions that meet per-request latency and
success conditions, so it measures useful service delivered per second.

### Why did every rate fail if many requests were compliant?

The aggregate policy required E2E P95 at or below five seconds. E2E P95 was above
six seconds even at the lowest load. Individual compliance and aggregate tail
compliance are different views, so 80–90% individual compliance does not imply a
P95 pass.

### Could E006 be done without renting another GPU?

Yes. E006 only applies explicit price and billing assumptions to preserved E005
durations, completions, compliance counts, and token totals. A new GPU run is
needed only if the performance inputs change.

### Why was measured cost much lower than the credit decrease?

The normalized estimate prices only 137 billed GPU-seconds inside measured
windows. The session charge covers a much larger lifecycle, potentially including
setup, download, loading, idle time, storage, bandwidth, and provider accounting.
Without itemization, the difference should remain unattributed.

---

## 21. Exercises

1. At 2 req/s with 30 scheduled requests, what is the intended arrival window?

2. If 30 requests complete in a 17.5-second measured window, what is achieved
   throughput?

3. If 27 of those requests meet individual SLO conditions, what is goodput?

4. Explain why the answers to exercises 2 and 3 use measured duration rather
   than the intended arrival window.

5. At 4 offered req/s, calculate the approximate percentage by which offered
   load exceeds achieved throughput.

6. Why is `peak_in_flight = 15` evidence that `max_in_flight = 64` did not bind?

7. With 30 requests and 90% compliance, how many were non-compliant?

8. At USD 0.190/GPU-hour, what is the price per GPU-second?

9. A point lasts 12.2 seconds with one-second billing granularity. How many
   seconds are billed?

10. If that point has one GPU, 30 completions, and costs USD 0.000686, calculate
    approximate USD per 1,000 completions.

11. Why can USD per 1,000 compliant requests rise relative to USD per 1,000
    completed requests?

12. Name four session costs or phases intentionally excluded from the E006
    measured-window estimate.

13. If a new GPU price is USD 0.25/hour but all performance data remains valid,
    must inference be rerun? Explain.

14. If prefix caching is enabled, can the old E005 timings still be treated as
    measured performance for that deployment? Explain.

15. Design a follow-up that estimates an exact saturation knee more reliably.

---

## 22. Exercise Solutions

1. `30 / 2 = 15 seconds`.

2. `30 / 17.5 = 1.714 req/s`.

3. `27 / 17.5 = 1.543 compliant req/s`.

4. Work can remain after the last scheduled arrival. Using only the arrival
   window would ignore drain work and overstate delivery rate.

5. `(4 - 2.359) / 4 × 100 ≈ 41.0%`. Equivalently, achieved throughput is about
   59.0% of offered load.

6. Only 15 requests were simultaneously in flight, far below the safety cap of
   64, so no arrival should have been rejected because of that cap.

7. `30 × 0.10 = 3 requests`.

8. `0.190 / 3600 = USD 0.0000527778 per GPU-second`.

9. `ceil(12.2 / 1) × 1 = 13 billed seconds`.

10. `(0.000686 / 30) × 1000 ≈ USD 0.0229`.

11. The same total cost is divided by fewer compliant outcomes. Slow successful
    requests remain completions but are excluded from the compliant denominator.

12. Any four of provisioning, dependency setup, model download, model loading,
    warm-up, inter-point idle time, packaging, storage, bandwidth, and shutdown.

13. No. Recompute the deterministic cost model with the new price assumption.
    Clearly label it as a price scenario over the original measurements.

14. No. Prefix caching changes performance behavior. The new configuration needs
    empirical validation even if the offline pricing formulas remain the same.

15. Run longer windows with more requests, repeat each rate across seeds, add
    closely spaced rates around the observed transition, randomize point order,
    and report confidence intervals. Confirm client timing and server health.

---

## 23. Glossary

**Aggregate SLO** — A policy applied to load-point statistics such as P95 latency
and error rate.

**Arrival window** — Time over which requests are scheduled, excluding later drain.

**Billed GPU-seconds** — GPU count multiplied by duration after billing rounding.

**Closed loop** — Load model where new work depends on previous completions.

**Dispatch delay** — Difference between intended arrival and actual client dispatch.

**Drain** — Completion or cancellation of work remaining after arrivals stop.

**E2E latency** — Request start to complete response.

**Goodput** — Individually SLO-compliant completions per measured second.

**In flight** — Dispatched requests not yet completed; queued and active combined.

**Individual compliance** — Whether one request met success, TTFT, and E2E limits.

**Knee** — Region where additional load produces sharply diminishing useful output.

**Measured window** — Schedule start through final completion or drain deadline.

**Open loop** — Load model where arrivals are scheduled independently of completion.

**Offered rate** — Requested external arrival rate.

**Saturation** — Region where demand outgrows proportional service capacity.

**SLO** — Service-level objective defining acceptable performance and reliability.

**Throughput** — Successful completions per measured second.

**TTFT** — Time from request start to first generated token.

**Unit cost** — Cost normalized by a denominator such as requests or tokens.

---

## 24. Final Mental Model

Keep this chain in your head:

```text
Offered load is what users ask for.
Achieved throughput is what the server finishes.
In-flight growth shows accumulating pressure.
Latency tails show what waiting does to experience.
Goodput is the amount of useful service delivered on time.
The aggregate SLO defines which operating points are acceptable.
Economics should compare cost only inside that acceptable region.
The cloud bill includes a wider lifecycle than benchmark windows.
```

E005 demonstrates why capacity cannot be summarized by “all requests succeeded”
or “the GPU was busy.” E006 demonstrates why economics cannot be summarized by
the cheapest raw completion or by dividing an unitemized cloud charge by request
count. Together, they establish the production-serving discipline:

> Measure externally offered demand, observe pressure and tail quality, qualify
> capacity with an explicit SLO, and only then optimize cost—with every boundary
> and assumption preserved.
