# E006 — Serving Economics

## Research question

What benchmark-normalized serving costs follow from the canonical E005
measurements under an explicit GPU-hour price assumption?

E006 is an offline analysis. It does not start vLLM, send requests, or require a
GPU. It converts recorded E005 wall-clock duration, completions, individually
SLO-compliant completions, and token counts into narrowly scoped cost estimates.

## Price and measurement assumptions

The canonical configuration uses the user-observed Vast.ai estimate of
`USD 0.190/GPU-hour` for one RTX 3090. This is configuration data, not a claim
about a universal or current market price.

Point cost includes the complete E005 measured wall-clock window: scheduled
arrival gaps and backlog drain are included. Warm-up, instance provisioning,
container/model transfer, model loading, time between points, result packaging,
storage, and bandwidth are excluded because E005 did not attribute those times
or charges to individual points.

The approximate `USD 0.50` credit decrease is retained separately as session
context. It is not divided by requests because its compute, storage, and
bandwidth components are unavailable.

## Run

From the repository root:

```bash
python experiments/E006-serving-economics/run_e006.py
```

No server or GPU is required. Override assumptions when needed:

```bash
python experiments/E006-serving-economics/run_e006.py \
  --gpu-price-per-hour 0.250 \
  --output-dir outputs/e006_alternative_price
```

## Metrics

E006 reports:

- measured and billed GPU-seconds;
- GPU-seconds per completed request;
- GPU-seconds per individually SLO-compliant request;
- estimated cost per completed request and per 1,000 completions;
- estimated cost per compliant request and per 1,000 compliant completions;
- estimated cost per million input, output, and total tokens;
- benchmark-normalized total cost;
- approximate session-spend reconciliation.

Failed requests consume measured time and cost but do not enter successful or
SLO-compliant denominators. A zero denominator produces `null`, never a
misleading zero cost.

## SLO limitation inherited from E005

No E005 point passed the complete aggregate SLO because E2E P95 exceeded five
seconds at every rate. E006 therefore does not report a “cost at sustainable
SLO capacity.” It can still report cost per individually compliant completion:
that is a different denominator and does not retroactively make a load point
pass its aggregate percentile objective.

## Artifacts

The configured output directory contains:

```text
outputs/e006_serving_economics/
├── config.json
├── e006_summary.json
├── e006_report.md
└── plots/
    ├── cost_per_1000_requests_vs_offered_rate.png
    ├── cost_per_million_tokens_vs_offered_rate.png
    └── cost_quality_tradeoff.png
```

These are deterministic offline derivatives of the preserved E005 summary and
the explicit E006 assumptions.

## Canonical offline results

Using `USD 0.190/GPU-hour`, one RTX 3090, and one-second billing increments:

| Offered req/s | Estimated cost | USD/1k completed | USD/1k compliant | USD/1M output tokens | Aggregate SLO |
| ---: | ---: | ---: | ---: | ---: | :--- |
| 0.5 | 0.003167 | 0.1056 | 0.1173 | 1.1781 | Fail |
| 1.0 | 0.001636 | 0.0545 | 0.0606 | 0.6087 | Fail |
| 2.0 | 0.000950 | 0.0317 | 0.0352 | 0.3534 | Fail |
| 3.0 | 0.000792 | 0.0264 | 0.0304 | 0.2945 | Fail |
| 4.0 | 0.000686 | 0.0229 | 0.0286 | 0.2552 | Fail |

The five measured windows total 134.879 seconds. Rounding each point to the
configured one-second billing increment produces 137 billable GPU-seconds and a
benchmark-normalized estimate of `USD 0.007231`.

The approximate session credit decrease was `USD 0.50`, or about 69.15 times
the measured-window estimate. The remaining `USD 0.492769` is intentionally not
allocated to requests because storage, bandwidth, setup, model transfer/loading,
warm-up, inter-point idle time, and other session components were not itemized.

Higher offered load lowered normalized unit cost through batching, but individual
SLO compliance fell from 90% to 80%. The cheapest completion was therefore not
an aggregate-SLO-compliant serving point.

See the [generated report](../../outputs/e006_serving_economics/e006_report.md),
[machine-readable summary](../../outputs/e006_serving_economics/e006_summary.json),
and [plots](../../outputs/e006_serving_economics/plots/).
