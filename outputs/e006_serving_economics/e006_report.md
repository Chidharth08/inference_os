# E006 Serving Economics — Report

## Scope and conclusion

E006 is an offline cost analysis of the canonical E005 open-loop run. It
does not execute inference or rent another GPU. All estimates use the
explicit price assumption of **USD 0.190 per GPU-hour**.

No E005 point passed the complete aggregate SLO because E2E P95 exceeded
five seconds at every offered rate. Therefore this report does not present
a cost at sustainable aggregate-SLO capacity. It reports measured point
costs and per-request SLO-compliant cost without changing the original SLO.

## Assumptions

- Source: `runs/E005_20260927_063839_565f514e/e005_summary.json`
- Currency: USD
- GPU price: USD 0.190/hour
- GPU count: 1
- Billing granularity: 1.000 second(s)
- Idle time inside each measured point: included
- Measurement scope: Per-point measured wall time, including arrival gaps and backlog drain; excluding warm-up, instance setup, model download/loading, time between load points, result packaging, and other session overhead.

## Point estimates

| Offered req/s | Completed | Individually compliant | Measured s | Estimated USD | USD/1k completed | USD/1k compliant | USD/1M output tokens | Aggregate SLO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 0.5 | 30 | 27 | 59.640 | 0.003167 | 0.1056 | 0.1173 | 1.1781 | Fail |
| 1.0 | 30 | 27 | 30.804 | 0.001636 | 0.0545 | 0.0606 | 0.6087 | Fail |
| 2.0 | 30 | 27 | 17.495 | 0.000950 | 0.0317 | 0.0352 | 0.3534 | Fail |
| 3.0 | 30 | 26 | 14.225 | 0.000792 | 0.0264 | 0.0304 | 0.2945 | Fail |
| 4.0 | 30 | 24 | 12.715 | 0.000686 | 0.0229 | 0.0286 | 0.2552 | Fail |

Higher offered load reduced normalized cost per completion through
batching and higher utilization, but individual SLO compliance fell.
A cheaper completion is not automatically a better service outcome.

## Benchmark cost versus session spend

The five measured windows total 134.879 seconds. Applying per-point billing granularity produces 137.000 billable seconds and an estimated benchmark-normalized cost of USD 0.007231.

The observed Vast.ai credit decrease was approximately USD 0.50. The difference of USD 0.492769 is not assigned to serving requests because the available account observation does not separate model download, container transfer, storage, bandwidth, setup, warm-up, idle time, and other session activity.

## Metric definitions

```text
measured GPU-seconds = measured wall seconds × GPU count
estimated point cost = billable GPU-seconds × GPU price per second
cost per completed request = estimated point cost / completions
cost per compliant request = estimated point cost / individually compliant completions
cost per million output tokens = estimated point cost / output tokens × 1,000,000
```

Failed requests would remain in billed time but not in the completed or
compliant denominator. A zero denominator is reported as null/`N/A`, not
zero cost.

## Limitations

1. The GPU-hour price is a user-supplied estimate and varies by host and time.
2. Storage and bandwidth prices were not captured, so session overhead cannot be decomposed.
3. Point estimates exclude warm-up and all time outside measured E005 windows.
4. The actual session cost is an approximate credit-balance change, not an invoice line-item total.
5. No tested point passed the aggregate SLO; E006 makes no sustainable-capacity claim.
6. Results apply only to the recorded model, workload, hardware, backend, and SLO.
7. E005 measured one workload shape, so E006 compares load-point economics, not cost differences between workload profiles.
