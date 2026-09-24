# E004 — Fixed versus Variable Workloads

## Primary Question

> Why can workloads with the same average token counts have different tail
> latency and throughput?

E004 compares a fixed workload with a heterogeneous workload while controlling
the configured and canonical realized mean input and maximum-output lengths.
The experiment isolates token-length variance; it does not introduce open-loop
arrivals or production application traffic.

## Hypotheses

1. The variable workload will have worse P95/P99 TTFT and E2E latency despite
   matching the fixed workload's mean token shape.
2. Median latency may remain closer than tail latency because most variable
   requests retain the mean-sized shape.
3. Variable lengths will reduce request or output-token throughput by creating
   less uniform batching and iteration completion.
4. Per-length buckets will expose latency differences hidden by aggregate means:
   long prompts should have the largest TTFT, and long output caps should have
   the largest E2E latency.
5. Long requests mixed with shorter requests will create latency dispersion
   consistent with head-of-line interference under closed-loop concurrency.

## Controlled Workload Design

| Property | Fixed | Variable |
| :--- | :--- | :--- |
| Input values | `[4096]` | `[1024, 4096, 7168]` |
| Input weights | `[1.0]` | `[0.20, 0.60, 0.20]` |
| Configured input mean | 4,096 | 4,096 |
| Maximum-output values | `[512]` | `[128, 512, 896]` |
| Maximum-output weights | `[1.0]` | `[0.20, 0.60, 0.20]` |
| Configured output-cap mean | 512 | 512 |
| Sampling | Stratified | Stratified |

For the canonical 50-request run, stratified sampling realizes exact category
counts of 10/30/10 for each variable distribution. This makes both realized
means exactly match the fixed profile while retaining shuffled, heterogeneous
request order. Input and output plans are independently shuffled, preventing a
forced correlation between prompt and output-cap length.

The pilot has only 12 measured requests, so its category allocation is an
approximation. It is a connectivity and artifact smoke test, not canonical
evidence.

## Other Controlled Variables

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Hardware target: 1× NVIDIA RTX 3090 24 GB
- Precision: BF16
- Backend: vLLM OpenAI-compatible endpoint
- Closed-loop concurrency: 4
- Measured requests: 50 per profile
- Warm-up requests: 5 per profile
- Temperature: 0.0
- Request inactivity timeout: 300 seconds
- Prefix caching: disabled
- Chunked prefill: disabled
- Seed: 42

## Metrics and Analysis

Aggregate metrics:

- TTFT, TPOT, and E2E P50/P95/P99,
- request throughput,
- input-, output-, and total-token throughput,
- error rate,
- GPU utilization and memory.

Distribution diagnostics:

- configured mean, population standard deviation, and coefficient of variation,
- realized mean, sample standard deviation, and percentiles,
- configured-versus-realized mean error,
- exact per-request workload plan.

Length-bucket analysis:

- TTFT, TPOT, and E2E distributions by target input length,
- TTFT, TPOT, and E2E distributions by maximum output length,
- actual input/output token statistics and failures in every bucket.

## Canonical Server Command

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --gpu-memory-utilization 0.90 \
  --port 18000
```

## Execution

Run the pilot first:

```bash
python experiments/E004-fixed-vs-variable/run_e004.py \
  --pilot \
  --base-url http://localhost:18000
```

Run the canonical comparison only after the pilot completes with zero errors:

```bash
python experiments/E004-fixed-vs-variable/run_e004.py \
  --base-url http://localhost:18000
```

The comparison directory contains:

```text
runs/E004_<timestamp>_<id>/
├── e004_summary.json
├── fixed/<run-id>/
├── variable/<run-id>/
└── plots/
    ├── distribution_comparison.png
    ├── latency_percentiles_by_profile.png
    ├── throughput_by_profile.png
    ├── length_bucket_latency.png
    └── gpu_by_profile.png
```

Each profile retains `config.json`, `environment.json`, `summary.json`,
`requests.jsonl`, `workload.jsonl`, and GPU telemetry when available. The E004
comparison summary embeds both profiles' length-bucket results so the analysis
is reproducible without reverse-engineering the plots.

## Interpretation Rules

- Matching configured means is verified before GPU execution.
- Canonical realized means should have zero sampling error by construction.
- A higher total-token throughput does not automatically mean better serving;
  input and output throughput must remain separate.
- P95/P99 comparisons must include bucket counts. With 10 requests in each tail
  category, bucket percentiles are descriptive and should not be overgeneralized.
- Bucket differences can reveal interference patterns, but this two-profile
  experiment cannot fully isolate every scheduler mechanism.

## Empirical Results (1× NVIDIA GeForce RTX 3090)

- Run ID: `E004_20260924_065119_c3c70a46`
- Source commit: `204b3961380299aa88c7b6d8ecbd37e1fe73af57`
- Backend: vLLM 0.30.0, BF16
- Workload: 50 measured requests per profile, 100/100 successful
- Realized means: exactly 4,096 input tokens and 512 output tokens per request

| Profile | Req/s | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 | Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 0.2816 | 2,582.03 ms | 3,368.47 ms | 3,423.06 ms | 13.789 s | 13.890 s | 13.910 s | 0% |
| Variable | 0.2823 | 1,640.65 ms | 3,848.66 ms | 3,979.81 ms | 13.436 s | 24.526 s | 25.543 s | 0% |

Key findings:

1. Variable E2E P95/P99 increased 76.6%/83.6% despite identical mean and total
   token work.
2. Variable median E2E remained close and median TTFT improved because 20% of
   prompts were short; median-only reporting would therefore miss the tail cost.
3. Throughput was effectively identical, rejecting the hypothesis that
   heterogeneity must reduce throughput at concurrency 4.
4. Median TTFT increased monotonically across 1,024/4,096/7,168-token input
   buckets, while median E2E increased from 4.915 s to 24.459 s across
   128/512/896-token output buckets.
5. Fixed and variable peak VRAM were identical at 19,858 MiB.

See the [complete validation report](../../outputs/e004_fixed_vs_variable_validation.md),
[canonical raw run](../../runs/E004_20260924_065119_c3c70a46/), and
[publication plots](../../outputs/plots/e004/).

## Limitations and What E004 Does Not Prove

- Workloads are synthetic discrete distributions, not production traces.
- Input and output lengths are independently assigned.
- The experiment uses one deterministic request order and does not estimate
  variance across repeated seeds.
- Traffic is closed-loop at concurrency 4 and therefore self-throttles.
- It does not identify a saturation rate or SLO-compliant capacity; those belong
  to E005's open-loop design.
- Prompt caching and chunked prefill remain disabled.
- Results apply only to the recorded model, hardware, backend, and software
  environment.
