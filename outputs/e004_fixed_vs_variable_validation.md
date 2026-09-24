# E004 Fixed versus Variable Workloads — Validation Report

## Executive Summary

E004 compared fixed and heterogeneous token-length workloads with exactly the
same realized mean input length (4,096 tokens), mean maximum output length (512
tokens), total input tokens (204,800), total output tokens (25,600), concurrency,
and request count. All 100 measured requests completed successfully.

The variable workload did not reduce throughput: both profiles completed in
about 177 seconds at approximately 0.282 requests/s. It did, however, transform
the latency distribution. Relative to fixed:

- median E2E latency remained close and was 2.6% lower,
- E2E P95 increased 76.6%, from 13.890 s to 24.526 s,
- E2E P99 increased 83.6%, from 13.910 s to 25.543 s,
- TTFT P95 increased 14.3%, and TTFT P99 increased 16.3%,
- median TTFT decreased 36.5% because 20% of prompts were only 1,024 tokens.

The central result is that equal average token counts and nearly identical
throughput did not imply equal user-facing latency tails. Average sequence
length is therefore insufficient to characterize a mixed serving workload.

## Run Identity and Environment

| Item | Value |
| :--- | :--- |
| Run ID | `E004_20260924_065119_c3c70a46` |
| Source commit | `204b3961380299aa88c7b6d8ecbd37e1fe73af57` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| GPU | 1× NVIDIA GeForce RTX 3090, 24,576 MiB |
| Driver | 595.71.05 |
| PyTorch | 2.13.0+cu132 |
| vLLM | 0.30.0 |
| Python | 3.12.14 |
| Precision | BF16 |
| Concurrency | 4, closed loop |
| Requests | 5 warm-up + 50 measured per profile |
| Prefix caching / chunked prefill | Disabled / disabled |
| Sampling | Temperature 0.0, seed 42 |
| Request inactivity timeout | 300 seconds |

The captured Git status was dirty only because the environment inventory and
pilot/canonical run directories were untracked on the ephemeral host. The
recorded source commit was cleanly merged on `main`; no source-code differences
were involved in benchmark execution. `server_command.txt` was reconstructed
locally from the exact canonical command used and documented before the run; it
was the only intended metadata file absent from the downloaded archive.

## Controlled Workload Validation

| Property | Fixed | Variable |
| :--- | ---: | ---: |
| Input-length coefficient of variation | 0.0000 | 0.4743 |
| Output-cap coefficient of variation | 0.0000 | 0.4743 |
| Realized mean input tokens | 4,096 | 4,096 |
| Input mean sampling error | 0 | 0 |
| Realized mean maximum output tokens | 512 | 512 |
| Output mean sampling error | 0 | 0 |
| Total actual input tokens | 204,800 | 204,800 |
| Total actual output tokens | 25,600 | 25,600 |

The fixed profile contained 50 requests of shape 4,096/512. The variable
profile realized exact 10/30/10 counts for both distributions:

- input targets: 1,024 / 4,096 / 7,168 tokens,
- output caps: 128 / 512 / 896 tokens.

Every generation reached its configured maximum-output cap, so actual aggregate
output work also matched exactly.

## Aggregate Results

| Profile | Req/s | Input tok/s | Output tok/s | Total tok/s | TTFT P50 | TTFT P95 | TTFT P99 | TPOT P50 | E2E P50 | E2E P95 | E2E P99 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed | 0.2816 | 1,153.59 | 144.20 | 1,297.79 | 2,582.03 ms | 3,368.47 ms | 3,423.06 ms | 21.69 ms | 13.789 s | 13.890 s | 13.910 s |
| Variable | 0.2823 | 1,156.12 | 144.51 | 1,300.63 | 1,640.65 ms | 3,848.66 ms | 3,979.81 ms | 23.65 ms | 13.436 s | 24.526 s | 25.543 s |

### Tail amplification

Fixed E2E latency was tightly clustered: P99 was only 0.9% above P50. For the
variable workload, P99 was 90.1% above P50. TTFT showed the same qualitative
change: fixed P99 was 1.33× its median, while variable P99 was 2.43× its median.

The variable profile's median improved because its distribution included short
requests, but its upper tail degraded because it also included substantially
larger requests. Reporting only the mean or median would hide this trade-off.

## Length-Bucket Results

### Variable workload by input target

| Input tokens | Requests | TTFT P50 | TTFT P95 | TTFT P99 | E2E P50 | E2E P95 | E2E P99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 10 | 353.73 ms | 1,942.87 ms | 1,981.20 ms | 12.445 s | 18.771 s | 21.999 s |
| 4,096 | 30 | 1,392.38 ms | 3,627.34 ms | 3,928.92 ms | 13.436 s | 24.570 s | 25.918 s |
| 7,168 | 10 | 1,798.69 ms | 3,933.24 ms | 3,978.45 ms | 15.293 s | 23.854 s | 24.371 s |

Median TTFT rose monotonically with input length. The 7,168-token bucket's
median was 5.08× the 1,024-token bucket's median. The short-input bucket also
had a P95 5.49× its own median, showing that short prompts did not always receive
short first-token latency when sharing a heterogeneous concurrent workload.

This is consistent with queueing and head-of-line interference, but the bucket
comparison alone does not isolate scheduler causality: output caps were assigned
independently, and requests overlapped under closed-loop execution.

### Variable workload by maximum output length

| Maximum output | Requests | Actual output mean | TPOT P50 | TPOT P95 | E2E P50 | E2E P95 | E2E P99 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10 | 128 | 20.93 ms | 36.52 ms | 4.915 s | 6.318 s | 6.611 s |
| 512 | 30 | 512 | 23.65 ms | 27.66 ms | 13.436 s | 15.387 s | 15.678 s |
| 896 | 10 | 896 | 24.51 ms | 26.33 ms | 24.459 s | 25.618 s | 26.292 s |

E2E latency scaled strongly with generation length: the 896-token bucket's
median was 4.98× the 128-token bucket's median. TPOT medians changed much less,
showing that the large E2E difference primarily came from performing more
sequential decode steps rather than a proportional increase in per-token time.

The unusually high TPOT tail in the 128-token bucket is based on only 10
requests and may reflect concurrent interference and the deterministic request
order. It should not be generalized without repeated seeds.

## Throughput and GPU Results

| Profile | Duration | Avg GPU utilization | Peak VRAM | Errors |
| :--- | ---: | ---: | ---: | ---: |
| Fixed | 177.533 s | 98.83% | 19,858 MiB | 0% |
| Variable | 177.144 s | 99.48% | 19,858 MiB | 0% |

Request throughput differed by only 0.25%, and total-token throughput differed
by only 0.22%. At concurrency 4 on this server, heterogeneity changed latency
distribution rather than aggregate throughput or peak memory. This is an
important negative result: variance does not automatically reduce throughput
when total work and offered closed-loop concurrency remain controlled.

## Hypothesis Evaluation

| Hypothesis | Empirical result | Status |
| :--- | :--- | :--- |
| Variable lengths worsen P95/P99 latency. | TTFT P95/P99 rose 14.3%/16.3%; E2E P95/P99 rose 76.6%/83.6%. | Confirmed |
| Median latency remains closer than tail latency. | E2E P50 differed by only −2.6%, but TTFT P50 fell 36.5% because of short prompts. | Partially confirmed |
| Variability reduces throughput. | Request and token throughput were effectively unchanged; variable was slightly higher. | Rejected for this setup |
| Length buckets expose behavior hidden by averages. | Input length separated median TTFT; output length separated E2E latency by nearly 5×. | Confirmed |
| Mixed lengths create dispersion consistent with head-of-line interference. | Short-input TTFT P95 was 5.49× its median, while aggregate variable tails widened sharply. | Supported, not causally proven |

## Why Average Length Was Insufficient

Both profiles performed exactly the same aggregate input and output work. A
mean-only model would therefore predict similar behavior—and it correctly
predicted similar total runtime and throughput. It failed to predict latency
distribution because individual request completion depends on each request's
prefill/decode work and on the other requests with which it overlaps.

The fixed workload kept all four concurrent slots similar in shape, producing a
tight E2E distribution. The variable workload mixed requests that completed at
very different times. Short requests improved the median while long requests
and concurrent interference stretched the upper tail. Capacity and SLO planning
must therefore model distributions, not only average tokens per request.

## Artifacts

- Raw canonical run: [`runs/E004_20260924_065119_c3c70a46/`](../runs/E004_20260924_065119_c3c70a46/)
- Distribution comparison: [`plots/e004/distribution_comparison.png`](plots/e004/distribution_comparison.png)
- Aggregate latency: [`plots/e004/latency_percentiles_by_profile.png`](plots/e004/latency_percentiles_by_profile.png)
- Throughput: [`plots/e004/throughput_by_profile.png`](plots/e004/throughput_by_profile.png)
- Length buckets: [`plots/e004/length_bucket_latency.png`](plots/e004/length_bucket_latency.png)
- GPU comparison: [`plots/e004/gpu_by_profile.png`](plots/e004/gpu_by_profile.png)

The raw run retains both profiles' configurations, environment snapshots,
per-request measurements, exact workload plans, GPU telemetry, aggregate and
bucket summaries, plots, package inventory, GPU report, served-model response,
source revision, and canonical server command.

## Limitations

1. This is one deterministic ordering on one RTX 3090; repeated seeds are needed
   to estimate run-to-run and ordering variance.
2. Each tail category contains 10 requests. Its P95/P99 values are descriptive,
   not precise population estimates.
3. Input and output lengths were assigned independently. Input-bucket E2E and
   output-bucket TTFT therefore include variation from the other dimension.
4. Closed-loop concurrency self-throttles and does not represent an externally
   imposed arrival rate.
5. Prefix caching and chunked prefill were disabled.
6. The experiment identifies latency-distribution effects but does not determine
   saturation or SLO-compliant capacity; E005 introduces open-loop load for that.
7. Conclusions apply only to the recorded model, precision, backend, server
   configuration, hardware, and software environment.

## Conclusion

E004 demonstrates that mean-matched workloads can have virtually identical
throughput yet radically different tail latency. It validates distribution and
length-bucket reporting as necessary benchmark capabilities and motivates E005:
measuring how these workload shapes behave when arrivals no longer self-throttle
and the server approaches saturation.
