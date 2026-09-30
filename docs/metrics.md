# Metric Definitions

This document defines the canonical request, throughput, workload, and GPU
metrics emitted by `inference_os`.

## Timing Source

Request lifecycle timestamps use a monotonic nanosecond clock. They are suitable
for elapsed-time calculations and are not interpreted as wall-clock dates.

## Request Latency

### Time to First Token (TTFT)

```text
TTFT = first_token_time - request_start_time
```

TTFT includes client-observed HTTP/network overhead, backend queueing, prompt
prefill, and delivery of the first non-empty streamed text chunk. It does not
separate those components.

### End-to-End Latency (E2E)

```text
E2E = completion_time - request_start_time
```

Completion is observed when the response stream ends or fails.

### Time Per Output Token (TPOT)

```text
TPOT = (completion_time - first_token_time) / (actual_output_tokens - 1)
```

The first output token is attributed to prefill/TTFT. TPOT is undefined when a
first token was not observed or fewer than two output tokens were generated.

TPOT is an average decode interval. V1–V2 do not store every individual
inter-token arrival timestamp.

## Aggregate Statistics

For TTFT, E2E, and TPOT, successful requests contribute to distribution
statistics. The framework reports:

- count,
- mean,
- sample standard deviation,
- minimum and maximum,
- P50, P90, P95, and P99 using linear interpolation.

Failed requests are counted in error metrics and excluded from latency
percentiles.

## Throughput

All throughput metrics use the measured benchmark phase's wall-clock duration,
not the sum of overlapping request durations.

### Request Throughput

```text
successful requests / measured wall-clock seconds
```

### Input-Token Throughput

```text
total tokenizer-observed input tokens submitted / measured wall-clock seconds
```

The current aggregate includes measured requests that later fail because the
client cannot always determine how much input work a remote backend completed.
Error rate must therefore be reported alongside token throughput.

### Output-Token Throughput

```text
total tokenizer-observed output tokens from successful requests
/ measured wall-clock seconds
```

### Total Token Throughput

```text
(total observed input tokens + successful observed output tokens)
/ measured wall-clock seconds
```

Input, output, and total token throughput describe different work mixes and must
not be substituted for one another.

## Error Metrics

```text
error rate = failed measured requests / total measured requests
```

The raw error message is retained for each failed request when available.

## E003 Workload Metrics

### Target Input Tokens

The configured request-plan target passed to tokenizer-aware synthetic prompt
generation. The actual tokenizer-observed input count remains a request
measurement.

### Maximum Output Tokens

The per-request generation limit sent to the backend. This is not an actual
output-token measurement because the model may stop early.

### Realized Workload Summary

The measured (non-warm-up) request plan is summarized with the same distribution
statistics used by other numerical metrics. The complete ordered plan is stored
in `workload.jsonl`; summary statistics are not a substitute for that artifact.

## GPU Telemetry

GPU memory and utilization are sampled periodically through `nvidia-smi` while
the warm-up and measured benchmark phases execute.

- peak GPU memory is the maximum sampled `memory.used`,
- average GPU memory is the arithmetic mean of samples,
- peak GPU utilization is the maximum sampled utilization percentage,
- average GPU utilization is the arithmetic mean of samples.

`nvidia-smi` utilization is a coarse activity signal. It does not directly
measure tensor-core efficiency, memory-bandwidth saturation, useful FLOPs, or
end-to-end serving capacity.

## E007 Prefix-Cache Metrics

E007 captures raw vLLM Prometheus snapshots immediately after disjoint warm-up
and immediately after the measured request window. Counter metrics are reported
as non-negative differences between those snapshots.

### Prefix-Cache Query Tokens

The number of prompt tokens queried against the prefix cache during the measured
window. This metric is derived from the vLLM prefix-cache query counter for the
pinned server version.

### Prefix-Cache Hit Tokens

The number of queried prompt tokens found in the prefix cache during the
measured window.

### Observed Prefix-Cache Hit Fraction

```text
measured-window prefix-cache hit tokens
/ measured-window prefix-cache query tokens
```

The fraction is undefined when the query-token delta is zero or unavailable.
It is token weighted, not a percentage of requests that had at least one hit.

### Configured and Actual Reusable Prefix

`configured_shared_prefix_tokens` is the intended common-prefix length from the
workload configuration. `actual_reusable_prefix_tokens` is the exact common
token prefix between the current request and a previously scheduled measured
request.

Neither field proves that vLLM reused those tokens. Observed server hit counters
provide that evidence.

### Input Throughput Under Prefix Caching

Existing input-token throughput counts tokenizer-observed tokens submitted by
the client. It is effective workload throughput and does not claim that every
submitted prompt token was physically recomputed. Cache-hit and cached-prompt
metrics must be reported separately.
