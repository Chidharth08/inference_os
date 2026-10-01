# E008 — Prefix Reuse Sensitivity

E008 asks:

> How does prefix-caching benefit change as the reusable fraction of a fixed-size
> prompt increases?

It pairs cache OFF and cache ON at five requested reusable-prefix fractions:

| Requested fraction | Resolved shared tokens | Resolved fraction | Unique suffix |
| ---: | ---: | ---: | ---: |
| 0% | 0 | 0.00% | 4,096 |
| 25% | 1,024 | 25.00% | 3,072 |
| 50% | 2,048 | 50.00% | 2,048 |
| 75% | 3,072 | 75.00% | 1,024 |
| 90% | 3,680 | 89.84% | 416 |

The 90% target is rounded down to the nearest 16-token cache-block boundary.
Every condition holds total input length at 4,096 tokens, output limit at 64
tokens, concurrency at 1, and request order and seed constant.

## Isolation and Validity

Run every condition against a newly started vLLM server. This is required even
between cache-ON fractions: the deterministic prefixes are related, so residual
blocks from one condition could contaminate the compulsory miss of another.

Before cache-enabled traffic, the runner verifies:

- prefix caching is enabled on the live server;
- the live cache block size is 16 tokens;
- prefix-cache query and hit counters are pristine.

Warm-up prompts are disjoint from measured prompts. After warm-up, the runner
captures a metrics snapshot, executes the measured plan, captures another
snapshot, and compares measured-window hit tokens to the exact cacheable prefix
tokens in `workload.jsonl`. A condition is invalid if the counts disagree.

At 0%, prompts are unique within the first cacheable block. At positive
fractions, request 1 is the compulsory miss and requests 2–30 share the resolved
prefix.

## Server Commands

Cache OFF:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 18000 \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill
```

Cache ON:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 18000 \
  --dtype bfloat16 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --no-enable-chunked-prefill
```

## Condition Commands

Choose the config matching the currently running server and run one fraction:

```bash
python experiments/E008-prefix-reuse-sensitivity/run_e008.py \
  --config configs/e008_cache_off.yaml \
  --shared-fraction 25 \
  --pilot
```

Remove `--pilot` for the canonical condition. Valid fraction values are `0`,
`25`, `50`, `75`, and `90`.

The complete experiment comprises these ten isolated combinations:

```text
cache OFF: 0, 25, 50, 75, 90
cache ON:  0, 25, 50, 75, 90
```

Restart vLLM before every command. Do not run the five values in a shell loop
against one server.

## Summarization

After all ten canonical conditions complete, pass their directories in any
order:

```bash
python experiments/E008-prefix-reuse-sensitivity/run_e008.py \
  --summarize \
  runs/<0-off> runs/<0-on> \
  runs/<25-off> runs/<25-on> \
  runs/<50-off> runs/<50-on> \
  runs/<75-off> runs/<75-on> \
  runs/<90-off> runs/<90-on>
```

The summarizer rejects missing, duplicate, or unexpected conditions. It writes:

```text
e008_summary.json
plots/ttft_vs_shared_fraction.png
plots/reuse_vs_ttft_improvement.png
plots/e2e_tpot_vs_shared_fraction.png
```

## Condition Artifacts

Each condition preserves the standard benchmark artifacts plus:

```text
cache_metrics_preflight.prom
cache_metrics_before.prom
cache_metrics_after.prom
cache_metrics_delta.json
server_cache_config.json
e008_condition.json
```

`e008_condition.json` retains the requested fraction, fractional token target,
resolved cache-aligned boundary, actual token-level reuse range, expected cache
hit tokens, observed hit fraction, latency by cache phase, and mechanism checks.

## Interpretation

The primary relationship is observed cached-token fraction versus TTFT change
relative to the paired cache-OFF condition. Overall measured-window hit fraction
includes the compulsory miss; steady-state hit fraction excludes it. Both are
reported explicitly.

TPOT is a sanity check and should remain broadly stable. Request throughput is
reported but is a concurrency-1 elapsed-time observation, not a production
capacity claim.

E008 does not characterize eviction, reuse distance, working-set capacity,
popularity distributions, or concurrent traffic. Those are outside this
controlled sensitivity experiment.
