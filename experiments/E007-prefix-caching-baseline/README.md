# E007 — Prefix-Caching Mechanism Validation

E007 asks:

> Does verified prefix reuse reduce prefill latency, and can the benchmark
> distinguish a real cache hit from merely enabling prefix caching?

It is a 2×2 controlled experiment:

| Prompt pattern | Cache OFF | Cache ON |
|---|---:|---:|
| Unique prefixes | Yes | Yes |
| Shared prefix plus unique suffix | Yes | Yes |

The unique-prefix conditions are negative controls. The shared-prefix conditions
use a 4096-token prompt containing a 3072-token common prefix and a 1024-token
unique suffix. All conditions use a 64-token output limit and concurrency 1.

## Why Conditions Run Separately

Prefix caching is a vLLM server setting. One running server cannot represent
both cache-OFF and cache-ON conditions. Each condition therefore runs against a
fresh server whose effective configuration is verified through `/metrics`.

The condition runner fails before sending benchmark traffic when the live
server's prefix-caching state or cache block size disagrees with its YAML file.

## Measurement Protocol

Each condition performs:

1. a preflight `/metrics` capture and server-cache configuration check,
2. disjoint warm-up requests that cannot share a full block with measured
   prompts,
3. a second `/metrics` capture immediately after warm-up,
4. measured requests,
5. a final `/metrics` capture,
6. counter-delta calculation restricted to the measured window.

For cache-ON conditions, preflight prefix-cache query and hit counters must be
zero. A non-pristine server is rejected because the first target request could
otherwise be an unrecognized hit.

For cache-ON shared-prefix traffic, the first measured request is labeled
`cold`; later requests are labeled `warm`. Cache-OFF requests are labeled
`disabled`. Unique cache-ON requests after the first are labeled
`no_cacheable_prefix`.

The workload artifact stores token-sequence hashes, reuse-group identity,
configured shared-prefix length, actual reusable prefix length, and expected
cache state. Raw Prometheus snapshots are retained for audit.

## Server Commands

Use the same model and all the same server settings except prefix caching.

Cache OFF:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 18000 \
  --dtype bfloat16 \
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
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --no-enable-chunked-prefill
```

If the installed vLLM version uses different CLI spellings, record the exact
working commands. The runner trusts observed `/metrics` configuration rather
than command text.

## Pilot Procedure

With a fresh cache-OFF server:

```bash
python experiments/E007-prefix-caching-baseline/run_e007.py \
  --config configs/e007_cache_off_unique.yaml --pilot

# Restart the cache-OFF server before the next condition.
python experiments/E007-prefix-caching-baseline/run_e007.py \
  --config configs/e007_cache_off_repeated.yaml --pilot
```

Restart vLLM with prefix caching enabled, then run:

```bash
python experiments/E007-prefix-caching-baseline/run_e007.py \
  --config configs/e007_cache_on_unique.yaml --pilot

# Restart the cache-ON server before the next condition.
python experiments/E007-prefix-caching-baseline/run_e007.py \
  --config configs/e007_cache_on_repeated.yaml --pilot
```

Inspect each `e007_condition.json` before proceeding. In particular:

- cache ON must produce cache-query tokens,
- unique-prefix/cache-ON must produce zero hit tokens,
- shared-prefix/cache-ON must produce positive hit tokens,
- every condition must report the expected live cache configuration.

## Canonical Procedure

Repeat the same four commands without `--pilot`, restarting vLLM before every
condition. Alternate the order across independent repetitions so cache-ON or
cache-OFF does not always run first.

After collecting one complete set, combine the four run directories:

```bash
python experiments/E007-prefix-caching-baseline/run_e007.py \
  --summarize \
  runs/<unique-off-run> \
  runs/<unique-on-run> \
  runs/<shared-off-run> \
  runs/<shared-on-run>
```

The order of directories does not matter. The summarizer rejects missing,
unexpected, or duplicate conditions.

## Condition Artifacts

Each condition preserves the standard run artifacts plus:

```text
cache_metrics_preflight.prom
cache_metrics_before.prom
cache_metrics_after.prom
cache_metrics_delta.json
server_cache_config.json
e007_condition.json
```

The four-condition comparison contains:

```text
e007_summary.json
plots/latency_by_condition.png
plots/cache_observations_by_condition.png
```

## Interpretation Rules

E007 supports a prefix-caching conclusion only when:

1. the server configuration is verified,
2. shared-prefix/cache-ON records positive cache-hit tokens,
3. unique-prefix/cache-ON records no cacheable-prefix hits,
4. measured requests complete successfully,
5. TTFT changes are interpreted together with cache evidence.

TPOT is a sanity check. It should remain broadly similar because automatic
prefix caching avoids repeated prefill work rather than decode work.

## Limitations

E007 uses one model, one GPU, one fixed prompt shape, one output limit, and
concurrency 1. It does not establish production capacity, cache-eviction
behavior, working-set effects, or application-level benefit. Those questions
belong to E008 and E009.
