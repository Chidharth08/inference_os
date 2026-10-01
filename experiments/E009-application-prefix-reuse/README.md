# E009 — Application-Shaped Prefix Reuse

E009 asks:

> Which application-shaped request relationships benefit from prefix caching,
> and does prompt ordering affect serving performance when prompt lengths and
> content remain comparable?

The profiles describe synthetic model requests. They do not execute chat, RAG,
summarization, tools, or agents, and they do not measure application quality.

## Controlled Profiles

E009 first pairs cache OFF and cache ON for five profiles at concurrency 1:

| Profile | Requests | Shape | Intended reusable prefix |
| :--- | ---: | :--- | :--- |
| `chat_like` | 24 | 4 conversations × 6 turns | 512-token system prompt, then growing exact history |
| `rag_friendly` | 24 | `[system][document][question]` | 3,328-token system/document prefix |
| `rag_hostile` | 24 | `[system][question][document]` | 256-token system prefix; the same document occurs after divergence |
| `summarization_like` | 24 | `[instruction][unique document]` | 256-token instruction |
| `agent_like` | 24 | 4 trajectories × 6 calls | 4,000-token system/tools prefix, then growing exact state |

Chat inputs grow from 1,024 to 3,584 tokens. Agent inputs grow from 5,504
to 8,064 tokens and output limits range from 64 to 256 tokens. Conversation and
trajectory plans are generated completely before measurement; no measured
request depends on nondeterministic model output.

RAG-friendly and RAG-hostile prompts use the same deterministic system and
document content, total 4,096-token length, and 64-token output limit. Only the
question/document ordering changes.

All controlled conditions require a fresh vLLM server. The runner verifies the
live cache setting and block size, requires pristine counters for cache-ON runs,
uses disjoint warm-ups, and requires measured hit tokens to exactly equal the
cacheable token-level request plan.

## Selected Open-Loop Stage

The second stage compares the two RAG layouts under deterministic open-loop
traffic at 1 request/s:

```text
rag_friendly × cache OFF/ON
rag_hostile  × cache OFF/ON
```

Each condition offers 24 requests. The SLO is:

- TTFT P95 ≤ 1 second;
- E2E P95 ≤ 3 seconds;
- error rate ≤ 1%.

This rate is a representative pressure point informed by earlier capacity
measurements, not an automatic capacity recommendation. Open-loop cache-hit
counts are observed rather than required to match a sequential theoretical
maximum because overlapping requests can query before an earlier request has
finished populating a prefix.

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

Restart vLLM before every condition, including between profiles using the same
cache setting.

## Controlled Condition Command

```bash
python experiments/E009-application-prefix-reuse/run_e009.py \
  --config configs/e009_cache_on.yaml \
  --profile agent_like \
  --pilot
```

Valid profiles are:

```text
chat_like
rag_friendly
rag_hostile
summarization_like
agent_like
```

Use the matching cache-OFF or cache-ON configuration and remove `--pilot` for a
canonical run. A pilot uses two disjoint warm-ups and eight measurements; a
canonical controlled run uses three warm-ups and 24 measurements.

## Open-Loop Condition Command

Only the two RAG profiles are eligible:

```bash
python experiments/E009-application-prefix-reuse/run_e009.py \
  --config configs/e009_cache_on.yaml \
  --profile rag_friendly \
  --load-rate 1.0 \
  --pilot
```

Remove `--pilot` for the canonical 24-request condition. Restart vLLM before
each of the four profile/cache combinations.

## Summarization

Combine the ten controlled directories:

```bash
python experiments/E009-application-prefix-reuse/run_e009.py \
  --summarize \
  runs/<chat-off> runs/<chat-on> \
  runs/<rag-friendly-off> runs/<rag-friendly-on> \
  runs/<rag-hostile-off> runs/<rag-hostile-on> \
  runs/<summarization-off> runs/<summarization-on> \
  runs/<agent-off> runs/<agent-on>
```

Combine the four load directories:

```bash
python experiments/E009-application-prefix-reuse/run_e009.py \
  --summarize-load \
  runs/<rag-friendly-off-load> runs/<rag-friendly-on-load> \
  runs/<rag-hostile-off-load> runs/<rag-hostile-on-load>
```

Directory order does not matter. Both summarizers reject missing, duplicate, or
unexpected conditions.

## Artifacts

Each condition retains standard benchmark artifacts plus raw cache metric
snapshots, cache deltas, verified server configuration, and
`e009_condition.json`. The workload artifact includes:

- application profile and prompt layout;
- relationship/conversation/trajectory identifier;
- turn or call index;
- exact prompt and reusable-prefix hashes;
- intended stable prefix;
- actual common-prefix tokens;
- expected cache state.

The controlled comparison produces:

```text
e009_summary.json
plots/latency_by_application_profile.png
plots/cache_reuse_and_ttft_benefit.png
plots/cacheable_prefix_by_sequence_index.png
```

The open-loop comparison produces:

```text
e009_load_summary.json
plots/throughput_and_goodput_by_rag_layout.png
plots/open_loop_latency_by_rag_layout.png
```

## Interpretation Boundaries

Input-token throughput represents effective workload tokens, not physical
prefill computation after cached tokens are removed. Cache counters must be
interpreted alongside latency and throughput.

E009 does not measure retrieval quality, generated-answer quality, summary
quality, tool correctness, agent success, or end-user task completion. It does
not characterize cache eviction, multi-tenant pressure, distributed caching, or
production traffic.

## Published Results

The canonical RTX 3090 collection is complete. All ten controlled and four
open-loop conditions passed their execution and mechanism checks. The raw runs
are preserved in `runs/E009_20261001_*`; derived comparisons and plots are in
`runs/E009_controlled_20261001_173303_29917f34` and
`runs/E009_load_20261001_173304_917d3f2c`.

See `outputs/e009_application_prefix_reuse/e009_report.md` for the complete
analysis, `e009_summary.json` for the controlled comparison, and
`e009_load_summary.json` for the open-loop comparison.
