# E003 — Application-Shaped Workload Profiles

## Primary Question

> How does application request shape affect inference performance when the model,
> hardware, serving backend, and client concurrency remain unchanged?

E003 compares three synthetic token-shape profiles:

- `chat_like`: predominantly short inputs and short outputs,
- `rag_like`: long inputs and moderate outputs,
- `summarization_like`: long inputs and long outputs.

These labels describe controlled token-length distributions. E003 does not build
chat, retrieval, or summarization applications and does not claim to reproduce a
specific production traffic trace.

## Hypotheses

1. `chat_like` will achieve the highest request throughput because each request
   performs less prefill and decode work.
2. `rag_like` will have higher TTFT because its longer inputs require more
   prefill computation.
3. `summarization_like` will have the highest E2E latency because it combines
   long prefills with long sequential decoding.
4. `rag_like` and `summarization_like` will show wider P95–P50 latency gaps than
   `chat_like` because their broader, more expensive request mixes create more
   interference under continuous batching.
5. Output-token throughput may not rank profiles in the same order as request
   throughput because larger batches can amortize model-weight reads differently.

## Controlled Variables

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Hardware: 1× NVIDIA RTX 3090 24 GB
- Precision: BF16
- Backend: vLLM OpenAI-compatible endpoint
- Closed-loop concurrency: 4
- Measured requests: 50 per profile
- Warm-up requests: 5 per profile
- Temperature: 0.0
- Prefix caching: disabled
- Chunked prefill: disabled
- Seed: 42

The independent variable is the configured input/output token-length
distribution. Input and maximum-output lengths are sampled independently in
E003. Correlated request shapes are outside this experiment.

## Metrics

- TTFT, TPOT, and E2E P50/P95
- request throughput
- actual input-token throughput
- actual output-token throughput
- total token throughput
- error rate
- GPU utilization
- GPU memory
- realized target input/output distribution statistics

Requested maximum output length is not treated as an actual output count. The
model may stop early, and actual tokenizer-observed output tokens remain in
`requests.jsonl`.

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

Run a low-cost pilot first:

```bash
python experiments/E003-application-workload-shapes/run_e003.py --pilot
```

Run the canonical comparison:

```bash
python experiments/E003-application-workload-shapes/run_e003.py
```

The comparison directory contains:

```text
runs/E003_<timestamp>_<id>/
├── e003_summary.json
├── chat_like/<run-id>/
├── rag_like/<run-id>/
├── summarization_like/<run-id>/
└── plots/
    ├── latency_by_profile.png
    ├── throughput_by_profile.png
    ├── workload_shape_by_profile.png
    └── gpu_by_profile.png
```

Each profile run also contains `config.json`, `environment.json`, `summary.json`,
`requests.jsonl`, `workload.jsonl`, and GPU telemetry when available.

## Empirical Results (1× NVIDIA GeForce RTX 3090)

- Run ID: `E003_20260924_051307_29213e50`
- Source commit: `29ed58d6c0ee1edbf0077bd4b04aeba607b6e492`
- Backend: vLLM 0.30.0, BF16
- Workload: 50 measured requests per profile, 150/150 successful

| Profile | Req/s | Input tok/s | Output tok/s | TTFT P50 | E2E P50 | Peak VRAM | Errors |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat_like` | 1.566 | 577.28 | 124.46 | 216.50 ms | 1.810 s | 18,060 MiB | 0% |
| `rag_like` | 0.372 | 1,815.57 | 84.87 | 2,026.88 ms | 8.676 s | 21,024 MiB | 0% |
| `summarization_like` | 0.218 | 1,082.44 | 112.94 | 2,131.41 ms | 17.293 s | 21,024 MiB | 0% |

The main findings are:

1. Chat achieved the highest request throughput and lowest latency.
2. The two long-input profiles had roughly 9–10× chat's median TTFT.
3. Summarization had the highest E2E latency because it combined long prefills
   with long decoding.
4. RAG had the highest total-token throughput due to input-token volume, despite
   having the lowest output-token throughput.
5. Long-context profiles used roughly 3 GiB more peak VRAM than chat.

See the [complete validation report](../../outputs/e003_application_workload_shapes_validation.md),
[canonical raw run](../../runs/E003_20260924_051307_29213e50/), and
[publication plots](../../outputs/plots/e003/).

## Limitations

- Profiles are synthetic and configuration-defined.
- Input and output distributions are sampled independently.
- Traffic remains closed-loop at a fixed concurrency.
- Prompt content is synthetic; only token shape is being studied.
- Prompt caching and chunked prefill are disabled and not evaluated.
- Results apply only to the recorded model, backend, hardware, and server setup.
- E003 compares performance under equal client concurrency, not equal offered
  token load or equal production arrival rate.

## What E003 Does Not Prove

E003 does not establish production capacity, cost, application quality, or a
universally representative workload. Those require later experiments with
open-loop arrivals, SLOs, real traffic evidence, or quality evaluation.
