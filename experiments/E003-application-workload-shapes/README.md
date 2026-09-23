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
