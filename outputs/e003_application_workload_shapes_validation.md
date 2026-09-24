# E003 Application Workload Shapes — Validation Report

## Executive Summary

E003 measured three deterministic synthetic workload profiles on one NVIDIA
GeForce RTX 3090 while holding the model, serving configuration, concurrency,
and request count constant. All 150 measured requests completed successfully.

The experiment confirms that request shape materially changes which performance
metric looks favorable:

- `chat_like` delivered the best request throughput and responsiveness.
- `rag_like` and `summarization_like` had roughly 9–10× higher median TTFT than
  `chat_like` because their prompts required much more prefill work.
- `summarization_like` had the highest end-to-end latency because long prompts
  were followed by long generation.
- `rag_like` produced the highest total-token throughput, but this was driven by
  input tokens. It did not provide the best request throughput, output-token
  throughput, or latency.

These results demonstrate why a single fixed-length benchmark or a single
throughput number cannot characterize application-facing inference behavior.

## Run Identity and Environment

| Item | Value |
| :--- | :--- |
| Run ID | `E003_20260924_051307_29213e50` |
| Source commit | `29ed58d6c0ee1edbf0077bd4b04aeba607b6e492` |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| GPU | 1× NVIDIA GeForce RTX 3090, 24,576 MiB |
| Driver / CUDA reported by `nvidia-smi` | 590.48.01 / 13.1 |
| PyTorch | 2.13.0+cu132 |
| vLLM | 0.30.0 |
| Python | 3.12.14 |
| Precision | BF16 |
| Concurrency | 4, closed loop |
| Requests | 5 warm-up + 50 measured per profile |
| Prefix caching / chunked prefill | Disabled / disabled |
| Sampling | Temperature 0.0, seed 42 |
| Request inactivity timeout | 300 seconds |

The captured environment reports the worktree as dirty because a local
environment inventory file was added before execution. The benchmark source
commit is recorded separately and matches the commit above. The full package
inventory and GPU report are retained with the raw run.

## Workload Shapes

The profiles sample input and maximum-output lengths independently from fixed,
seeded distributions. The realized measured-request means were:

| Profile | Mean target input tokens | Mean maximum output tokens | Workload interpretation |
| :--- | ---: | ---: | :--- |
| `chat_like` | 368.64 | 82.56 | Mostly short prompts and short answers |
| `rag_like` | 4,874.24 | 227.84 | Long context with moderate answers |
| `summarization_like` | 4,956.16 | 517.12 | Long context with long answers |

These are token-shape labels, not real chat, retrieval, or summarization
pipelines. Prompt content was synthetic.

## Results

All profiles completed 50/50 measured requests with a 0% error rate.

| Profile | Req/s | Input tok/s | Output tok/s | Total tok/s | TTFT P50 | TTFT P95 | TPOT P50 | E2E P50 | E2E P95 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `chat_like` | 1.566 | 577.28 | 124.46 | 701.74 | 216.50 ms | 507.99 ms | 24.77 ms | 1.810 s | 4.891 s |
| `rag_like` | 0.372 | 1,815.57 | 84.87 | 1,900.44 | 2,026.88 ms | 5,089.50 ms | 34.16 ms | 8.676 s | 20.965 s |
| `summarization_like` | 0.218 | 1,082.44 | 112.94 | 1,195.38 | 2,131.41 ms | 4,648.11 ms | 28.83 ms | 17.293 s | 33.633 s |

### GPU Results

| Profile | Average GPU utilization | Peak GPU utilization | Average VRAM | Peak VRAM |
| :--- | ---: | ---: | ---: | ---: |
| `chat_like` | 96.72% | 100% | 18,010 MiB | 18,060 MiB |
| `rag_like` | 99.56% | 100% | 20,639 MiB | 21,024 MiB |
| `summarization_like` | 99.79% | 100% | 21,024 MiB | 21,024 MiB |

## Hypothesis Evaluation

| Hypothesis | Result | Status |
| :--- | :--- | :--- |
| Chat has the highest request throughput. | Chat reached 1.566 req/s, 4.20× RAG and 7.17× summarization. | Confirmed |
| Long-input profiles have higher TTFT. | Median TTFT was 9.36× chat for RAG and 9.85× chat for summarization. RAG P95 was the highest, while summarization P50 was slightly higher. | Confirmed, with mixed ordering between the two long-input profiles |
| Summarization has the highest E2E latency. | Its median E2E was 17.293 s, 1.99× RAG and 9.55× chat. | Confirmed |
| Long profiles have wider latency tails. | TTFT P95–P50 gaps were 0.291 s (chat), 3.063 s (RAG), and 2.517 s (summarization). E2E tail gaps widened further. | Confirmed |
| Output-token throughput can rank profiles differently from request throughput. | RAG exceeded summarization in req/s, while summarization exceeded RAG in output tok/s. | Confirmed |

## Interpretation

### Request throughput favors small jobs

At equal concurrency, the closed-loop client can recycle chat requests much
more quickly. This is why chat completed 1.566 requests per second even though
its total-token throughput was the lowest. Request throughput therefore reflects
both server efficiency and the amount of work in each request.

### Long prompts dominate time to first token

RAG and summarization had similar realized input distributions and nearly equal
median TTFT. Both were much slower than chat before the first output token. This
is consistent with prefill cost and contention increasing with prompt length.

### Decode length dominates end-to-end latency

RAG and summarization began with similar input lengths, but summarization's mean
maximum output length was more than twice as large. Its median E2E latency was
therefore almost twice RAG's even though their median TTFT values were close.

### Aggregate token throughput needs a denominator label

RAG achieved 1,900 total tok/s—the largest number in that column—because the
measurement counts its many input tokens. Its output throughput was actually the
lowest at 84.87 tok/s. Capacity reports should keep input, output, and total-token
throughput separate rather than presenting an unlabeled “tokens/s” value.

### Long contexts increase memory pressure

Peak VRAM increased from 18,060 MiB for chat to 21,024 MiB for both long-context
profiles. With model weights held constant, the increase is consistent with
larger active KV-cache and workspace requirements.

## Artifacts

- Raw canonical run: [`runs/E003_20260924_051307_29213e50/`](../runs/E003_20260924_051307_29213e50/)
- Latency comparison: [`plots/e003/latency_by_profile.png`](plots/e003/latency_by_profile.png)
- Throughput comparison: [`plots/e003/throughput_by_profile.png`](plots/e003/throughput_by_profile.png)
- Workload shapes: [`plots/e003/workload_shape_by_profile.png`](plots/e003/workload_shape_by_profile.png)
- GPU comparison: [`plots/e003/gpu_by_profile.png`](plots/e003/gpu_by_profile.png)

The raw run includes every profile's configuration, environment, per-request
measurements, realized workload plan, telemetry, summary, and generated plots,
plus the source commit, package inventory, and full `nvidia-smi` capture.

## Limitations

1. This is one run on one RTX 3090; it does not estimate run-to-run variance.
2. Profiles are synthetic token distributions, not production traffic traces.
3. Input and output caps are sampled independently rather than jointly.
4. Traffic is closed-loop at concurrency 4, so this is not an arrival-rate or
   saturation experiment.
5. Prefix caching and chunked prefill were intentionally disabled.
6. Maximum output tokens are caps. Chat produced 3,974 actual output tokens
   against 4,128 requested maximum tokens because some generations stopped
   early; the other two profiles reached their requested caps.
7. Conclusions apply to the recorded model, precision, backend, server options,
   hardware, and software versions.

## Conclusion

E003 establishes that application-shaped token mixes produce substantially
different latency, throughput, and memory behavior even when concurrency and
infrastructure are identical. It also provides the workload machinery and clean
baseline needed for E004 to study open-loop offered load, saturation, and SLO
attainment without conflating those effects with request shape.
