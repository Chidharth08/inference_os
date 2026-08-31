# E001 — Prefill and Decode Scaling

## Primary Question

How do prompt length and generated output length independently affect prefill and decode latency?

## Sub-Experiments

- **[E001-A — Input Length Scaling](E001A-input-length/README.md)**: Varies prompt token length ($[128, 512, 2048, 4096]$) holding output length ($128$) and concurrency ($1$) constant to study TTFT and prefill dynamics.
- **E001-B — Output Length Scaling** *(Pending E001-A)*: Varies generated output length ($[16, 32, 64, 128, 256, 512, 1024]$) holding prompt length ($128$) and concurrency ($1$) constant to study decode latency and TPOT.

