# Metric Definitions

This document will define the canonical latency, throughput, and telemetry metrics collected by `inference_os`.

## Planned Metrics

Detailed mathematical definitions and measurement procedures for the following metrics are pending E000 design:

* **Time to First Token (TTFT)**: Latency from request submission until arrival of the first generated token. *(Pending E000 design)*
* **End-to-End Latency (E2E)**: Total duration from request submission until complete output generation. *(Pending E000 design)*
* **Time Per Output Token (TPOT)**: Average time per generated token during the decode phase. *(Pending E000 design)*
* **Throughput**: Request throughput (requests/sec) and token throughput (tokens/sec). *(Pending E000 design)*
* **GPU Utilization**: Percentage of GPU compute capacity utilized during benchmark runs. *(Pending E000 design)*
* **GPU Memory**: VRAM allocation and consumption patterns over time. *(Pending E000 design)*
