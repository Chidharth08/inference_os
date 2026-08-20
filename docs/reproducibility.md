# Environment & Reproducibility Metadata

This document outlines the environment metadata captured alongside each benchmark execution to ensure experimental reproducibility.

## Planned Reproducibility Metadata

Experiments will capture metadata including:

* **Version Control**: Git commit hash, branch state, and repository status.
* **Model Configuration**: Model name, revision, precision (e.g. BF16), and parameter settings.
* **Serving Backend**: Backend name (e.g. vLLM), software version, and runtime serving configuration flags.
* **Software Stack**: Python version, PyTorch version, CUDA version, and NVIDIA driver version.
* **Hardware Profile**: GPU model, memory capacity, host system specs, provider (e.g., Vast.ai vs GCP), and topology.
* **Execution Parameters**: Workload configurations, random seeds, warm-up policies, and timestamps.

## Hardware Environment Isolation Policy

* **Hardware Tracking**: The current benchmark execution target is 1× NVIDIA RTX 3090 (24 GB VRAM) hosted ephemerally on Vast.ai.
* **Isolation Guarantee**: Results obtained on different GPU models or hosting environments (e.g., RTX 3090 on Vast.ai vs L4 on GCP) must be recorded as separate hardware environments in result metadata and never directly combined or compared without explicit environment isolation.

## Ephemeral GPU Operational Workflow

To ensure cost efficiency and reproducible artifact tracking:
1. **Local Development**: All code development, unit testing, architecture design, documentation, and data analysis occur locally.
2. **Ephemeral Sessions**: Vast.ai GPU instances are started only for focused benchmark runs (1–3 hours).
3. **Commit & Push**: All code changes, raw benchmark data, and generated reports must be committed and pushed to GitHub prior to ending a session.
4. **Instance Destruction**: GPU instances must be destroyed (not left paused) at the end of each session to prevent ongoing stopped storage fees (~$0.32/day).
