# GPU Smoke Test Run Output

* **Date**: 2026-08-21
* **Hardware**: 1× NVIDIA GeForce RTX 3090 (24 GB VRAM) on Vast.ai
* **Driver Version**: 580.173.02 | **CUDA Version**: 13.0
* **Model**: `Qwen/Qwen2.5-7B-Instruct` (bfloat16)
* **Serving Backend**: vLLM 0.27.1 on `http://localhost:8001`
* **Branch**: `gpu_smoke_test`

---

## 1. Model Verification (`/v1/models`)

```json
{"object":"list","data":[{"id":"Qwen/Qwen2.5-7B-Instruct","object":"model","created":1787317934,"owned_by":"vllm","root":"Qwen/Qwen2.5-7B-Instruct","parent":null,"max_model_len":8192,"permission":[{"id":"modelperm-8b768eebc191d3a2","object":"model_permission","created":1787317934,"allow_create_engine":false,"allow_sampling":true,"allow_logprobs":true,"allow_search_indices":false,"allow_view":true,"allow_fine_tuning":false,"organization":"*","group":null,"is_blocking":false}]}]}
```

---

## 2. Client Execution & Setup

```bash
# 1. Navigate to workspace & clone repository
cd /workspace
git clone https://github.com/chidharth-b/inference_os.git
cd inference_os
git checkout gpu_smoke_test

# 2. Install uv & sync dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync

# 3. Run the GPU Smoke Test on port 8001
uv run python scripts/smoke_test_vllm.py \
    --base-url "http://localhost:8001" \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --prompt "Explain why the sky appears blue in two concise sentences."
```

---

## 3. Raw Console Output

```text
Cloning into 'inference_os'...
remote: Enumerating objects: 57, done.
remote: Counting objects: 100% (57/57), done.
remote: Compressing objects: 100% (36/36), done.
remote: Total 57 (delta 1), reused 57 (delta 1), pack-reused 0 (from 0)
Receiving objects: 100% (57/57), 26.64 KiB | 26.64 MiB/s, done.
Resolving deltas: 100% (1/1), done.
Branch 'gpu_smoke_test' set up to track remote branch 'gpu_smoke_test' from 'origin'.
Switched to a new branch 'gpu_smoke_test'
downloading uv 0.12.5 x86_64-unknown-linux-gnu
installing to /root/.local/bin
  uv
  uvx
everything's installed!

To add $HOME/.local/bin to your PATH, either restart your shell or run:

    source $HOME/.local/bin/env (sh, bash, zsh)
    source $HOME/.local/bin/env.fish (fish)
WARN: The following commands are shadowed by other commands in your PATH: uv uvx
warning: `VIRTUAL_ENV=/venv/main` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
Using CPython 3.11.15
Creating virtual environment at: .venv
Resolved 15 packages in 0.49ms
      Built inference-os @ file:///workspace/inference_os                                                                                                              
Prepared 6 packages in 500ms
Installed 14 packages in 22ms
 + anyio==4.14.2
 + certifi==2026.7.22
 + h11==0.16.0
 + httpcore==1.0.9
 + httpx==0.28.1
 + idna==3.19
 + inference-os==0.1.0 (from file:///workspace/inference_os)
 + iniconfig==2.3.0
 + packaging==26.3
 + pluggy==1.6.0
 + pygments==2.20.0
 + pytest==9.1.1
 + ruff==0.16.2
 + typing-extensions==4.16.0
warning: `VIRTUAL_ENV=/venv/main` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
Connecting to vLLM server at: http://localhost:8001
Model: Qwen/Qwen2.5-7B-Instruct
Prompt: Explain why the sky appears blue in two concise sentences.
--------------------------------------------------
Streaming output:  The sky appears blue because the Earth's atmosphere scatters shorter (blue) wavelengths of sunlight more than longer (red) wavelengths. This scattering, known as Rayleigh scattering, makes the blue light more visible from the ground.
--------------------------------------------------
Measurement Results:
  Success: True
  TTFT: 218.72 ms (0.2187 s)
  E2E Latency: 1091.29 ms (1.0913 s)
```
