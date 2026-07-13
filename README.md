# LLM-Hybrid-Offload-Engine 🌌

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.14-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-76B900.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-orange.svg)

**A plug-and-play, honest benchmarking harness for hybrid CPU+GPU LLM inference.**

This tool wraps `llama-cpp-python` to make it **trivial to run 30B-class models
on consumer hardware** — with reproducible setup, a GPU preflight that never
lies, and clean prefill/decode telemetry.

> **What this is — and what it isn't (read this first).**
> The heavy lifting (layer offload, FlashAttention, mmap) is done by
> **llama.cpp** — we do not reimplement inference. What *this* project adds is a
> **reproducible product layer**: a bulletproof CUDA installer, a no-lies GPU
> preflight, honest telemetry, and a **measured tuning finding** (physical-core
> thread sizing beats SMT — see Benchmarks). If you want raw kernels, use
> llama.cpp directly. If you want a turnkey, honest, on-prem harness, use this.

## The Problem
- Cloud LLMs = recurring cost **and your data leaves the building**.
- Naive on-prem = a 30B model on CPU-only RAM is painfully slow.
- Datacenter GPUs (A100/H100) = thousands of euros.

There is a gap: **cheap, private, reproducible on-prem inference** — with numbers
you can trust.

## The Solution
A thin, honest wrapper that:
- offloads model layers to the GPU (via llama.cpp `n_gpu_layers`),
- **sizes CPU threads to physical cores** to avoid SMT contention,
- uses FlashAttention for large context windows,
- and **proves** the model actually landed on the GPU before running.

## Features
- **Hardware-Aware Offload:** static layer split (llama.cpp) to avoid PCIe spill-over.
- **Physical-Core Thread Sizing:** sets `n_threads` to *physical* core count.
  Empirically this beats using all logical (SMT) threads — see Benchmarks.
  *(Note: this sets the thread count; it is not hard CPU affinity pinning.)*
- **GPU Preflight Diagnostics:** verifies the CUDA build *before* loading —
  no silent CPU fallback, ever.
- **Live VRAM Probe:** measures VRAM delta via `nvidia-smi` so you *see* the
  model land on the GPU.
- **Clean Telemetry:** separates **Prefill** and **Decode** throughput for
  honest, comparable numbers.
- **Interactive Mode:** chat with live tokens/s output.

## Requirements

| Component | Requirement | Notes |
| :--- | :--- | :--- |
| **NVIDIA Driver** | Any recent (CUDA ≥ 12.4) | `nvidia-smi` must work |
| **CUDA Toolkit** | 12.4 (`nvcc`) | Only needed if building from source |
| **Host Compiler** | `gcc-13` / `g++-13` | CUDA 12.4 does **not** support GCC ≥ 14 |
| **Python** | 3.10 – 3.14 | Prebuilt wheels: 3.10–3.13 · 3.14 → source build |

> ⚠️ **Prebuilt wheels are not available for Python 3.14.** On 3.14 the installer
> compiles from source (handled automatically by `install.sh`).

> ℹ️ **No NVIDIA GPU?** `install.sh` completes fine (CPU-only build), but the
> benchmark **stops on purpose** at the GPU preflight — it never runs a "fake"
> CPU benchmark labeled as *hybrid*. We never lie about the numbers.

## Installation (Plug & Play)

```bash
git clone https://github.com/Soliton0382/LLM-Hybrid-Offload-Engine.git
cd LLM-Hybrid-Offload-Engine
chmod +x install.sh && ./install.sh
source .venv/bin/activate
Then:
Place your .gguf model inside models/.
Copy template.env → .env and set MODEL_NAME, GPU_LAYERS, CONTEXT_WINDOW.
Run: python3 bench/hybrid_offload_benchmark.py
Verify the GPU build at any time:python -c "import llama_cpp; print('GPU:', llama_cpp.llama_supports_gpu_offload())"
# → GPU: True
At startup you get a GPU proof block:[GPU OFFLOAD — log llama.cpp]
   • offloaded 25/49 layers to GPU
[VRAM] before: 320 MiB  →  after: 9800 MiB   (Δ +9480 MiB)
   ✅ Model is in VRAM.
Environment FlagsVariableDefaultPurposeMODEL_NAME(nemotron)GGUF filename in models/GPU_LAYERS25Layers offloaded to GPU (Hybrid mode)CONTEXT_WINDOW32768KV-cache context sizeSHOW_LLAMA_LOGS01 = print all raw llama.cpp logs (debug)ENABLE_TOPO_PRUNING0Experimental & unproven input-token pruning. Keep OFF for honest benchmarks.BenchmarksTest rig: AMD Ryzen 7 7700X (8C/16T) · 64 GB DDR5 · RTX 5060 Ti 16 GB
Model: Nemotron-Cascade-2-30B-A3B-Q8_0 — Mixture-of-Experts, ~3B active
params/token (this is why CPU speed is ~11 t/s, not <2 t/s).Hardware ConfigThreadsOffloadDecode t/sCPU Only (Pure RAM)8 (Physical)0 GPU layers11.80CPU Only (SMT On)16 (Logical)0 GPU layers8.24Hybrid Offload8 (Physical)25 GPU layers21.80Reading the results:
Hybrid ≈ +85% over CPU-only → from "sluggish" to "conversational".
The SMT finding (counter-intuitive): using more threads (16 logical) is
~30% slower than 8 physical, because SMT siblings contend for the same
execution units. Most people set threads = nproc and lose performance.
Honest comparison noteThis is not faster than raw llama.cpp — it is llama.cpp underneath, so at
equal settings throughput is equivalent. The value here is reproducibility,
honest telemetry, a turnkey CUDA installer, and the measured SMT tuning finding
— not a new inference kernel.Also: comparisons against Q4 models (e.g. ollama Q4_K_M ≈ 23 t/s) are not
apples-to-apples — Q4 reads ~1.75× fewer bytes/token than Q8. At equal Q8,
CPU-only ollama lands near ~13 t/s. Always compare the same quantization.Roadmap
Real CPU affinity pinning (sched_setaffinity) to make the thread claim
literally true (currently we only size the thread count).
Native KV-cache quantization (type_k/type_v = q8_0/q4_0) for extreme
context windows — llama.cpp supports it; we'll expose & benchmark it.
Context-per-VRAM benchmark (max tokens at fixed VRAM) as a more meaningful
differentiator than raw t/s.
Evaluate whether experimental topological input-pruning yields any measured
gain (currently off, unproven).
LicenseMIT — see LICENSE.
