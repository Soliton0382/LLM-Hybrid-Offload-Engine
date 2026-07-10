# LLM-Hybrid-Offload-Engine 🌌

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10--3.14-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4%2B-76B900.svg)
![GPU](https://img.shields.io/badge/GPU-NVIDIA-orange.svg)

Welcome to the **LLM Hybrid Offload Engine**.
This Python tool provides a plug-and-play solution for offloading Large Language Model (LLM) inference between CPU and GPU, structurally addressing the Memory Bandwidth Wall inherent in DDR5 systems.

## The Problem
Running a 30B+ parameter model entirely on system RAM typically yields `< 2` tokens/s due to memory bus bottlenecks (~90 GB/s on DDR5). The CPU spends the majority of its time idling, waiting for weights.

## The Solution
By utilizing hardware-aware threading and strategic layer-splitting, this engine enables local execution of heavy models at fluid speeds. It ensures that the CPU uses strict physical-core binding (bypassing SMT context-switching delays) and leverages FlashAttention to manage extreme context windows.

### Flow Comparison
**Standard Offload Flow:**
`RAM (DDR5) -> CPU (Bottleneck) -> Wait -> GPU Compute -> Output`

**Hybrid Flow (Optimized):**
`Input -> Hardware-Aware Thread Pinning -> Simultaneous AVX-512 (CPU) & CUDA (GPU) Execution -> High-Speed Output`

<p align="center">
  <img src="img/flow_diagram.svg" alt="Hybrid Offload Flow Diagram" width="90%">
</p>

## Features
- **Hardware-Aware Offload:** Statically splits model layers to prevent PCIe spill-over.
- **AVX-512 Thread Pinning:** Limits threading strictly to physical CPU cores.
- **GPU Preflight Diagnostics:** Verifies the CUDA build *before* loading — no silent CPU fallback.
- **Live VRAM Probe:** Measures VRAM delta via `nvidia-smi` so you can *see* the model land on the GPU.
- **Clean Telemetry:** Separates **Prefill** and **Decode** throughput for honest, comparable numbers.
- **Interactive Mode:** Chat with the model with live Token/s output.

## Requirements

| Component | Requirement | Notes |
| :--- | :--- | :--- |
| **NVIDIA Driver** | Any recent (CUDA ≥ 12.4) | `nvidia-smi` must work |
| **CUDA Toolkit** | 12.4 (`nvcc`) | Only needed if building from source |
| **Host Compiler** | `gcc-13` / `g++-13` | CUDA 12.4 does **not** support GCC ≥ 14 |
| **Python** | 3.10 – 3.14 | Prebuilt wheels: 3.10–3.13 · Python 3.14 → source build |

> ⚠️ **Prebuilt wheels are not available for Python 3.14.** On 3.14 the installer compiles from source (handled automatically by `install.sh`).

> ℹ️ **No NVIDIA GPU?** `install.sh` completes fine (it falls back to a CPU-only build),
> but `hybrid_offload_benchmark.py` will **stop on purpose** at the GPU preflight check —
> it never runs a "fake" CPU benchmark labeled as *hybrid*. This is intentional:
> we never lie about the numbers. The **CPU Only** rows in the Benchmarks table
> show exactly what to expect on a GPU-less machine.

## Installation (Plug & Play)

The installer auto-detects your CUDA version and installs the correct build.
It follows a cascade: **prebuilt wheel → source build (GPU) → CPU fallback**.

```bash
# 1. Clone the repository
git clone https://github.com/Soliton0382/LLM-Hybrid-Offload-Engine.git

# 2. Enter the folder
cd LLM-Hybrid-Offload-Engine

# 3. Run the installer (creates .venv and installs the correct build)
chmod +x install.sh && ./install.sh

# 4. Activate the environment
source .venv/bin/activate
Then configure and run:
Place your .gguf model inside the models/ directory.
Copy template.env to .env and set MODEL_NAME, GPU_LAYERS, CONTEXT_WINDOW.
Run the interactive benchmark:
python3 bench/hybrid_offload_benchmark.py


Verify the GPU build at any time:python -c "import llama_cpp; print('GPU:', llama_cpp.llama_supports_gpu_offload())"
# → GPU: True
At startup you will see a GPU proof block confirming the model is on the GPU:[GPU OFFLOAD — log llama.cpp]
   • offloaded 25/49 layers to GPU

[VRAM] before: 320 MiB  →  after: 9800 MiB   (Δ +9480 MiB)
   ✅ Model is in VRAM.
Environment FlagsConfigure these in your .env file:VariableDefaultPurposeMODEL_NAME(nemotron)GGUF filename located in models/GPU_LAYERS25Number of layers offloaded to the GPU (Hybrid mode)CONTEXT_WINDOW32768KV-cache context sizeSHOW_LLAMA_LOGS0Set to 1 to print all raw llama.cpp logs (debug)ENABLE_TOPO_PRUNING0Experimental prompt pruning (keep off for an honest benchmark)Benchmarks
Test rig: AMD Ryzen 7 7700X (8C/16T) · 64 GB DDR5 · RTX 5060 Ti 16 GB
Model: Nemotron-Cascade-30B-Q8_0
Hardware ConfigThreadsOffload StrategyDecode Tokens/secCPU Only (Pure RAM)8 (Physical)0 GPU Layers11.80 T/sCPU Only (SMT On)16 (Logical)0 GPU Layers8.24 T/sHybrid Offload8 (Physical)25 GPU Layers21.80 T/sReading the results: the SMT row exposes the context-switching penalty.
Using more logical threads is actually slower, because the physical cores end up
contending for the same execution units. Hybrid offload nearly doubles the
CPU-only throughput.Troubleshooting<details open>
<summary><b>❌ <code>ninja: build stopped: subcommand failed</code> during install</b></summary><br>Your system GCC is too new for CUDA 12.4 (which supports GCC ≤ 13).
Force the host compiler:CC=gcc-13 CXX=g++-13 \
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13" \
pip install --force-reinstall --no-cache-dir llama-cpp-python
If gcc-13 is missing, install it first:sudo apt install gcc-13 g++-13
</details><details>
<summary><b>🐌 The model runs but VRAM stays empty / tokens/s are low</b></summary><br>Your llama-cpp-python is a CPU-only build. Verify with:python -c "import llama_cpp; print(llama_cpp.llama_supports_gpu_offload())"
If it prints False, reinstall using the CUDA method above.</details><details>
<summary><b>🌊 Output is flooded with <code>CUDA Graph reused</code> and I can't see the text</b></summary><br>That spam is normal GPU activity, but it buries the generated output.
The bundled benchmark filters it automatically.
To inspect the raw logs on purpose, set:SHOW_LLAMA_LOGS=1 python3 bench/hybrid_offload_benchmark.py
</details><details>
<summary><b>🤔 Driver shows CUDA 13.x but <code>nvcc</code> says 12.4</b></summary><br>This is expected and not a problem:
nvidia-smi reports the maximum CUDA version supported by the driver.
nvcc reports the installed toolkit version.
The driver is backward-compatible, so the cu124 build works correctly.</details>LicenseThis project is licensed under the MIT License — see the LICENSE file for details.
