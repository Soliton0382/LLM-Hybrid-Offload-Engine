# Hybrid-Offload-Engine 🌌

Welcome to the **Hybrid Offload Engine**.
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

## Features
- **Hardware-Aware Offload:** Statically splits model layers to prevent PCIe spill-over.
- **AVX-512 Thread Pinning:** Limits threading strictly to physical CPU cores.
- **Interactive Mode & Live Telemetry:** Outputs raw Token/s throughput and latency during chat.

## Benchmarks
*(Tests conducted on 16 × AMD Ryzen 7 7700X, 64GB DDR5, RTX 5060Ti 16GB. Model: Nemotron-Cascade-30B-Q8_0)*

| Hardware Config | Threads (Physical) | Offload Strategy | Avg. Tokens/sec |
| :--- | :--- | :--- | :--- |
| **CPU Only (Pure RAM)** | 8 | 0 GPU Layers | `11.80 T/s` |
| **CPU Only (SMT On)** | 16 (Logical) | 0 GPU Layers | `8.24 T/s` |
| **Hybrid Offload** | 8 | 25 GPU Layers | `21.80 T/s` |

## Usage
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Place your `.gguf` model in the `models/` directory.
4. Copy `template.env` to `.env` and configure your parameters.
5. Run the interactive benchmark: `python3 bench/hybrid_offload_benchmark.py`

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
