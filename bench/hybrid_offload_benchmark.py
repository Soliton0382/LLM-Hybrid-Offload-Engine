#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hybrid_offload_benchmark.py - Hybrid CPU/GPU Offload Engine
===========================================================
Runs a 30B-class GGUF model locally, streaming, and measures the real
inference telemetry (prefill / decode throughput, TTFT, VRAM).

It is the harness behind the headline throughput numbers:
    Pure CPU (physical cores)   ~ baseline
    Pure CPU (logical / SMT)    ~ slower (SMT penalty on compute-bound work)
    Hybrid offload (N GPU layers + CPU)   ~ fastest

TWO WAYS TO RUN
---------------
    # Interactive chat (pick a mode from the menu):
    python bench/hybrid_offload_benchmark.py

    # Automated 3-mode comparison (non-interactive, prints a table):
    python bench/hybrid_offload_benchmark.py --auto
    python bench/hybrid_offload_benchmark.py --auto --tokens 128

Environment (inherited from ../.env):
    MODEL_NAME, GPU_LAYERS (25), CONTEXT_WINDOW (32768)
    ENABLE_TOPO_PRUNING (0)  -> experimental token filter, OFF by default
    SHOW_LLAMA_LOGS (0)      -> 1 to see ALL raw llama.cpp logs
"""
import os
import sys
import time
import ctypes
import argparse
import subprocess
from dotenv import load_dotenv
import psutil
import llama_cpp
from llama_cpp import Llama

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))

MODEL_NAME = os.getenv("MODEL_NAME", "model.gguf")
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", MODEL_NAME)

ENV_GPU_LAYERS = int(os.getenv("GPU_LAYERS", 25))
CTX_WINDOW     = int(os.getenv("CONTEXT_WINDOW", 32768))
ENABLE_PRUNING = os.getenv("ENABLE_TOPO_PRUNING", "0") in ("1", "true", "True", "yes")
SHOW_LLAMA_LOGS = os.getenv("SHOW_LLAMA_LOGS", "0") in ("1", "true", "True", "yes")

# EXPERIMENTAL constant for the optional token filter (OFF by default). Kept as a
# transparent negative control -- it does NOT improve quality and is disabled.
M_THETA = 181

# -- LOG FILTER: captures offload, suppresses CUDA-graph spam -----------------
_CAPTURED_OFFLOAD = []           # useful lines on GPU loading
_KEEP = ("offload", "assigned to device", "layer(s) to gpu", "layers to gpu",
         "buffer size", "using device", "cuda0", "kv self size", "flash")
_DROP = ("cuda graph", "graph warmup", "graph_compute", "prefix-match")

@llama_cpp.llama_log_callback
def _log_filter(level, text, user_data):
    try:
        s = text.decode("utf-8", "ignore") if isinstance(text, (bytes, bytearray)) else str(text)
    except Exception:
        return
    low = s.lower()
    if SHOW_LLAMA_LOGS:
        sys.stderr.write(s)
        return
    if any(d in low for d in _DROP):
        return                                  # per-token spam -> silenced
    if any(k in low for k in _KEEP):
        line = s.strip()
        if line:
            _CAPTURED_OFFLOAD.append(line)      # GPU proof -> buffered

# Register the filter BEFORE creating the model (keep the ref alive!)
llama_cpp.llama_log_set(_log_filter, ctypes.c_void_p(0))


def gpu_vram_used_mib():
    """VRAM used (MiB) via nvidia-smi. None if not available."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def preflight_gpu_check(requested_layers: int, non_interactive: bool = False) -> None:
    print("\n" + "=" * 60)
    print(" [*] PREFLIGHT - BUILD & GPU DIAGNOSTICS")
    print("=" * 60)
    gpu_ok = False
    try:
        gpu_ok = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        print(f" [!] gpu_offload check not available: {e}")
    print(f"  llama-cpp-python : {getattr(llama_cpp,'__version__','?')}")
    print(f"  GPU offload build: {'ACTIVE (CUDA)' if gpu_ok else 'MISSING (CPU-only)'}")
    vram = gpu_vram_used_mib()
    print(f"  VRAM used now    : {vram} MiB" if vram is not None else "  VRAM: nvidia-smi not found")

    if requested_layers > 0 and not gpu_ok:
        print("\n" + "!" * 60)
        print(" [!] GPU layers requested but CPU-only build -> ignored.")
        print("     Reinstall with the CUDA wheel/source (see README).")
        print("!" * 60)
        if not non_interactive:
            if input("\n  Continue on CPU? (y/N): ").strip().lower() not in ("y", "yes"):
                sys.exit(1)
    print("=" * 60)


class SolitonHybridEngine:
    def __init__(self, model_path, gpu_layers, threads, mode_name):
        print(f"\n[~] INITIALIZING ENGINE IN MODE: {mode_name}")
        if not os.path.exists(model_path):
            print(f"[-] ERROR: Model not found -> {model_path}")
            sys.exit(1)

        self.gpu_layers = gpu_layers
        self.mode_name = mode_name
        self.load_time = 0.0
        vram_before = gpu_vram_used_mib()
        try:
            print(f"[*] Allocating Tensors ({gpu_layers} GPU Layers | {threads} CPU Threads)...")
            t0 = time.time()
            # verbose=True: emits logs -> intercepted by the filter. GPU proof
            # comes from the captured offload lines + the VRAM delta below.
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                n_ctx=CTX_WINDOW,
                flash_attn=True,
                use_mmap=True,
                verbose=True,
            )
            self.load_time = time.time() - t0
        except Exception as e:
            print(f"[-] Hardware Error: {e}")
            sys.exit(1)

        # -- GPU Proof: captured offload lines + VRAM delta --
        print(f"[+] Engine loaded in {self.load_time:.2f}s.")
        if _CAPTURED_OFFLOAD:
            print("\n[GPU OFFLOAD - llama.cpp log]")
            for ln in _CAPTURED_OFFLOAD:
                if any(k in ln.lower() for k in ("offload", "layers to gpu", "layer(s) to gpu")):
                    print(f"   - {ln}")
        vram_after = gpu_vram_used_mib()
        if vram_before is not None and vram_after is not None:
            print(f"\n[VRAM] before: {vram_before} MiB  ->  after: {vram_after} MiB  "
                  f"(delta +{vram_after - vram_before} MiB)")
            if gpu_layers > 0 and (vram_after - vram_before) < 50:
                print("   [!] VRAM almost unchanged: layers DID NOT load onto the GPU!")
            elif gpu_layers > 0:
                print("   [OK] The model is in VRAM.")
        print("-" * 60)

    def _prune(self, prompt_text):
        """Optional EXPERIMENTAL token filter (OFF by default via ENABLE_TOPO_PRUNING).
        Kept transparent as a negative control: it does NOT improve output quality.
        The real, validated compression lives in SSCC (see bench/sscc_benchmark.py)."""
        raw = self.llm.tokenize(prompt_text.encode("utf-8"))
        if not ENABLE_PRUNING:
            return raw
        pr = [t for t in raw if (t < 256 or t > 100000) or ((t ^ M_THETA) % 255) > 60]
        ratio = 1.0 - (len(pr) / max(1, len(raw)))
        if ratio > 0:
            print(f"[~] Topological pruning (EXPERIMENTAL): -{ratio*100:.2f}% tokens")
        return pr

    def run_benchmark(self, prompt, max_tokens=512, quiet=False):
        """Run one generation, print telemetry, and RETURN a metrics dict."""
        tokens = self._prune(prompt)
        n_prompt = len(tokens)

        if not quiet:
            print("\n[INFERENCE]:\n")
        t0 = time.time()
        first_t = None
        n_out = 0

        stream = self.llm.create_completion(
            prompt=tokens,
            max_tokens=max_tokens,
            temperature=0.2,
            stream=True,
            stop=["<|im_end|>", "<|endoftext|>"],
        )
        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if piece:
                if first_t is None:
                    first_t = time.time()
                if not quiet:
                    print(piece, end="", flush=True)
                n_out += 1

        t_end = time.time()
        if not quiet:
            print("\n\n" + "-" * 60)

        prefill_t = (first_t - t0) if first_t else 0.0
        decode_t  = (t_end - first_t) if first_t else 0.0
        total_t   = t_end - t0
        prefill_tps = (n_prompt / prefill_t) if prefill_t > 0 else 0
        decode_tps  = (n_out / decode_t) if decode_t > 0 else 0

        if not quiet:
            print("[TELEMETRY]")
            print(f"  - Prompt tokens (prefill) : {n_prompt}")
            print(f"  - Generated tokens        : {n_out}")
            print(f"  - Time to First Token     : {prefill_t:.2f} s")
            print(f"  - Prefill throughput      : {prefill_tps:.1f} tok/s")
            print(f"  - Decode throughput       : {decode_tps:.1f} tok/s   <- key metric")
            print(f"  - Total latency           : {total_t:.2f} s")
            v = gpu_vram_used_mib()
            if v is not None:
                print(f"  - VRAM in use             : {v} MiB")
            if self.gpu_layers == 0:
                print("  - NOTE: CPU mode (0 GPU layers).")

        return {
            "mode": self.mode_name, "gpu_layers": self.gpu_layers,
            "prompt_tokens": n_prompt, "gen_tokens": n_out,
            "ttft_s": prefill_t, "prefill_tps": prefill_tps,
            "decode_tps": decode_tps, "total_s": total_t,
            "vram_mib": gpu_vram_used_mib(), "load_s": self.load_time,
        }

    def interactive_chat(self):
        print("\n" + "=" * 60)
        print(" [*] ENGINE READY - INTERACTIVE MODE ('exit' to quit)")
        print("=" * 60)
        print("\n[SYSTEM]: Cold-start (warm-up)...")
        cold = ("<|im_start|>system\nYou are an advanced AI.<|im_end|>\n"
                "<|im_start|>user\nIn exactly 20 words, describe the relationship "
                "between entropy and time.<|im_end|>\n<|im_start|>assistant\n")
        self.run_benchmark(cold, max_tokens=40)
        print("\n[!] Model warmed up in cache. You can chat.")
        print("-" * 60)

        while True:
            try:
                ui = input("\n[YOU]: ")
                if ui.lower() in ("exit", "quit"):
                    print("\n[SYSTEM]: Shutting down. Goodbye.")
                    break
                if not ui.strip():
                    continue
                p = f"<|im_start|>user\n{ui}<|im_end|>\n<|im_start|>assistant\n"
                self.run_benchmark(p, max_tokens=1024)
            except KeyboardInterrupt:
                print("\n[SYSTEM]: Session terminated.")
                break


# --------------------------------------------------------------------------- #
#  AUTOMATED 3-MODE COMPARISON (non-interactive)                              #
# --------------------------------------------------------------------------- #
_BENCH_PROMPT = ("<|im_start|>system\nYou are an advanced AI.<|im_end|>\n"
                 "<|im_start|>user\nExplain, in a detailed technical paragraph, "
                 "how CPU/GPU hybrid offloading lets a consumer machine run a "
                 "30B-parameter language model.<|im_end|>\n<|im_start|>assistant\n")

def auto_benchmark(max_tokens=128):
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)
    plan = [
        ("PURE CPU (physical cores)", 0, physical),
        ("PURE CPU (logical / SMT)",  0, logical),
        (f"HYBRID OFFLOAD ({ENV_GPU_LAYERS} GPU layers)", ENV_GPU_LAYERS, physical),
    ]
    preflight_gpu_check(ENV_GPU_LAYERS, non_interactive=True)
    results = []
    for name, layers, threads in plan:
        print("\n" + "#" * 60)
        print(f"# RUN: {name}")
        print("#" * 60)
        eng = SolitonHybridEngine(MODEL_PATH, gpu_layers=layers,
                                  threads=threads, mode_name=name)
        # warm-up (not measured), then the measured run
        eng.run_benchmark(_BENCH_PROMPT, max_tokens=24, quiet=True)
        m = eng.run_benchmark(_BENCH_PROMPT, max_tokens=max_tokens, quiet=True)
        results.append(m)
        print(f"[RESULT] {name}: decode {m['decode_tps']:.2f} tok/s "
              f"| prefill {m['prefill_tps']:.1f} tok/s | VRAM {m['vram_mib']} MiB")
        del eng
        time.sleep(2)

    baseline = results[0]["decode_tps"] or 1e-9
    print("\n" + "=" * 72)
    print(" HYBRID OFFLOAD - AUTOMATED BENCHMARK SUMMARY")
    print("=" * 72)
    print(f"{'Mode':<34}{'Decode t/s':>12}{'vs baseline':>14}{'VRAM MiB':>12}")
    print("-" * 72)
    for r in results:
        delta = 100.0 * (r["decode_tps"] - baseline) / baseline
        sign = "+" if delta >= 0 else ""
        print(f"{r['mode']:<34}{r['decode_tps']:>12.2f}"
              f"{sign + format(delta, '.1f') + '%':>14}{str(r['vram_mib']):>12}")
    print("=" * 72)
    print(" Baseline = PURE CPU (physical cores). Higher decode t/s is better.")
    print("=" * 72)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Hybrid CPU/GPU Offload benchmark")
    ap.add_argument("--auto", action="store_true",
                    help="run the automated 3-mode comparison (non-interactive)")
    ap.add_argument("--tokens", type=int, default=128,
                    help="tokens to generate per measured run in --auto mode")
    args = ap.parse_args()

    if args.auto:
        auto_benchmark(max_tokens=args.tokens)
        sys.exit(0)

    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    print("\n" + "=" * 60)
    print(" HYBRID OFFLOAD ENGINE - HARDWARE SELECTION")
    print("=" * 60)
    print(f" Detected CPU: {physical} Physical / {logical} Logical")
    print("-" * 60)
    print(" [1] Pure CPU (Physical Cores - AVX)")
    print(" [2] Pure CPU (Logical Cores - SMT Penalty Test)")
    print(f" [3] Hybrid Offload ({ENV_GPU_LAYERS} GPU Layers + CPU)")
    print("=" * 60)
    print(" TIP: SHOW_LLAMA_LOGS=1 to see ALL raw llama.cpp logs")
    print(" TIP: pass --auto to run the automated 3-mode comparison")
    print("=" * 60 + "\n")

    choice = input("Select execution mode (1, 2, or 3): ").strip()
    if choice == "1":
        mode, layers, threads = "PURE CPU (AVX)", 0, physical
    elif choice == "2":
        mode, layers, threads = "PURE CPU (SMT TEST)", 0, logical
    else:
        if choice != "3":
            print("[-] Invalid choice. Default: Hybrid (3).")
        mode, layers, threads = "HYBRID OFFLOAD", ENV_GPU_LAYERS, physical

    preflight_gpu_check(layers)
    engine = SolitonHybridEngine(MODEL_PATH, gpu_layers=layers,
                                 threads=threads, mode_name=mode)
    engine.interactive_chat()
