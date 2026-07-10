import os
import sys
import time
import ctypes
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

M_THETA = 181

# ── LOG FILTER: cattura offload, sopprime lo spam CUDA-graph ─────────────────
_CAPTURED_OFFLOAD = []           # righe utili sul caricamento GPU
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
        return                                  # spam per-token → silenziato
    if any(k in low for k in _KEEP):
        line = s.strip()
        if line:
            _CAPTURED_OFFLOAD.append(line)      # prova GPU → bufferizzata

# Registra il filtro PRIMA di creare il modello (tenere il ref vivo!)
llama_cpp.llama_log_set(_log_filter, ctypes.c_void_p(0))


def gpu_vram_used_mib():
    """VRAM usata (MiB) via nvidia-smi. None se non disponibile."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:
        return None


def preflight_gpu_check(requested_layers: int) -> None:
    print("\n" + "=" * 60)
    print(" 🔬 PREFLIGHT — BUILD & GPU DIAGNOSTICS")
    print("=" * 60)
    gpu_ok = False
    try:
        gpu_ok = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as e:
        print(f" [!] gpu_offload check non disponibile: {e}")
    print(f"  llama-cpp-python : {getattr(llama_cpp,'__version__','?')}")
    print(f"  GPU offload build: {'✅ ATTIVO (CUDA)' if gpu_ok else '❌ ASSENTE (CPU-only)'}")
    vram = gpu_vram_used_mib()
    print(f"  VRAM usata ora   : {vram} MiB" if vram is not None else "  VRAM: nvidia-smi non trovato")

    if requested_layers > 0 and not gpu_ok:
        print("\n" + "!" * 60)
        print(" ⚠️  GPU layers richiesti ma build CPU-only → ignorati.")
        print("     Reinstalla con la wheel/sorgente CUDA (vedi README).")
        print("!" * 60)
        if input("\n  Continuo in CPU? (s/N): ").strip().lower() not in ("s","si","sì","y","yes"):
            sys.exit(1)
    print("=" * 60)


class SolitonHybridEngine:
    def __init__(self, model_path, gpu_layers, threads, mode_name):
        print(f"\n[~] INITIALIZING ENGINE IN MODE: {mode_name}")
        if not os.path.exists(model_path):
            print(f"[-] ERROR: Model non trovato -> {model_path}")
            sys.exit(1)

        self.gpu_layers = gpu_layers
        vram_before = gpu_vram_used_mib()
        try:
            print(f"[*] Allocating Tensors ({gpu_layers} GPU Layers | {threads} CPU Threads)...")
            t0 = time.time()
            # verbose=False: nessuno spam. La prova GPU arriva dal log-filter + VRAM.
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                n_ctx=CTX_WINDOW,
                flash_attn=True,
                use_mmap=True,
                verbose=True,          # emette i log → intercettati dal filtro
            )
            load_time = time.time() - t0
        except Exception as e:
            print(f"[-] Hardware Error: {e}")
            sys.exit(1)

        # ── Prova GPU: righe di offload catturate + delta VRAM ──
        print(f"[+] Engine loaded in {load_time:.2f}s.")
        if _CAPTURED_OFFLOAD:
            print("\n[GPU OFFLOAD — log llama.cpp]")
            for ln in _CAPTURED_OFFLOAD:
                if any(k in ln.lower() for k in ("offload", "layers to gpu", "layer(s) to gpu")):
                    print(f"   • {ln}")
        vram_after = gpu_vram_used_mib()
        if vram_before is not None and vram_after is not None:
            print(f"\n[VRAM] prima: {vram_before} MiB  →  dopo: {vram_after} MiB  "
                  f"(Δ +{vram_after - vram_before} MiB)")
            if gpu_layers > 0 and (vram_after - vram_before) < 50:
                print("   ⚠️  VRAM quasi invariata: i layer NON sono saliti in GPU!")
            elif gpu_layers > 0:
                print("   ✅ Il modello è in VRAM.")
        print("-" * 60)

    def _prune(self, prompt_text):
        raw = self.llm.tokenize(prompt_text.encode("utf-8"))
        if not ENABLE_PRUNING:
            return raw
        pr = [t for t in raw if (t < 256 or t > 100000) or ((t ^ M_THETA) % 255) > 60]
        ratio = 1.0 - (len(pr) / max(1, len(raw)))
        if ratio > 0:
            print(f"[~] Topological pruning (SPERIMENTALE): -{ratio*100:.2f}% token")
        return pr

    def run_benchmark(self, prompt, max_tokens=512):
        tokens = self._prune(prompt)
        n_prompt = len(tokens)

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
                print(piece, end="", flush=True)
                n_out += 1

        t_end = time.time()
        print("\n\n" + "-" * 60)

        prefill_t = (first_t - t0) if first_t else 0.0
        decode_t  = (t_end - first_t) if first_t else 0.0
        total_t   = t_end - t0
        prefill_tps = (n_prompt / prefill_t) if prefill_t > 0 else 0
        decode_tps  = (n_out / decode_t) if decode_t > 0 else 0

        print("⚡ [TELEMETRY]")
        print(f"  - Prompt tokens (prefill) : {n_prompt}")
        print(f"  - Generated tokens        : {n_out}")
        print(f"  - Time to First Token     : {prefill_t:.2f} s")
        print(f"  - Prefill throughput      : {prefill_tps:.1f} tok/s")
        print(f"  - Decode throughput       : {decode_tps:.1f} tok/s   ← key metric")
        print(f"  - Total latency           : {total_t:.2f} s")
        v = gpu_vram_used_mib()
        if v is not None:
            print(f"  - VRAM in uso             : {v} MiB")
        if self.gpu_layers == 0:
            print("  - NOTE: modalità CPU (0 GPU layers).")

    def interactive_chat(self):
        print("\n" + "=" * 60)
        print(" 🌌 ENGINE READY — INTERACTIVE MODE ('exit' per uscire)")
        print("=" * 60)
        print("\n[SYSTEM]: Cold-start (warm-up)...")
        cold = ("<|im_start|>system\nYou are an advanced AI.<|im_end|>\n"
                "<|im_start|>user\nIn exactly 20 words, describe the relationship "
                "between entropy and time.<|im_end|>\n<|im_start|>assistant\n")
        self.run_benchmark(cold, max_tokens=40)
        print("\n[!] Modello caldo in cache. Puoi chattare.")
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


if __name__ == "__main__":
    physical = psutil.cpu_count(logical=False)
    logical = psutil.cpu_count(logical=True)

    print("\n" + "=" * 60)
    print(" 🚀 HYBRID OFFLOAD ENGINE v3 - HARDWARE SELECTION")
    print("=" * 60)
    print(f" Detected CPU: {physical} Physical / {logical} Logical")
    print("-" * 60)
    print(" [1] Pure CPU (Physical Cores - AVX)")
    print(" [2] Pure CPU (Logical Cores - SMT Penalty Test)")
    print(f" [3] Hybrid Offload ({ENV_GPU_LAYERS} GPU Layers + CPU)")
    print("=" * 60)
    print(" TIP: SHOW_LLAMA_LOGS=1 per vedere TUTTI i log grezzi di llama.cpp")
    print("=" * 60 + "\n")

    choice = input("Select execution mode (1, 2, or 3): ").strip()
    if choice == "1":
        mode, layers, threads = "PURE CPU (AVX)", 0, physical
    elif choice == "2":
        mode, layers, threads = "PURE CPU (SMT TEST)", 0, logical
    else:
        if choice != "3":
            print("[-] Scelta non valida. Default: Hybrid (3).")
        mode, layers, threads = "HYBRID OFFLOAD", ENV_GPU_LAYERS, physical

    preflight_gpu_check(layers)
    engine = SolitonHybridEngine(MODEL_PATH, gpu_layers=layers,
                                 threads=threads, mode_name=mode)
    engine.interactive_chat()
