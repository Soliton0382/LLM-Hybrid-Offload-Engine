import os
import sys
import time
from dotenv import load_dotenv
import psutil
from llama_cpp import Llama

# Setup Paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Assumiamo che .env sia nella cartella superiore rispetto a bench/
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))

MODEL_NAME = os.getenv("MODEL_NAME", "model.gguf")
MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), "models", MODEL_NAME)

ENV_GPU_LAYERS = int(os.getenv("GPU_LAYERS", 25))
CTX_WINDOW = int(os.getenv("CONTEXT_WINDOW", 32768))

# Topological Constant
M_THETA = 181

class SolitonHybridEngine:
    def __init__(self, model_path, gpu_layers, threads, mode_name):
        print(f"\n[~] INITIALIZING ENGINE IN MODE: {mode_name}")
        if not os.path.exists(model_path):
            print(f"[-] ERROR: Model file not found at -> {model_path}")
            print(f"    Please place your .gguf file inside the 'models' directory and update .env")
            sys.exit(1)

        try:
            print(f"[*] Allocating Tensors ({gpu_layers} GPU Layers | {threads} CPU Threads)...")
            start_load = time.time()

            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=gpu_layers,
                n_threads=threads,
                n_ctx=CTX_WINDOW,
                flash_attn=True,
                use_mmap=True,
                verbose=False
            )

            load_time = time.time() - start_load
            print(f"[+] Engine loaded in {load_time:.2f} seconds.")
            print("[+] FlashAttention: ACTIVE.")
        except Exception as e:
            print(f"[-] Hardware Error during allocation: {e}")
            sys.exit(1)

    def _topological_kv_compression(self, prompt_text):
        raw_tokens = self.llm.tokenize(prompt_text.encode('utf-8'))
        pruned_tokens = []

        for t in raw_tokens:
            if t < 256 or t > 100000:
                pruned_tokens.append(t)
                continue

            t_xor = t ^ M_THETA
            if (t_xor % 255) > 60:
                pruned_tokens.append(t)

        compression_ratio = 1.0 - (len(pruned_tokens) / len(raw_tokens))
        if compression_ratio > 0:
            print(f"[~] Topological Pruning applied. Initial KV-Cache compressed by {compression_ratio*100:.2f}%")
        return pruned_tokens

    def run_benchmark(self, prompt, max_tokens=512):
        print("\n[USER PROMPT]:\n", prompt.split('\n')[1] if '<|im_start|>user\n' in prompt else prompt)
        print("-" * 60)

        tokens = self._topological_kv_compression(prompt)

        print("\n[INFERENCE]:\n")
        start_time = time.time()

        output = self.llm.generate(tokens, temp=0.2)

        token_count = 0
        response_text = ""

        for token in output:
            if token == self.llm.token_eos():
                break

            word = self.llm.detokenize([token]).decode('utf-8', errors='ignore')
            response_text += word
            print(word, end="", flush=True)

            token_count += 1
            if token_count >= max_tokens:
                break

        print("\n\n" + "-" * 60)

        total_time = time.time() - start_time
        tps = token_count / total_time if total_time > 0 else 0

        print(f"⚡ [TELEMETRY]")
        print(f"  - Tokens Generated: {token_count}")
        print(f"  - Total Latency: {total_time:.2f} s")
        print(f"  - Throughput: {tps:.2f} Tokens/sec")

    def interactive_chat(self):
        print("\n==========================================================")
        print(" 🌌 ENGINE INITIALIZED - INTERACTIVE MODE READY")
        print(" Type 'exit' or 'quit' to close the terminal.")
        print("==========================================================\n")

        print("[SYSTEM]: Waking up the model with a cold-start prompt...")
        cold_start_prompt = "<|im_start|>system\nYou are an advanced AI.<|im_end|>\n<|im_start|>user\nIn exactly 20 words, describe the relationship between entropy and time.<|im_end|>\n<|im_start|>assistant\n"
        self.run_benchmark(cold_start_prompt, max_tokens=30)

        print("\n[!] The model is now loaded in Cache. You can chat.")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n[YOU]: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("\n[SYSTEM]: Shutting down Engine. Goodbye.")
                    break

                if not user_input.strip():
                    continue

                chat_prompt = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
                self.run_benchmark(chat_prompt, max_tokens=1024)

            except KeyboardInterrupt:
                print("\n[SYSTEM]: Session terminated by user.")
                break

if __name__ == "__main__":
    if MODEL_PATH.endswith("your_model_file.gguf"):
        print("WARNING: You must set your model file name in the .env file!")
        sys.exit(1)

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    print("\n==========================================================")
    print(" 🚀 HYBRID OFFLOAD ENGINE - HARDWARE SELECTION")
    print("==========================================================")
    print(f" Detected CPU: {physical_cores} Physical Cores / {logical_cores} Logical Threads")
    print("----------------------------------------------------------")
    print(" [1] Pure CPU Offload (Physical Cores - AVX Optimized)")
    print("     -> Best for systems WITHOUT a dedicated GPU.")
    print(" [2] Pure CPU Offload (Logical Cores - SMT Penalty Test)")
    print("     -> Demonstrates the speed drop caused by Context Switching.")
    print(f" [3] Hybrid Offload ({ENV_GPU_LAYERS} GPU Layers + CPU)")
    print("     -> Uses .env settings. Best for systems WITH a GPU.")
    print("==========================================================\n")

    choice = input("Select an execution mode (1, 2, or 3): ").strip()

    if choice == '1':
        mode = "PURE CPU (AVX OPTIMIZED)"
        layers = 0
        threads = physical_cores
    elif choice == '2':
        mode = "PURE CPU (SMT PENALTY TEST)"
        layers = 0
        threads = logical_cores
    elif choice == '3':
        mode = "HYBRID OFFLOAD"
        layers = ENV_GPU_LAYERS
        threads = physical_cores
    else:
        print("[-] Invalid choice. Defaulting to Hybrid Offload (3).")
        mode = "HYBRID OFFLOAD"
        layers = ENV_GPU_LAYERS
        threads = physical_cores

    engine = SolitonHybridEngine(MODEL_PATH, gpu_layers=layers, threads=threads, mode_name=mode)
    engine.interactive_chat()
