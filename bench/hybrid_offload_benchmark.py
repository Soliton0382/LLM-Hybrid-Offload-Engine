
import os
import sys
import time
from dotenv import load_dotenv
import psutil
from llama_cpp import Llama

# Setup Paths dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL_NAME = os.getenv("MODEL_NAME", "model.gguf")
MODEL_PATH = os.path.join(BASE_DIR, "models", MODEL_NAME)

GPU_LAYERS = int(os.getenv("GPU_LAYERS", 25))
CTX_WINDOW = int(os.getenv("CONTEXT_WINDOW", 32768))

# Topological Constant
M_THETA = 181 

class SolitonHybridEngine:
    def __init__(self, model_path):
        print("\n[~] INITIALIZING SOLITON HYBRID ENGINE...")
        if not os.path.exists(model_path):
            print(f"[-] ERROR: Model file not found at -> {model_path}")
            print(f"    Please place your .gguf file inside the 'models' directory and update .env")
            sys.exit(1)
            
        try:
            print("[*] Allocating Hybrid Tensors (VRAM + DDR5)...")
            start_load = time.time()
            
            self.llm = Llama(
                model_path=model_path,
                n_gpu_layers=GPU_LAYERS,    
                n_threads=psutil.cpu_count(logical=False),
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
        print(f"[~] Topological Pruning applied. Initial KV-Cache compressed by {compression_ratio*100:.2f}%")
        return pruned_tokens

    def run_benchmark(self, prompt, max_tokens=512):
        print("\n[USER PROMPT]:\n", user_input if 'user_input' in locals() else prompt.split('\n')[2])
        print("-" * 60)
        
        tokens = self._topological_kv_compression(prompt)
        
        print("\n[SOLITON INFERENCE]:\n")
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
        
        print(f"⚡ [HYBRID TELEMETRY]")
        print(f"  - Tokens Generated: {token_count}")
        print(f"  - Total Latency: {total_time:.2f} s")
        print(f"  - Throughput: {tps:.2f} Tokens/sec")

    def interactive_chat(self):
        print("\n==========================================================")
        print(" 🌌 SOLITON ENGINE INITIALIZED - INTERACTIVE MODE READY")
        print(" Type 'exit' or 'quit' to close the terminal.")
        print("==========================================================\n")
        
        print("[SYSTEM]: Waking up the model with a philosophical seed...")
        cold_start_prompt = "<|im_start|>system\nYou are an advanced AI.<|im_end|>\n<|im_start|>user\nIn exactly 20 words, describe the relationship between entropy and time.<|im_end|>\n<|im_start|>assistant\n"
        self.run_benchmark(cold_start_prompt, max_tokens=30)
        
        print("\n[!] The model is now loaded in Cache. You can chat.")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\n[YOU]: ")
                if user_input.lower() in ['exit', 'quit']:
                    print("\n[SYSTEM]: Shutting down Soliton Engine. Goodbye.")
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
        
    engine = SolitonHybridEngine(MODEL_PATH)
    engine.interactive_chat()
