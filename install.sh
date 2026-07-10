#!/usr/bin/env bash
# =============================================================================
#  install.sh · Hybrid-Offload Engine — plug&play installer
#  Cascade strategy (stops at first success):
#     1) Precompiled CUDA WHEEL    (fast; for Python cp310-cp313)
#     2) BUILD from CUDA source    (auto gcc-13/14 host-compiler → resolves GCC15)
#     3) CPU FALLBACK              (no GPU/CUDA present)
#  Usage:  chmod +x install.sh && ./install.sh
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
WHL_BASE="https://abetlen.github.io/llama-cpp-python/whl"

echo "=============================================================="
echo " 🚀 Hybrid-Offload — Plug&play Installer"
echo "=============================================================="

# ── 1. venv ──────────────────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[*] Creating virtualenv in $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools >/dev/null

PYTAG="cp$(python -c 'import sys;print(f"{sys.version_info.major}{sys.version_info.minor}")')"
echo "[*] Python wheel tag: $PYTAG"

# ── 2. base dependencies ───────────────────────────────────────────────────────
echo "[*] Installing base dependencies..."
pip install --upgrade python-dotenv psutil >/dev/null

# ── 3. detect CUDA (from driver or nvcc) ────────────────────────────────────
detect_cuda_tag() {
  local ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
  fi
  if [ -z "$ver" ] && command -v nvcc >/dev/null 2>&1; then
    ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
  fi
  [ -z "$ver" ] && { echo ""; return; }
  local M m; M=$(echo "$ver" | cut -d. -f1); m=$(echo "$ver" | cut -d. -f2)
  if   [ "$M" -ge 13 ]; then echo "cu124"       # new driver → highest wheel avail.
  elif [ "$M" -eq 12 ] && [ "$m" -ge 4 ]; then echo "cu124"
  elif [ "$M" -eq 12 ] && [ "$m" -eq 3 ]; then echo "cu123"
  elif [ "$M" -eq 12 ] && [ "$m" -eq 2 ]; then echo "cu122"
  elif [ "$M" -eq 12 ]; then echo "cu121"
  else echo ""; fi
}

# Chooses a host-compiler compatible with nvcc (resolves case of too-new GCC).
pick_cuda_host_cxx() {
  for v in 13 14 12; do
    if command -v "g++-$v" >/dev/null 2>&1; then
      echo "/usr/bin/g++-$v"; return
    fi
  done
  echo ""   # none found → system g++ will be used (might fail)
}

verify_gpu() {
  python - <<'PY'
import sys
try:
    import llama_cpp
    print(bool(llama_cpp.llama_supports_gpu_offload()))
except Exception:
    print("BROKEN")
PY
}

install_wheel() {
  local tag="$1"
  echo "[+] Trying precompiled CUDA WHEEL: $tag ($PYTAG)"
  pip install "llama-cpp-python" \
      --extra-index-url "$WHL_BASE/$tag" \
      --upgrade --force-reinstall --no-cache-dir 2>/dev/null
}

build_source_cuda() {
  local hostcxx; hostcxx="$(pick_cuda_host_cxx)"
  echo "[+] BUILD from SOURCE with CUDA."
  if [ -n "$hostcxx" ]; then
    local cxxbin="${hostcxx##*/}"; local ccbin="${cxxbin/g++/gcc}"
    echo "    nvcc host-compiler: $hostcxx  (resolves recent GCC incompatibility)"
    CC="/usr/bin/$ccbin" CXX="$hostcxx" \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=$hostcxx" \
      pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
  else
    echo "    (no g++-13/14 found: using system compiler)"
    CMAKE_ARGS="-DGGML_CUDA=on" \
      pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
  fi
}

install_cpu() {
  echo "[!] Installing CPU build (no GPU or CUDA not available)."
  pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
}

# ── 4. installation cascade ──────────────────────────────────────────────
CUDA_TAG="$(detect_cuda_tag)"

if [ -z "$CUDA_TAG" ]; then
  install_cpu
else
  echo "[i] GPU/CUDA detected → target wheel: $CUDA_TAG"
  OK="false"
  # 4.1 precompiled wheel
  if install_wheel "$CUDA_TAG" && [ "$(verify_gpu)" = "True" ]; then
    OK="true"; echo "[+] ✅ CUDA Wheel OK."
  else
    echo "[!] Wheel not available for $PYTAG/$CUDA_TAG (typical on Python 3.14)."
    # 4.2 source build with pinned gcc
    if build_source_cuda && [ "$(verify_gpu)" = "True" ]; then
      OK="true"; echo "[+] ✅ CUDA source build OK."
    fi
  fi
  # 4.3 CPU fallback
  if [ "$OK" != "true" ]; then
    echo "[!] CUDA failed → CPU fallback."
    install_cpu
  fi
fi

# ── 5. final verification ───────────────────────────────────────────────────────
echo "--------------------------------------------------------------"
python - <<'PY'
import llama_cpp
gpu=False
try: gpu=bool(llama_cpp.llama_supports_gpu_offload())
except Exception as e: print(f"[!] check: {e}")
print(f"  llama-cpp-python : {getattr(llama_cpp,'__version__','?')}")
print(f"  GPU OFFLOAD      : {'✅ ACTIVE (CUDA)' if gpu else '⚠️ CPU-only'}")
PY
echo "=============================================================="
echo " ✅ Done.  Activate with:  source $VENV_DIR/bin/activate"
echo " ▶  Start:   python bench/hybrid_offload_benchmark.py"
echo "=============================================================="
