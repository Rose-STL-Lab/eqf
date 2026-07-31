#!/usr/bin/env bash
set -e
set -o pipefail

# Example usage: 
# ./setup_env.sh 
# ./setup_env.sh --cuda13
# ./setup_env.sh --cuda12.4

CUDA_VERSION=""
for arg in "$@"; do
  case "$arg" in
    --cuda13) CUDA_VERSION="13.0" ;;
    --cuda12.4) CUDA_VERSION="12.4" ;;
  esac
done

# Default and validate CUDA version flag
if [[ -z "$CUDA_VERSION" ]]; then
  CUDA_VERSION="13.0"
fi
case "$CUDA_VERSION" in
  13.0|12.4) ;;
  *)
    echo "ERROR: Unsupported --cuda-version '$CUDA_VERSION'. Allowed: 13.0 or 12.4. Or don't specify to default to settings for 13" >&2
    exit 1
    ;;
esac

echo "Using CUDA version flag: $CUDA_VERSION"

ENV_NAME="eqf"


# ---- robust conda initialization (for non-interactive scripts) ----
init_conda() {
  # 0) If conda isn't even on PATH, we can't proceed
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: 'conda' not found on PATH." >&2
    echo "  - If you're on an HPC, load your conda module (e.g., 'module load Mambaforge')." >&2
    echo "  - Otherwise install Miniconda/Mambaforge and reopen your shell." >&2
    exit 1
  fi

  # 1) Preferred: initialize via conda's shell hook (most robust)
  # This avoids hardcoding ~/miniconda3/... paths.
  if eval "$(conda shell.bash hook 2>/dev/null)"; then
    return 0
  fi

  # 2) Fallback: try common conda.sh locations
  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )

  for f in "${candidates[@]}"; do
    if [[ -f "$f" ]]; then
      # shellcheck source=/dev/null
      source "$f"
      return 0
    fi
  done

  # 3) Last resort: explain how to fix
  echo "ERROR: Found 'conda' but couldn't initialize it for 'conda activate'." >&2
  echo "Tried: 'eval \$(conda shell.bash hook)' and common conda.sh paths." >&2
  echo "Fix options:" >&2
  echo "  - Run: conda init bash  (then restart your shell)" >&2
  echo "  - Or edit this script to source your conda.sh explicitly." >&2
  exit 1
}
# -------------------------------------------------------------------


# 1. Create the conda environment
echo "Creating conda environment '$ENV_NAME'..."

# source ~/miniconda3/etc/profile.d/conda.sh
init_conda
conda create -n $ENV_NAME python=3.12 -y
conda activate $ENV_NAME

echo "Conda environment '$ENV_NAME' created and activated."

# 2. Install PyTorch with CUDA support

pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

echo "Installed PyTorch with CUDA support."

# 3. Run pip installs in order (flash attention and others require torch)
pip install -r requirements.txt
echo "Installed base requirements."

# 4. Install PyTorch with CUDA nvcc for Mamba and Conv1d
# conda install -y -c "nvidia/label/cuda-12.9.1" cuda-toolkit=12.9.1 cuda-nvcc
conda install -y -c nvidia cuda-toolkit=12.8 cuda-nvcc=12.8


# 5. Install Flash Attention (requires torch to be installed first)
if [[ "$CUDA_VERSION" == "13.0" ]]; then
  # Do this if cuda version 13
  pip install --no-build-isolation flash_attn==2.7.4.post1
elif [[ "$CUDA_VERSION" == "12.4" ]]; then
  # Do this if cuda version 12.4
  pip install --no-build-isolation flash-attn==2.8.3
else
  echo "ERROR: Unsupported CUDA version for FlashAttention: $CUDA_VERSION" >&2
  exit 1
fi
echo "Installed flash attention requirements."

# 6. Verify PyTorch CUDA and flash attention
python - <<'PY'
import torch
import flash_attn
from flash_attn import flash_attn_func

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot access CUDA. Check the NVIDIA driver and GPU availability."
    )

device = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"GPU: {device} (compute capability {capability[0]}.{capability[1]})")
print(f"FlashAttention: {getattr(flash_attn, '__version__', 'unknown')}")

# Shape: (batch, sequence length, attention heads, head dimension)
q = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
k = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
v = torch.randn(1, 128, 4, 64, device="cuda", dtype=torch.float16, requires_grad=True)
output = flash_attn_func(q, k, v)
output.sum().backward()
torch.cuda.synchronize()

print(f"FlashAttention CUDA test passed (output shape: {tuple(output.shape)}).")
PY
echo "Verified PyTorch CUDA and FlashAttention."


echo "All done! The environment '$ENV_NAME' is ready."
