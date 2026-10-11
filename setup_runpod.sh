#!/usr/bin/env bash
set -euo pipefail

# Run from the repository even if this script was called from another directory.
cd "$(dirname "$0")"

# Default: reuse the template's CUDA PyTorch. --cu126: compare an isolated build.
case "${1:-}" in
    "")
        venv_dir=".venv"
        python -m venv --system-site-packages "$venv_dir"
        ;;
    --cu126)
        venv_dir=".venv-cu126"
        python -m venv "$venv_dir"
        ;;
    *)
        echo "Usage: bash setup_runpod.sh [--cu126]" >&2
        exit 2
        ;;
esac

source "$venv_dir/bin/activate"
python -m pip install --upgrade pip

# The default environment keeps the PyTorch build supplied by the Pod image.
if [[ "$venv_dir" == ".venv-cu126" ]]; then
    python -m pip install \
        torch==2.7.1 \
        torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cu126
fi
python -m pip install -r requirements.txt

# RunPod templates may enable hf_transfer without installing its optional package.
HF_HUB_ENABLE_HF_TRANSFER=0 python - <<'PY'
import torch
from huggingface_hub import snapshot_download

from pipeline.common import path, read_args

if not torch.cuda.is_available():
    raise SystemExit("Choose a RunPod PyTorch GPU template; CUDA PyTorch is required.")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "PyTorch:", torch.__version__,
    "CUDA:", torch.version.cuda,
    "cuDNN:", torch.backends.cudnn.version(),
)

# Cache Qwen during setup so downloading the model is outside the benchmark.
config, _ = read_args()
snapshot_download(
    config["model_id"],
    local_dir=path(config, "model_dir"),
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model"],
)
print("Qwen is cached. Setup complete.")
PY

echo "Next: source $venv_dir/bin/activate && python -m pipeline.run_baseline"
