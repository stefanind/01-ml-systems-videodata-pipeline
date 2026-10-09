#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Reuse the template's CUDA PyTorch rather than installing another large build.
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python - <<'PY'
import torch
from huggingface_hub import snapshot_download
from pipeline.common import path, read_args

if not torch.cuda.is_available():
    raise SystemExit("Choose a RunPod PyTorch GPU template; CUDA PyTorch is required.")
print("GPU:", torch.cuda.get_device_name(0))
config, _ = read_args()
snapshot_download(
    config["model_id"], local_dir=path(config, "model_dir"),
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja", "*.model"],
)
print("Qwen is cached. Setup complete.")
PY

echo "Next: source .venv/bin/activate && python -m pipeline.run_baseline"
