"""Save environment and input details before the baseline's measured work."""

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import sys

import torch

from .common import ROOT, path, read_args, save_json, shard_urls


def git_info():
    """Record code provenance when Git is available; do not guess a revision."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        changes = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}

    return {"commit": commit, "dirty": bool(changes.strip())}


def gpu_info(index):
    """Read the monitor's selected GPU through NVML without running CUDA kernels."""
    try:
        import pynvml
    except ImportError:
        return None

    try:
        pynvml.nvmlInit()
        try:
            gpu = pynvml.nvmlDeviceGetHandleByIndex(index)
            return {
                "index": index,
                "name": pynvml.nvmlDeviceGetName(gpu),
                "total_memory_mib": pynvml.nvmlDeviceGetMemoryInfo(gpu).total / 2**20,
                "driver_version": pynvml.nvmlSystemGetDriverVersion(),
            }
        finally:
            pynvml.nvmlShutdown()
    except pynvml.NVMLError:
        return None


def record_experiment(config, command, working_directory, reuse_shards=False):
    """Write the two small recording files into this run's output directory."""
    output = path(config, "output_dir")
    output.mkdir(parents=True, exist_ok=True)

    # Read only these settings, rather than copying tokens or the entire environment.
    settings = [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "TORCH_CUDNN_V8_API_DISABLED",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]

    record = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "working_directory": working_directory,
        "input_mode": "local_shards" if reuse_shards else "network_download",
        "image": config.get("runpod_image") or None,
        "gpu": gpu_info(config["gpu_index"]),
        "software": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "pytorch": str(torch.__version__),
            "pytorch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "git": git_info(),
        "config_file": str(config["_config_file"]),
        # _config_file is an internal Path object; keep the original user settings.
        "config": {
            key: value
            for key, value in config.items()
            if key != "_config_file"
        },
        "shard_urls": shard_urls(config),
        "model_id": config["model_id"],
        "runtime_environment": {
            key: os.environ[key]
            for key in settings
            if key in os.environ
        },
    }

    # Include image-provided packages: the default venv reuses the image's PyTorch.
    packages = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
    )
    (output / "requirements.freeze.txt").write_text(packages, encoding="utf-8")
    save_json(output / "experiment.json", record)
    print(f"Recorded experiment in {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--command",
        required=True,
        help="Original runner command as a JSON list",
    )
    parser.add_argument("--working-directory", required=True)
    parser.add_argument("--reuse-shards", action="store_true")
    config, args = read_args(parser)

    record_experiment(
        config,
        json.loads(args.command),
        args.working_directory,
        args.reuse_shards,
    )


if __name__ == "__main__":
    main()
