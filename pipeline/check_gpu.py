"""Test Qwen's first convolution without loading the caption model's weights."""

import argparse
import faulthandler
import json
import time

import torch

from .common import path, read_args, selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cudnn", action="store_true", help="Test PyTorch's native convolution instead of cuDNN")
    config, args = read_args(parser)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])
    vision = json.loads(path(config, "model_dir", "config.json").read_text(encoding="utf-8"))["vision_config"]
    video_id = selection(config)[0]["video_id"]
    inputs = torch.load(path(config, "data_dir", "preprocessed", f"{video_id}.pt"), weights_only=True)
    kernel = (vision["temporal_patch_size"], vision["patch_size"], vision["patch_size"])
    channels = vision.get("in_channels", vision.get("in_chans", 3))
    pixels = inputs["pixel_values_videos"].view(-1, channels, *kernel)
    torch.backends.cudnn.enabled = not args.no_cudnn
    print(f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; cuDNN {torch.backends.cudnn.version()}", flush=True)
    print(f"Testing {video_id}: input={tuple(pixels.shape)}, device={device}, dtype={dtype}, cuDNN enabled={torch.backends.cudnn.enabled}", flush=True)
    faulthandler.dump_traceback_later(20, exit=True)
    try:
        print("Preparing convolution and transferring tensors...", flush=True)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        conv = torch.nn.Conv3d(channels, vision["hidden_size"], kernel_size=kernel, stride=kernel, bias=False)
        conv = conv.to(device=device, dtype=dtype).eval()
        pixels = pixels.to(device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print("Transfers completed. Starting Conv3d...", flush=True)
        start = time.perf_counter()
        with torch.inference_mode():
            output = conv(pixels)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(f"Conv3d passed in {time.perf_counter() - start:.3f} seconds: output={tuple(output.shape)}", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
