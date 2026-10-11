"""Test Qwen's first convolution without loading the caption model's weights."""

import argparse
import faulthandler
import json
import time

import torch

from .common import path, read_args, selection


def main():
    """Isolate the first vision convolution using one video's prepared tensors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cudnn",
        action="store_true",
        help="Test PyTorch's native convolution instead of cuDNN",
    )
    config, args = read_args(parser)
    device = torch.device(config["device"])
    dtype = getattr(torch, config["dtype"])

    model_config_file = path(config, "model_dir", "config.json")
    model_config = json.loads(model_config_file.read_text(encoding="utf-8"))
    vision = model_config["vision_config"]
    video_id = selection(config)[0]["video_id"]
    input_file = path(config, "data_dir", "preprocessed", f"{video_id}.pt")
    inputs = torch.load(input_file, weights_only=True)

    # Qwen's processor stores flattened video patches. Rebuild the Conv3d input
    # shape using the model's channel count and temporal/spatial patch sizes.
    kernel = (
        vision["temporal_patch_size"],
        vision["patch_size"],
        vision["patch_size"],
    )
    channels = vision.get("in_channels", vision.get("in_chans", 3))
    pixels = inputs["pixel_values_videos"].view(-1, channels, *kernel)
    torch.backends.cudnn.enabled = not args.no_cudnn

    print(
        f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}; "
        f"cuDNN {torch.backends.cudnn.version()}",
        flush=True,
    )
    print(
        f"Testing {video_id}: input={tuple(pixels.shape)}, "
        f"device={device}, dtype={dtype}, "
        f"cuDNN enabled={torch.backends.cudnn.enabled}",
        flush=True,
    )

    # A stuck GPU call may ignore Ctrl+C; print its stack and exit after 20 s.
    faulthandler.dump_traceback_later(20, exit=True)
    try:
        print("Preparing convolution and transferring tensors...", flush=True)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        # Random weights are enough to test the operation; this is a GPU/runtime
        # diagnostic, so it does not download weights or assess caption quality.
        conv = torch.nn.Conv3d(
            channels,
            vision["hidden_size"],
            kernel_size=kernel,
            stride=kernel,
            bias=False,
        )
        conv = conv.to(device=device, dtype=dtype).eval()
        pixels = pixels.to(device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        print("Transfers completed. Starting Conv3d...", flush=True)
        start = time.perf_counter()
        with torch.inference_mode():
            output = conv(pixels)
        if device.type == "cuda":
            # CUDA runs asynchronously; synchronize before reporting completion.
            torch.cuda.synchronize(device)
        print(
            f"Conv3d passed in {time.perf_counter() - start:.3f} seconds: "
            f"output={tuple(output.shape)}",
            flush=True,
        )
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
