"""Resize, normalize, and tokenize pre-sampled video arrays for Qwen."""

import numpy as np
import torch
from transformers import AutoProcessor

from .common import path, read_args, selection


def main():
    config, _ = read_args()
    processor = AutoProcessor.from_pretrained(
        path(config, "model_dir"), local_files_only=True, use_fast=False,
        min_pixels=config["min_pixels"], max_pixels=config["max_pixels"],
    )
    messages = [{"role": "user", "content": [
        {"type": "video"}, {"type": "text", "text": config["prompt"]},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    output = path(config, "data_dir", "preprocessed")
    output.mkdir(parents=True, exist_ok=True)
    for video in selection(config):
        video_id = video["video_id"]
        frames = np.load(path(config, "data_dir", "sampled", f"{video_id}.npy"))
        # These arrays are already sampled. No video paths or vision-reader helpers.
        inputs = processor(
            text=[text], videos=[frames],
            fps=len(frames) / video["duration_seconds"], return_tensors="pt",
        )
        temporal_patch = processor.image_processor.temporal_patch_size
        if int(inputs["video_grid_thw"][0, 0]) * temporal_patch != len(frames):
            raise ValueError("Processor changed the sampled frame count")
        torch.save(dict(inputs), output / f"{video_id}.pt")
        print(f"Preprocessed {video_id}: {inputs['input_ids'].shape[1]} input tokens", flush=True)


if __name__ == "__main__":
    main()
