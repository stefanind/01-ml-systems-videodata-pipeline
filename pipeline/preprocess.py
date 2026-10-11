"""Resize, normalize, and tokenize pre-sampled video arrays for Qwen."""

import numpy as np
import torch
from transformers import AutoProcessor

from .common import path, read_args, selection


def load_processor(config):
    """Load Qwen's cached tokenizer and image processor once for the stage."""
    # The processor prepares CPU inputs; the caption stage loads the GPU model.
    return AutoProcessor.from_pretrained(
        path(config, "model_dir"),
        local_files_only=True,
        use_fast=False,
        min_pixels=config["min_pixels"],
        max_pixels=config["max_pixels"],
    )


def build_prompt(config, processor):
    """Format the shared instruction with Qwen's video and chat markers."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": config["prompt"]},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def preprocess_video(video, config, processor, text=None):
    """Prepare and save one video's CPU tensors, returning the input dictionary."""
    video_id = video["video_id"]
    frames = np.load(path(config, "data_dir", "sampled", f"{video_id}.npy"))
    if text is None:
        text = build_prompt(config, processor)

    # Arrays are already sampled: Qwen resizes/normalizes them and tokenizes the
    # prompt without reading or decoding a video again. FPS describes these samples.
    inputs = processor(
        text=[text],
        videos=[frames],
        fps=len(frames) / video["duration_seconds"],
        return_tensors="pt",
    )

    # Qwen groups frames into temporal patches. Check that it kept our frame count.
    temporal_patch = processor.image_processor.temporal_patch_size
    processed_frame_count = int(inputs["video_grid_thw"][0, 0]) * temporal_patch
    if processed_frame_count != len(frames):
        raise ValueError("Processor changed the sampled frame count")

    output = path(config, "data_dir", "preprocessed")
    output.mkdir(parents=True, exist_ok=True)
    inputs = dict(inputs)
    torch.save(inputs, output / f"{video_id}.pt")
    print(
        f"Preprocessed {video_id}: {inputs['input_ids'].shape[1]} input tokens",
        flush=True,
    )
    return inputs


def main():
    config, _ = read_args()
    # Reuse the same processor and formatted prompt for every selected video.
    processor = load_processor(config)
    text = build_prompt(config, processor)
    for video in selection(config):
        preprocess_video(video, config, processor, text)


if __name__ == "__main__":
    main()
