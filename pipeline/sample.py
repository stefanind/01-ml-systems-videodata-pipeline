"""Sample frames using their presentation timestamps, without decoding again."""

import numpy as np

from .common import path, read_args, selection


def sample_indices(timestamps, duration, count):
    targets = (np.arange(count) + 0.5) * duration / count
    indices = np.searchsorted(timestamps, targets, side="right") - 1
    return np.clip(indices, 0, len(timestamps) - 1)


def main():
    config, _ = read_args()
    count = config["num_frames"]
    if count <= 0 or count % 2:
        raise ValueError("num_frames must be a positive even number for Qwen")
    output = path(config, "data_dir", "sampled")
    output.mkdir(parents=True, exist_ok=True)
    for video in selection(config):
        video_id = video["video_id"]
        frames = np.load(path(config, "data_dir", "decoded", f"{video_id}.npy"), mmap_mode="r")
        timestamps = np.load(path(config, "data_dir", "decoded", f"{video_id}.timestamps.npy"))
        indices = sample_indices(timestamps, video["duration_seconds"], count)
        np.save(output / f"{video_id}.npy", frames[indices])
        np.save(output / f"{video_id}.timestamps.npy", timestamps[indices])
        print(f"Sampled {video_id}: {count} frames", flush=True)


if __name__ == "__main__":
    main()
