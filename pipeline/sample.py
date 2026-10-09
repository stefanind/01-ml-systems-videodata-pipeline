"""Sample frames using their presentation timestamps, without decoding again."""

import numpy as np

from .common import path, read_args, selection


def sample_indices(timestamps, duration, count):
    """Choose the frame at or just before each evenly spaced target time."""
    # Bin midpoints cover the video portion without favoring its endpoints.
    targets = (np.arange(count) + 0.5) * duration / count
    indices = np.searchsorted(timestamps, targets, side="right") - 1
    return np.clip(indices, 0, len(timestamps) - 1)


def sample_video(video, config):
    """Save sampled RGB frames and timestamps for one video; return the frames."""
    video_id = video["video_id"]
    count = config["num_frames"]
    if count <= 0 or count % 2:
        raise ValueError("num_frames must be a positive even number for Qwen")
    output = path(config, "data_dir", "sampled")
    output.mkdir(parents=True, exist_ok=True)
    # Memory mapping lets us select a few frames without loading the full video.
    frames = np.load(path(config, "data_dir", "decoded", f"{video_id}.npy"), mmap_mode="r")
    timestamps = np.load(path(config, "data_dir", "decoded", f"{video_id}.timestamps.npy"))
    indices = sample_indices(timestamps, video["duration_seconds"], count)
    sampled_frames = frames[indices]
    np.save(output / f"{video_id}.npy", sampled_frames)
    np.save(output / f"{video_id}.timestamps.npy", timestamps[indices])
    print(f"Sampled {video_id}: {count} frames", flush=True)
    return sampled_frames


def main():
    config, _ = read_args()
    # Validate before reading selection.json; direct calls also check their inputs.
    count = config["num_frames"]
    if count <= 0 or count % 2:
        raise ValueError("num_frames must be a positive even number for Qwen")
    # V0 finishes sampling the complete selection before preprocessing starts.
    for video in selection(config):
        sample_video(video, config)


if __name__ == "__main__":
    main()
