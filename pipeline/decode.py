"""Read Parquet video bytes and fully decode the selected video portions on CPU."""

import io
import json
from urllib.parse import urlparse

import av
import numpy as np
import pyarrow.parquet as pq

from .common import path, read_args, save_json, shard_urls


def decode_video(video_bytes, max_seconds):
    frames, timestamps = [], []
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        first_time = None
        for index, frame in enumerate(container.decode(stream)):
            timestamp = frame.time if frame.time is not None else index / fps
            if first_time is None:
                first_time = timestamp
            timestamp -= first_time
            if max_seconds is not None and timestamp >= max_seconds:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(timestamp)
    if not frames:
        raise ValueError("Video has no frames in the requested portion")
    duration = timestamps[-1] + 1 / fps
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    return np.stack(frames), np.asarray(timestamps), duration


def main():
    config, _ = read_args()
    if config["max_videos"] <= 0 or (config["max_seconds"] is not None and config["max_seconds"] <= 0):
        raise ValueError("max_videos and max_seconds must be positive; use null for full video duration")
    output = path(config, "data_dir", "decoded")
    output.mkdir(parents=True, exist_ok=True)
    videos = []
    for url in shard_urls(config):
        shard_name = urlparse(url).path.rsplit("/", 1)[-1]
        shard = pq.ParquetFile(path(config, "data_dir", "shards", shard_name))
        for row_index, batch in enumerate(shard.iter_batches(batch_size=1, columns=["mp4", "json"], use_threads=False)):
            row = batch.to_pylist()[0]
            metadata = row["json"]
            if not isinstance(metadata, dict):
                metadata = json.loads(metadata)
            video_bytes = row["mp4"]
            if isinstance(video_bytes, dict):
                video_bytes = video_bytes["bytes"]
            video_id = f"video-{len(videos):05d}"
            frames, timestamps, duration = decode_video(video_bytes, config["max_seconds"])
            np.save(output / f"{video_id}.npy", frames)
            np.save(output / f"{video_id}.timestamps.npy", timestamps)
            videos.append({
                "video_id": video_id,
                "source_video_id": metadata.get("original_json_filename"),
                "shard": shard_name,
                "row_index": row_index,
                "duration_seconds": duration,
                "metadata": metadata,
            })
            print(f"Decoded {video_id}: {len(frames)} frames, {duration:.2f} seconds", flush=True)
            del frames
            if len(videos) == config["max_videos"]:
                break
        if len(videos) == config["max_videos"]:
            break
    if not videos:
        raise ValueError("No videos found in the selected shards")
    save_json(path(config, "data_dir", "selection.json"), videos)


if __name__ == "__main__":
    main()
