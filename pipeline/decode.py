"""Read Parquet video bytes and fully decode the selected video portions on CPU."""

import io
import json
from urllib.parse import urlparse

import av
import numpy as np
import pyarrow.parquet as pq

from .common import path, read_args, save_json, shard_urls


def decode_video(video_bytes, max_seconds):
    """Return RGB frames, relative timestamps, and the decoded portion's duration."""
    frames, timestamps = [], []
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        # One decoding thread keeps V0's CPU work simple and predictable.
        stream.codec_context.thread_count = 1
        fps = float(stream.average_rate or stream.guessed_rate or 30)
        first_time = None
        for index, frame in enumerate(container.decode(stream)):
            # Presentation timestamps handle variable frame rates; FPS is a fallback.
            timestamp = frame.time if frame.time is not None else index / fps
            if first_time is None:
                first_time = timestamp
            # Start at zero and stop at the configured portion, e.g. ten seconds.
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
    # Keep every decoded frame in V0. RGB arrays are much larger than MP4 bytes;
    # stacking creates one [frames, height, width, channels] array for the next stage.
    return np.stack(frames), np.asarray(timestamps), duration


def iter_videos(config):
    """Yield (video record, encoded bytes) in shard/row order, up to max_videos."""
    if config["max_videos"] <= 0 or (config["max_seconds"] is not None and config["max_seconds"] <= 0):
        raise ValueError("max_videos and max_seconds must be positive; use null for full video duration")
    video_index = 0
    for url in shard_urls(config):
        shard_name = urlparse(url).path.rsplit("/", 1)[-1]
        shard = pq.ParquetFile(path(config, "data_dir", "shards", shard_name))
        # Read one row at a time; the whole Parquet shard need not fit in RAM.
        for row_index, batch in enumerate(shard.iter_batches(batch_size=1, columns=["mp4", "json"], use_threads=False)):
            row = batch.to_pylist()[0]
            metadata = row["json"]
            if not isinstance(metadata, dict):
                metadata = json.loads(metadata)
            video_bytes = row["mp4"]
            if isinstance(video_bytes, dict):
                video_bytes = video_bytes["bytes"]
            # Assign IDs from input order before doing any work on the video.
            # These IDs will stay stable when later versions use parallel workers.
            video = {
                "video_id": f"video-{video_index:05d}",
                "source_video_id": metadata.get("original_json_filename"),
                "shard": shard_name,
                "row_index": row_index,
                "metadata": metadata,
            }
            yield video, video_bytes
            video_index += 1
            if video_index == config["max_videos"]:
                return


def decode_record(video, video_bytes, config):
    """Save one video's decoded arrays and return its record with duration added."""
    output = path(config, "data_dir", "decoded")
    output.mkdir(parents=True, exist_ok=True)
    video_id = video["video_id"]
    frames, timestamps, duration = decode_video(video_bytes, config["max_seconds"])
    # The separate sampling script reads these two files without decoding again.
    np.save(output / f"{video_id}.npy", frames)
    np.save(output / f"{video_id}.timestamps.npy", timestamps)
    record = {
        "video_id": video_id,
        "source_video_id": video["source_video_id"],
        "shard": video["shard"],
        "row_index": video["row_index"],
        "duration_seconds": duration,
        "metadata": video["metadata"],
    }
    print(f"Decoded {video_id}: {len(frames)} frames, {duration:.2f} seconds", flush=True)
    return record


def main():
    config, _ = read_args()
    videos = []
    for video, video_bytes in iter_videos(config):
        videos.append(decode_record(video, video_bytes, config))
    if not videos:
        raise ValueError("No videos found in the selected shards")
    # Publish the selection only after all requested videos have decoded.
    save_json(path(config, "data_dir", "selection.json"), videos)


if __name__ == "__main__":
    main()
