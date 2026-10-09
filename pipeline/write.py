"""Combine the selected videos' intermediate captions into one JSONL file."""

import json

from .common import path, read_args, selection


def write_caption(video, caption, output):
    """Write one video record to an open JSONL file and return that record."""
    caption = caption.strip()
    if not caption:
        raise ValueError(f"Empty caption for {video['video_id']}")
    # Retain IDs and source locations, leaving the large dataset metadata out.
    record = {key: value for key, value in video.items() if key != "metadata"}
    record["caption"] = caption
    # JSONL stores one independent JSON object per line; preserve non-English text.
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def main():
    config, _ = read_args()
    destination = path(config, "output_dir", "captions.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    videos = selection(config)
    # Open the final file once and write videos in the original selection order.
    with destination.open("w", encoding="utf-8") as output:
        for video in videos:
            caption = path(config, "data_dir", "captions", f"{video['video_id']}.txt").read_text(encoding="utf-8").strip()
            write_caption(video, caption, output)
    print(f"Wrote {len(videos)} captions to {destination}", flush=True)


if __name__ == "__main__":
    main()
