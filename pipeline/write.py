"""Combine the selected videos' intermediate captions into one JSONL file."""

import json

from .common import path, read_args, selection


def main():
    config, _ = read_args()
    destination = path(config, "output_dir", "captions.jsonl")
    destination.parent.mkdir(parents=True, exist_ok=True)
    videos = selection(config)
    with destination.open("w", encoding="utf-8") as output:
        for video in videos:
            caption = path(config, "data_dir", "captions", f"{video['video_id']}.txt").read_text(encoding="utf-8").strip()
            if not caption:
                raise ValueError(f"Empty caption for {video['video_id']}")
            record = {key: value for key, value in video.items() if key != "metadata"}
            record["caption"] = caption
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(videos)} captions to {destination}", flush=True)


if __name__ == "__main__":
    main()
