"""Complete each shard through all six stages before starting the next shard."""

import argparse
from contextlib import contextmanager
import csv
import json
import os
import subprocess
import sys
import time
from urllib.parse import urlparse

from .common import ROOT, path, read_args, save_json, shard_urls
from .monitor import ResourceMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-shards",
        action="store_true",
        help="Use local shards; label this as a local-input run",
    )
    config, args = read_args(parser)
    output = path(config, "output_dir")
    output.mkdir(parents=True, exist_ok=True)

    # A failed attempt must not leave final results from an older successful run.
    for filename in (
        "captions.jsonl",
        "metrics.json",
        "resource_timeline.png",
        "experiment.json",
        "requirements.freeze.txt",
    ):
        (output / filename).unlink(missing_ok=True)

    runner_command = [
        sys.executable,
        "-m",
        "pipeline.run_sharded",
        *sys.argv[1:],
    ]
    recorder = [
        sys.executable,
        "-m",
        "pipeline.record_experiment",
        "--config",
        str(config["_config_file"]),
        "--command",
        json.dumps(runner_command),
        "--working-directory",
        os.getcwd(),
    ]
    if args.reuse_shards:
        recorder.append("--reuse-shards")
    # Recording stays outside the benchmark, just as it does in V0.
    subprocess.run(recorder, cwd=ROOT, check=True)

    monitor = ResourceMonitor(
        output / "resources.csv",
        config["sample_interval_seconds"],
        config["gpu_index"],
    )
    monitor.start()
    metrics = {
        "input_mode": "local_shards" if args.reuse_shards else "network_download",
        "execution_mode": "shard_sequential",
        "status": "failed",
        "download_seconds": 0.0,
        "decode_seconds": 0.0,
        "sampling_seconds": 0.0,
        "preprocess_seconds": 0.0,
        "caption_seconds": 0.0,
        "write_seconds": 0.0,
    }
    all_videos = []
    filenames = set()
    processor = None
    model = None
    pid = os.getpid()
    start = time.perf_counter()
    monitor.origin = start

    try:
        with (output / "stages.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stage_file, (output / "captions.jsonl").open(
            "w", encoding="utf-8"
        ) as captions:
            writer = csv.DictWriter(
                stage_file,
                fieldnames=["stage", "shard", "start_seconds", "end_seconds"],
            )
            writer.writeheader()

            # Each block measures one stage and adds it to that stage's run total.
            # The finally block also records work when a stage fails.
            @contextmanager
            def stage(name, shard_name):
                label = shard_name or "final output"
                print(f"\n--- {label}: {name} ---", flush=True)
                stage_start = time.perf_counter()
                # All work runs here, so RSS covers this persistent runner process.
                monitor.activity = (name, pid)
                try:
                    yield
                finally:
                    stage_end = time.perf_counter()
                    metrics[f"{name}_seconds"] += stage_end - stage_start
                    writer.writerow(
                        {
                            "stage": name,
                            "shard": shard_name,
                            "start_seconds": stage_start - start,
                            "end_seconds": stage_end - start,
                        }
                    )
                    stage_file.flush()
                    monitor.activity = ("", None)

            urls = shard_urls(config)
            for shard_index, url in enumerate(urls):
                shard_name = urlparse(url).path.rsplit("/", 1)[-1]

                # Import stage libraries inside timed work. Later imports reuse
                # Python's cached modules rather than starting new stage processes.
                with stage("download", shard_name):
                    from huggingface_hub import get_token
                    from .download import download_shard

                    if shard_index == 0:
                        token = get_token()
                    if shard_name in filenames:
                        raise ValueError(
                            f"Shard filenames must be unique: {shard_name}"
                        )
                    filenames.add(shard_name)
                    download_shard(url, config, token, args.reuse_shards)

                # Keep downloading every listed shard, even when no more videos
                # are needed. The decoder validates limits before the first record.
                if all_videos and len(all_videos) >= config["max_videos"]:
                    continue

                with stage("decode", shard_name):
                    from .decode import decode_record, iter_shard_videos

                    shard_videos = []
                    for video, video_bytes in iter_shard_videos(
                        config, url, start_index=len(all_videos)
                    ):
                        record = decode_record(video, video_bytes, config)
                        shard_videos.append(record)
                    all_videos.extend(shard_videos)

                if not shard_videos:
                    continue

                with stage("sampling", shard_name):
                    from .sample import sample_video

                    for video in shard_videos:
                        sample_video(video, config)

                with stage("preprocess", shard_name):
                    from .preprocess import (
                        build_prompt,
                        load_processor,
                        preprocess_video,
                    )

                    if processor is None:
                        processor = load_processor(config)
                        text = build_prompt(config, processor)
                    for video in shard_videos:
                        preprocess_video(video, config, processor, text)

                with stage("caption", shard_name):
                    from .caption import caption_video, load_model

                    if model is None:
                        model, tokenizer = load_model(config)
                    for video in shard_videos:
                        print(f"Captioning {video['video_id']}...", flush=True)
                        caption_video(video, config, model, tokenizer)
                        # A completed caption is already saved as an intermediate
                        # text file; final JSONL writing follows the whole shard.
                        if "time_to_first_caption_seconds" not in metrics:
                            metrics["time_to_first_caption_seconds"] = (
                                time.perf_counter() - start
                            )

                with stage("write", shard_name):
                    from .write import write_caption

                    for video in shard_videos:
                        caption_file = path(
                            config,
                            "data_dir",
                            "captions",
                            f"{video['video_id']}.txt",
                        )
                        caption = caption_file.read_text(encoding="utf-8")
                        write_caption(video, caption, captions)
                    # Make this shard's results visible before downloading the next.
                    captions.flush()

            if not all_videos:
                raise ValueError("No videos found in the selected shards")

            # The empty shard name marks final output work for the entire run.
            with stage("write", ""):
                save_json(path(config, "data_dir", "selection.json"), all_videos)
                # Close inside the timed block; the surrounding with also closes
                # this file if an earlier stage fails.
                captions.close()

        end = time.perf_counter()

        # Validate the final IDs after stopping the pipeline clock, as in V0.
        caption_lines = (output / "captions.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        records = [json.loads(line) for line in caption_lines]
        ids = [record["video_id"] for record in records]
        expected = {video["video_id"] for video in all_videos}
        if not ids or len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError(
                "Final caption IDs must exactly match the selected videos, without duplicates"
            )
        metrics.update(
            status="complete",
            total_seconds=end - start,
            num_videos=len(ids),
            throughput=len(ids) / (end - start),
        )
    finally:
        metrics.setdefault("total_seconds", time.perf_counter() - start)
        monitor.stop()
        save_json(output / "metrics.json", metrics)

    print(json.dumps(metrics, indent=2), flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.plot_resources",
            "--config",
            str(config["_config_file"]),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
