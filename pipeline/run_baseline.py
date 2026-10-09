"""Run each complete stage sequentially and measure the full pipeline."""

import argparse
import csv
import json
import subprocess
import sys
import time

from .common import ROOT, path, read_args, save_json, selection
from .monitor import ResourceMonitor

STAGES = [
    ("download", "download"), ("decode", "decode"),
    ("sampling", "sample"), ("preprocess", "preprocess"),
    ("caption", "caption"), ("write", "write"),
]


def main():
    """Run all six stages and save their timings, resource samples, and captions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-shards", action="store_true", help="Use local shards; label this as a local-input run")
    config, args = read_args(parser)
    output = path(config, "output_dir")
    output.mkdir(parents=True, exist_ok=True)
    # Remove old final outputs so a failed attempt cannot look like a successful run.
    for filename in ("captions.jsonl", "metrics.json", "resource_timeline.png"):
        (output / filename).unlink(missing_ok=True)
    # One background sampler observes the whole run, including stage transitions.
    monitor = ResourceMonitor(output / "resources.csv", config["sample_interval_seconds"], config["gpu_index"])
    monitor.start()
    metrics = {"input_mode": "local_shards" if args.reuse_shards else "network_download", "status": "failed"}
    process = None
    start = time.perf_counter()
    # Both CSV timelines use the same starting point so their times line up.
    monitor.origin = start
    try:
        with (output / "stages.csv").open("w", newline="", encoding="utf-8") as stage_file:
            writer = csv.DictWriter(stage_file, fieldnames=["stage", "start_seconds", "end_seconds"])
            writer.writeheader()
            for stage, module in STAGES:
                print(f"\n--- {stage} ---", flush=True)
                command = [sys.executable, "-u", "-m", f"pipeline.{module}", "--config", str(config["_config_file"])]
                if stage == "download" and args.reuse_shards:
                    command.append("--reuse-shards")
                # Include Python startup, loading, computation, and file writes in
                # each stage's time. Waiting here keeps the baseline sequential.
                stage_start = time.perf_counter()
                monitor.activity = (stage, None)
                process = subprocess.Popen(command, cwd=ROOT)
                monitor.activity = (stage, process.pid)
                code = process.wait()
                stage_end = time.perf_counter()
                metrics[f"{stage}_seconds"] = stage_end - stage_start
                writer.writerow({"stage": stage, "start_seconds": stage_start - start, "end_seconds": stage_end - start})
                stage_file.flush()
                monitor.activity = ("", None)
                if code:
                    raise subprocess.CalledProcessError(code, command)
                print(f"{stage}: {stage_end - stage_start:.2f} seconds", flush=True)
        # Stop the pipeline clock at the last stage; validation and plotting are
        # reporting work and are outside the measured pipeline time.
        end = stage_end
        records = [json.loads(line) for line in (output / "captions.jsonl").read_text(encoding="utf-8").splitlines()]
        ids = [record["video_id"] for record in records]
        expected = {video["video_id"] for video in selection(config)}
        # Count completed videos only after every selected ID has one caption.
        if not ids or len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError("Final caption IDs must exactly match the selected videos, without duplicates")
        metrics.update(status="complete", total_seconds=end - start,
                       num_videos=len(ids), throughput=len(ids) / (end - start))
    finally:
        # Also save partial timings when a stage fails or the run is interrupted.
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        metrics.setdefault("total_seconds", time.perf_counter() - start)
        monitor.stop()
        save_json(output / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    subprocess.run([sys.executable, "-m", "pipeline.plot_resources", "--config", str(config["_config_file"])], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
