# V0: simple scripts, shard downloads, and resource measurements

Build a small, understandable baseline with a real script for each stage. Every stage finishes the complete selected workload before the next begins. The resource monitor runs throughout the pipeline.

## 1. Files to build

```text
run_baseline.py     # Runs the stage scripts in order and records timings
download.py         # Downloads the selected remote shards
decode.py           # Reads videos from shards and decodes frames
sample.py           # Selects frames from the decoded output
preprocess.py       # Uses the caption model's processor
caption.py          # Generates one video's caption at a time
write.py            # Assembles the final captions file
monitor.py          # Collects resource samples during the run
plot_resources.py   # Plots the resource timeline after the run
config.json         # A few settings shared by the scripts
shards.txt          # The remote shard URLs to download
```

Each stage script can run on its own. `run_baseline.py` invokes them sequentially and waits for each to exit successfully. It stops if a stage fails. There is one processing stage active at a time, with no stage overlap, worker pools, or inference batching in V0. Monitoring runs in a small background thread so it can observe the active stage.

## 2. Begin with real shard downloads

Use existing dataset shards rather than downloading individual videos and packaging them afterward.

FineVideo already has downloadable Parquet shards. Its file listing currently includes `train-00000-of-01357.parquet` at approximately 494 MB and `train-00001-of-01357.parquet` at approximately 566 MB. Those are two concrete starting inputs, subject to accepting dataset access conditions. They do not need to be converted into TAR archives for V0. [FineVideo shard listing](https://huggingface.co/datasets/HuggingFaceFV/finevideo/tree/main/data)

`download.py` downloads each listed shard completely over HTTP, one at a time, into `data/shards/`. Use a straightforward streaming HTTP request that writes chunks to disk. Cache model weights once during setup so the measured network download stage concerns dataset shards.

Start with one shard while getting the scripts working, then use two. Keep the processing workload small: for example, the first four videos across the selected shards, decoding only their first 10 seconds. Save the selected IDs and time limits in a simple `selection.json`. **The full shards are downloaded; downstream processing uses the explicitly selected small workload.** This lets you learn the complete path without decoding hundreds of minutes on the first attempt.

Keep those choices the same when comparing changes. Increase video count and duration once the baseline works. If you want to caption full videos from the beginning, remove the time limit after checking how much decoded output one video produces.

## 3. Stage responsibilities and file handoffs

```text
Download ALL selected shards
    -> Decode ALL selected video portions
    -> Sample ALL decoded videos
    -> Preprocess ALL sampled videos
    -> Caption ALL prepared videos
    -> Write ALL final caption records
```

| Script | What it does | Output |
|---|---|---|
| `download.py` | Downloads the complete remote shard files | `data/shards/` |
| `decode.py` | Reads the Parquet video bytes and metadata; decodes every frame in each selected video portion on CPU | `data/decoded/`: per-video RGB arrays and timestamps |
| `sample.py` | Selects, for example, 8 frames spaced across each decoded portion; saves selected timestamps too | `data/sampled/`: per-video sampled arrays |
| `preprocess.py` | Applies the chosen model processor to the sampled frames and a fixed caption prompt | `data/preprocessed/`: CPU input tensors |
| `caption.py` | Loads the model, reads prepared inputs, generates one caption at a time, and saves intermediate caption text | `data/captions/`: one text file per video |
| `write.py` | Combines video IDs and caption text into the final JSONL output | `output/captions.jsonl` |

Simple NumPy files for frames/timestamps and PyTorch files for prepared tensors are enough. Separate processes need file handoffs; decoded frames therefore live on disk rather than in a list shared between scripts. Start with one video's frames in memory at a time while creating those files.

Choose one decoder and one video-capable caption model that work on your hardware. Get a single caption working first. The Qwen model discussed earlier remains an option. The processor must consume your sampled arrays so it does not silently decode or sample the original video again.

Intermediate reads and writes are part of each stage's timing. `write_seconds` measures the final output assembly; it does not represent every disk write in the pipeline. Keep this file-based baseline while first experimenting with scheduling. Removing intermediate files can be a later change.

## 4. Record stage timings, video count, and throughput

The runner measures each script with `time.perf_counter()` and writes `output/metrics.json`:

```python
metrics = {
    "download_seconds": ...,
    "decode_seconds": ...,
    "sampling_seconds": ...,
    "preprocess_seconds": ...,
    "caption_seconds": ...,
    "write_seconds": ...,
    "total_seconds": ...,
    "num_videos": ...,
    "throughput": num_videos / total_seconds,
}
```

`num_videos` is the number of distinct videos successfully processed and written to the final captions file, rather than the total number contained in the downloaded shards. `throughput` is measured in videos per second and uses the complete run's `total_seconds`.

These are elapsed wall times around the script executions, including process startup, file I/O, and stage setup. For the first version, `caption_seconds` includes loading the cached model. That is fine as long as its meaning is clear; separate model loading later if it becomes a question you want to investigate.

`total_seconds` runs from just before the first stage starts until the last stage exits. Resource initialization and plotting sit outside that timer. Also save each stage's start/end offset from pipeline start in `output/stages.csv`, so the resource plots can show stage boundaries.

The caption script must wait for GPU work and return completed caption text before finishing. A script that exits successfully should have closed its output files.

A real download measurement uses a fresh download destination. A rerun that uses existing shards is labeled as local-input execution rather than counted as another network download.

## 5. Collect the requested resources throughout V0

For small runs, `monitor.py` targets one sample every **250 ms** (`sample_interval_seconds = 0.25`, approximately four samples per second), beginning before downloading and ending after final writing. Keep the interval in `config.json`. Save the samples to `output/resources.csv` with elapsed time and the currently active stage.

| Resource | Record |
|---|---|
| CPU utilization | System CPU percentage |
| GPU utilization | Utilization percentage for the selected GPU |
| GPU memory utilization | VRAM used in MiB and percentage of total VRAM capacity |
| Network bandwidth | Receive and transmit MB/s |
| Disk read MB/s | Change in disk read bytes divided by elapsed sample time |
| Disk write MB/s | Change in disk write bytes divided by elapsed sample time |
| RAM usage | System RAM used in MiB and active stage process RSS in MiB |

Use `psutil` for CPU, RAM, network, and disk counters. Compute bandwidth from counter differences over the actual elapsed interval; initialize the counters and discard the first nonblocking CPU reading. These system counters can include other applications, so label them as system measurements. [psutil API reference](https://psutil.io/api/)

Use NVIDIA NVML through a Python binding for GPU utilization and VRAM usage. VRAM capacity used is a different measurement from memory-controller activity; label the plotted value as VRAM used. [NVML device queries](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)

Unsupported GPU or disk readings should be marked unavailable, rather than filled with zero.

## 6. Produce the resource timeline

After the baseline finishes, `plot_resources.py` creates `output/resource_timeline.png` with a shared time axis and stage boundaries:

- CPU and GPU utilization.
- RAM and VRAM usage.
- Network receive/transmit MB/s.
- Disk read/write MB/s.

Use separate panels or scales for quantities with different units. Shade or label the download, decode, sampling, preprocess, caption, and write intervals. Show `num_videos` and throughput alongside the stage timing summary.

This plot should make observations visible: GPU inactivity before captioning, CPU activity during decoding, network activity during download, and disk traffic at the file handoffs. The actual measurements determine how pronounced those patterns are.

## 7. Build it in this order

1. Implement `download.py` and retrieve one actual shard.
2. Implement `decode.py` for one selected video portion and save its frames.
3. Implement `sample.py`, then `preprocess.py`.
4. Implement `caption.py`, then `write.py`; confirm the saved caption matches its input ID.
5. Connect the scripts with `run_baseline.py` and add stage timings.
6. Add `monitor.py` and `plot_resources.py`, then run the complete small workload.

V0 is complete when all selected videos have captions, stages execute in the required order, all seven timing fields plus `num_videos` and `throughput` are recorded, the resource samples exist, and the resource timeline is generated. A missing hardware metric is explicitly identified.

The first improvements can then be: finish the pipeline for one shard before starting the next; overlap downloading the next shard with processing the current shard; add decoding workers if useful; and try GPU batching. Change one thing at a time and compare both total runtime and resource timelines.

## Feedback on the idea

Separate scripts, real shards, and resource monitoring are appropriate parts of this learning baseline. Keep the execution straightforward and the workload small. The timeline will help you choose the next experiment by showing which resources are active, idle, or waiting. Some waiting is normal when another stage is slower; the eventual goal is to keep the limiting stage supplied without accumulating excessive data.
