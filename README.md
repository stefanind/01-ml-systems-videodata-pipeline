# Video pipeline V0

A learning baseline: download complete shards, then decode all selected videos,
sample all frames, preprocess all inputs, caption all videos, and write the results.
Every stage is a standalone script. The runner waits for each stage to finish.

## RunPod setup

Use a [RunPod PyTorch GPU template](https://docs.runpod.io/pods/overview) with Python 3.10–3.13. It supplies CUDA-enabled
PyTorch, which the setup script reuses. Clone this repository on your Pod,
preferably under `/workspace` so the model and decoded files have room. Run all
commands below from the repository root.

1. Accept access conditions on the [FineVideo dataset page](https://huggingface.co/datasets/HuggingFaceFV/finevideo).
2. On the Pod, set a Hugging Face read token with dataset access:

   ```bash
   export HF_TOKEN=hf_your_token
   ```

3. Run setup. This installs dependencies in `.venv` and downloads Qwen2.5-VL-3B
   weights once, outside the benchmark:

   ```bash
   bash setup_runpod.sh
   source .venv/bin/activate
   python -m pipeline.run_baseline
   ```

The default dtype is `bfloat16`. If your GPU does not support it, set `dtype` to
`float16` in `config.json`. The entire model must fit on one GPU; there is no CPU
offloading or quantization in this baseline.

## Small starting workload

`shards.txt` starts with one actual FineVideo Parquet shard, approximately 494 MB.
Uncomment the second URL when ready. All listed shards are downloaded in full.
The default workload processes the first **4 videos**, taking the first **10 seconds**
of each, then sampling **8 frames** per video. Change these settings in `config.json`;
set `max_seconds` to `null` for full videos.

The decoder is PyAV, using one CPU decoding thread. Full decoded RGB frames are
much larger than compressed MP4 bytes and are saved to disk. Start small and check
`data/decoded/` before increasing the workload. Sampling uses presentation timestamps.
Qwen receives the sampled arrays directly, so preprocessing does not decode again.

## Run individual stages

```bash
python -m pipeline.download
python -m pipeline.decode
python -m pipeline.sample
python -m pipeline.preprocess
python -m pipeline.caption
python -m pipeline.write
```

Each script accepts `--config path/to/config.json`. Paths inside a config are
relative to that config's directory. Model weights must already be cached by setup.

`python -m pipeline.run_baseline` downloads shards again so it measures real network input.
For experiments using existing local shards:

```bash
python -m pipeline.run_baseline --reuse-shards
```

Those runs are labeled `local_shards`. All listed shards must already exist;
a missing shard stops the run instead of mixing network input into a local run.
Stages recompute their selected outputs. Each run overwrites `output/`; copy it
elsewhere or change `data_dir`/`output_dir` to preserve an experiment.

## Results

- `output/captions.jsonl`: video IDs, source information, and captions.
- `output/metrics.json`: all six stage times, total seconds, `num_videos`, and
  `throughput = num_videos / total_seconds` in videos/second.
- `output/stages.csv`: stage start/end offsets.
- `output/resources.csv`: samples every **250 ms**, controlled by
  `sample_interval_seconds` in the config.
- `output/resource_timeline.png`: CPU/GPU utilization, RAM/VRAM usage, network
  bandwidth, and disk read/write bandwidth, with stage boundaries.

Stage times include Python startup, stage setup, and intermediate file I/O.
Caption time includes loading the cached Qwen model. Total time ends after final
writing and excludes validation and plotting. Failed runs retain partial timings
with `status: failed`; their throughput is not reported as a successful result.

Stage RSS includes child processes, so Python launchers do not hide the stage's RAM.
System CPU, RAM, disk, and network counters include other applications. GPU counters
describe the selected NVML GPU (`gpu_index`); missing metrics are blank, not zero.
Sampling at 250 ms does not change the GPU driver's own measurement window.
GPU idle time is not collected.

Use `python -m pipeline.plot_resources` to regenerate a plot, or
`python -m pipeline.monitor` for
standalone monitoring until Ctrl+C. A standalone monitor overwrites `resources.csv`,
so do not run it alongside the baseline runner.

## Repository layout

```text
.
├── README.md
├── requirements.txt
├── setup_runpod.sh
├── config.json
├── shards.txt
├── .gitignore
├── docs/
│   └── IMPLEMENTATION_PLAN.md
├── pipeline/
│   ├── __init__.py
│   ├── common.py
│   ├── run_baseline.py
│   ├── download.py
│   ├── decode.py
│   ├── sample.py
│   ├── preprocess.py
│   ├── caption.py
│   ├── write.py
│   ├── monitor.py
│   └── plot_resources.py
├── data/       # Generated; ignored by Git
├── models/     # Downloaded; ignored by Git
└── output/     # Results; ignored by Git
```

The six stage scripts live in `pipeline/`. `common.py` shares config, paths, and
selection-file helpers. `run_baseline.py`, `monitor.py`, and `plot_resources.py`
handle execution, measurements, and the plot. Run each script with
`python -m pipeline.<module>` from the repository root.

The existing plan is in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
Its contents are unchanged. Data, model weights, and results are created when you
run setup and the pipeline; they are excluded from Git.
