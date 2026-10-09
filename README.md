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

Setup disables the optional `hf_transfer` downloader for the model download,
so RunPod's inherited `HF_HUB_ENABLE_HF_TRANSFER=1` setting does not require an
extra package. If you encounter that error with an older checkout, rerun setup:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 bash setup_runpod.sh
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

Captioning prints the model's device and dtype, model load time, and progress and
elapsed time for each video. For a slow run, check `nvidia-smi` in a second terminal
and the latest rows of `output/resources.csv` to see GPU utilization and memory.

If captioning fails with a CUDA launch error, retry that stage in a fresh process
using the existing prepared inputs:

```bash
CUDA_LAUNCH_BLOCKING=1 python -m pipeline.caption
```

This makes CUDA calls synchronous so the traceback can identify the failing
operation more precisely. Leave this debug setting unset for latency measurements.

To test FP16 separately, wait for the current caption process to exit, then run:

```bash
CUDA_LAUNCH_BLOCKING=1 python -m pipeline.caption --config config.fp16.json
```

This tests whether the failure depends on precision. It uses the existing prepared
inputs and writes per-video captions in `data/captions/`. A complete run with this
config writes final results to `output/fp16/`. The FP16 trial is not a confirmed fix.

If that attempt also hangs, stop its caption process before running this bounded
diagnostic. It prints the blocked Python stack after 30 seconds and exits:

```bash
CUDA_LAUNCH_BLOCKING=1 python -u - <<'PY'
import faulthandler
import runpy
import sys

sys.argv = ["pipeline.caption", "--config", "config.fp16.json"]
faulthandler.dump_traceback_later(30, exit=True)
try:
    runpy.run_module("pipeline.caption", run_name="__main__")
finally:
    faulthandler.cancel_dump_traceback_later()
PY
```

The 30-second limit includes model startup. This is only for troubleshooting.

For a failure located in `Conv3d`, test that operation separately after the failed
caption process has exited:

```bash
env -u CUDA_LAUNCH_BLOCKING python -m pipeline.check_gpu --config config.fp16.json --no-cudnn
```

This reads the first video's prepared pixels and Qwen's cached vision config. It
uses test weights for the same convolution shape and tests PyTorch's native path
with cuDNN disabled. It prints each step and exits with a stack trace if GPU work
takes over 20 seconds. It does not load the full model or write benchmark results.
The test synchronizes before and after the convolution. Leave launch blocking
unset here: the [native CUDA implementation](https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/NaiveDilatedConvolution.cu)
processes patches one at a time, so forcing every launch to synchronize adds
overhead. A watchdog timeout by itself does not establish a GPU fault.

If the convolution keeps failing or timing out, compare a separate environment
with [PyTorch 2.7.1, torchvision 0.22.1, and CUDA 12.6](https://pytorch.org/get-started/previous-versions/#v271):

```bash
bash setup_runpod.sh --cu126
source .venv-cu126/bin/activate
env -u CUDA_LAUNCH_BLOCKING -u TORCH_CUDNN_V8_API_DISABLED python -m pipeline.check_gpu --config config.fp16.json
```

This installs its own PyTorch and project dependencies in `.venv-cu126` without
inheriting template packages. Setup reuses the cached model files. The check uses
cuDNN normally and retains its 20-second watchdog. If it passes, try captioning:

```bash
env -u CUDA_LAUNCH_BLOCKING -u TORCH_CUDNN_V8_API_DISABLED python -m pipeline.caption --config config.fp16.json
```

This is an environment comparison, not a confirmed fix. The CUDA packages are
large, so the first installation takes longer than setup with template PyTorch.

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
│   ├── check_gpu.py
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

The existing plan is in [docs/v0_implementation_plan.md](docs/v0_implementation_plan.md).
Its contents are unchanged. Data, model weights, and results are created when you
run setup and the pipeline; they are excluded from Git.
