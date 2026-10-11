# V0 baseline evaluation

The 64-video baseline completed successfully. Excluding the warm-up, its median
pipeline time is **106.22 seconds**, with median throughput of **0.6025 videos/s**
(about **36.15 videos/minute**). Use this as the network-download reference for
later experiments on the same workload and environment.

## Run comparison

| Run | Total seconds | Videos/s | Caption seconds | Decode seconds |
|---|---:|---:|---:|---:|
| Warm-up, excluded from median | 108.89 | 0.5878 | 67.29 | 27.96 |
| Run 1 | 112.47 | 0.5690 | 65.32 | 32.29 |
| Run 2 | 106.22 | 0.6025 | 65.04 | 23.67 |
| Run 3 | 102.68 | 0.6233 | 65.58 | 23.72 |

These numbers come from the saved metrics in
[run1](../output/v0_baseline/run1/metrics.json),
[run2](../output/v0_baseline/run2/metrics.json), and
[run3](../output/v0_baseline/run3/metrics.json). All are labeled `network_download`.
The stage CSV durations match the metrics, and the stages do not overlap.
Timing between stages adds approximately one millisecond to each total.

Run times span 102.68–112.47 seconds: a 9.79-second spread, or 9.2% of the median.
Caption time varies by only 0.54 seconds, under 1%. Most variation comes from
decoding and downloading. Run 1 decoded more slowly; run 2 downloaded more slowly.
The saved counters cannot establish whether caching, contention, storage, or
network conditions caused those differences. Three runs establish a useful
reference, but a small improvement needs repeated comparisons to separate it
from this variation.

## Output checks

Each of the four runs contains:

- 64 unique video IDs and 64 unique source IDs, with 64 nonempty captions.
- The same ordered selection: first-shard rows 0–32, then second-shard rows 0–30.
- Ten seconds of decoded video per record.
- Source IDs that match the corresponding local FineVideo shard metadata.

All four `captions.jsonl` files are byte-identical. The same inputs produced the
same saved captions while run time varied. Captions do not visibly end mid-sentence.
Visual accuracy still requires reviewing the videos; generated-token counts and
stopping reasons were not saved, so token-limit behavior cannot be verified here.

## Where the time goes

Run 2 is the run with the median total time:

| Stage | Seconds | Share of run 2 |
|---|---:|---:|
| Download | 10.00 | 9.42% |
| Decode | 23.67 | 22.28% |
| Sampling | 0.56 | 0.53% |
| Preprocess | 6.90 | 6.49% |
| Caption | 65.04 | 61.23% |
| Write | 0.05 | 0.05% |

Captioning is the largest stage to investigate. Its time includes Python startup,
loading the model, reading prepared inputs, generation, and writing intermediate
captions. It is not a generation-only measurement. Decoding is second and also
includes reading shard rows and saving full RGB arrays. Sampling and final writing
are too small to be useful first optimization targets.

## Resource interpretation

The [run 2 resource timeline](../output/v0_baseline/run2/resource_timeline.png)
illustrates the baseline's full-stage barriers: GPU activity begins during
captioning, after CPU preparation has finished.

- Caption-stage GPU utilization averages approximately **59–61%**, including
  startup, across the three measured runs. It peaks at **73–79%**. There is
  sustained activity after startup; these completed runs show no prolonged stall.
- Selected-GPU VRAM peaks at **8.11 GiB** in every run, approximately **40.55%**
  of its reported capacity. This suggests capacity for a later batching experiment,
  but does not determine a safe or faster batch size.
- Stage process RSS peaks at **3.46–4.13 GiB** during captioning. Decoding peaks
  around **2.2 GiB**. Full decoded arrays accumulate on disk, while the decoder
  handles one video at a time in RAM.
- Download receives approximately **106–177 MB/s** on average. Network activity
  is negligible during captioning.
- Resource sampling has a median interval of **250 ms**. Final writing lasts
  less than 67 ms and receives no samples; sampling receives only two or three.

GPU utilization alone does not identify the limiting operation or predict a
batching speedup. System CPU, RAM, and disk counters include activity outside the
pipeline. Use stage process RSS to assess its resident memory. Disk writeback can
also continue into later stages, so a write spike during captioning does not
establish that captioning is storage-bound.

## Next steps

1. Preserve these results and record the Pod's GPU, image, Torch/CUDA versions,
   configuration, and code revision before the next experiment. These were not
   saved per run, so the output files alone cannot reconstruct the environment.
2. Continue iteration 3 in the [progression plan](iterative_progression_plan.md):
   process one shard at a time, reusing one loaded model and processor. Keep the
   exact same 64 selected rows, frames, prompt, precision, and generation settings.
   This changes scheduling and prepares for overlap; it does not guarantee a speedup.
3. Add overlap and CPU workers in the planned order. Later, benchmark small GPU
   batches because captioning dominates. Measure each change against this reference
   with the same input mode and warm-up policy.

For downstream-only experiments, establish a separate `--reuse-shards` reference
with the same workload. Compare local-input runs with that reference rather than
with these network-download runs.
