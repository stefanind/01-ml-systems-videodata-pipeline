# Iterative progression plan

Improve throughput through small experiments you can understand, measure, and
compare. Implement one iteration at a time. Each working version becomes the
reference for the next experiment.

The goal is the highest useful throughput for your workload and Pod, within RAM,
VRAM, and disk limits. Some stages will wait when another stage is slower. Keeping
the limiting stage supplied matters more than making every resource show 100%.

This is the next-step plan after [V0](v0_implementation_plan.md). Iteration 2's
reusable-function refactor and iteration 3's sequential shard runner are implemented.
The remaining scheduling iterations are future work.

## Starting point

Your successful default BF16 run on the ADA 2000 / RunPod PyTorch 2.4 Pod processed
four videos, ten seconds per video, with eight sampled frames:

| Measurement | V0 result |
|---|---:|
| Download | 2.97 s |
| Decode | 3.12 s |
| Sampling | 0.32 s |
| Preprocess | 4.73 s |
| Caption | 10.71 s |
| Write | 0.05 s |
| Total | 21.90 s |
| Throughput | 0.183 videos/s |

These values come from your downloaded `output/metrics.json`. Captioning includes
loading Qwen, Python startup, and file I/O. Its time does not tell us how much was
GPU generation alone. Four videos are also too few to establish sustained
throughput. The separate FP16 run changed timing and captions, so it does not
establish a general precision-performance ranking.

Keep the working PyTorch 2.4 image, Qwen2.5-VL-3B, BF16, prompt, frame count,
resolution limits, and generation settings fixed during scheduling experiments.
Continue downloading complete FineVideo shards.

## Roadmap

| Iteration | Implement | Main lesson |
|---|---|---|
| 1 | Repeatable comparisons and a larger fixed workload | Measuring useful work fairly |
| 2 | Reusable per-video functions | Separating stage logic from execution order |
| 3 | Complete one shard through the pipeline at a time | Smaller units of work |
| 4 | Download the next shard while processing the current one | Overlapping I/O with processing |
| 5 | Hand off individual videos to the next stage | Queues and backpressure |
| 6 | Add workers to the measured CPU bottleneck | Parallelism and its costs |
| 7 | Caption small batches on one GPU | Throughput versus memory and latency |
| 8 | Remove expensive intermediate file handoffs | Reducing unnecessary data movement |
| 9 | Profile and tune the remaining bottleneck | Choosing optimizations from evidence |

Iterations 1-5 establish the learning foundation. Iterations 6-8 are experiments:
keep a change when it helps the workload, and retain a simple fallback when it
does not. Iteration 9 is optional until the earlier versions work reliably.

## 1. Make comparisons repeatable

**Implement:** a small experiment log and a few additional timings.

- Preserve the existing four-video run as the quick correctness check.
- Create a fixed throughput workload of 32 or 64 videos, checking RAM and disk
  capacity first. Keep the ten-second limit and eight frames initially. Establish
  a fresh V0 reference for each workload size using the refactored code.
- For shard-overlap experiments, select videos from at least two shards. Your
  current 64-video workload uses the first 33 videos of shard A and the first 31
  of shard B. Both runners preserve this first-`max_videos` selection and its IDs;
  no additional workload manifest is needed for this comparison.
- For a future smaller workload that needs rows from multiple shards, an optional
  fixed manifest could record shard names, row indices, and assigned video IDs.
  Adding another URL alone does not guarantee selecting videos from that shard.
  Introduce manifest support only when that workload needs it, and have both
  runners honor the same selection.
- Run a candidate and its reference three times on the same Pod. Record every
  result and compare median throughput and the variation between runs.
- Use separate data/output directories per experiment so stale files cannot
  satisfy a new run. Record the code revision, configuration, actual Torch/CUDA
  versions, GPU name, and input mode alongside the results.
- Add `model_load_seconds` and `time_to_first_caption_seconds`. Measure the latter
  from pipeline start to the first completed caption. Record generated token
  counts too, because output lengths can differ even with greedy generation.

Use the existing `--reuse-shards` mode to compare downstream changes without
network variation. Recompute decode, sampling, preprocessing, and captions;
reusing those artifacts would change what is being measured. Confirm a promising
change with full network-download runs afterward. Compare each input mode with
the same input mode, and record that repeated local reads may benefit from the
operating system's file cache.

Keep the main metric:

```text
throughput = num_videos / total_seconds
```

Count unique videos successfully written to the final output. Include model
loading and worker startup in total time; stop after final output is closed.
Model weight downloading remains setup work outside the benchmark. Validation
and plotting remain outside the timer. Failed runs are not successful throughput
results. Continue the 250 ms resource sampler.

**Learn:** startup cost, network variation, workload selection, and why a larger
run can have higher videos/s even without a code improvement.

**Ready to continue:** repeated runs produce exactly the expected video IDs and
usable comparisons. A handwritten Markdown table is enough; an experiment
tracking service is unnecessary here.

## 2. Make stage logic reusable without changing the schedule

**Status:** implemented before the larger baseline. Each stage keeps its CLI and
now exposes a per-item function. `load_processor(...)` and `load_model(...)` keep
loading separate from per-video processing. The runner still executes all six
complete stages sequentially; file formats, selection order, and metrics are retained.

**Implement:** small functions in the existing stage modules that operate on one
shard, video, or record. Keep their standalone command-line entry points.

Examples are `download_shard(...)`, `sample_video(...)`, `preprocess_video(...)`,
and `caption_video(...)`. Pass a loaded processor or model to the functions that
need it. Separate loading those objects from processing an item.

Keep `run_baseline.py` executing the same full-stage sequence. Refactor one module
at a time and check its saved artifacts before moving to the next. Sampling
indices, timestamps, prepared tensors, and source metadata should be preserved.
Captioning must still return a completed, nonempty caption for the correct ID.

Assign video IDs from the fixed selection order before submitting work to workers.
Store the mapping to `(shard, row_index)`; completion order must never decide
which video gets which ID.

**Learn:** the difference between what a stage does and when it runs. Separate
`.py` files can remain useful even when a runner imports their functions.

**Ready to continue:** standalone scripts and the V0 runner still work. This is
preparation for scheduling changes, so expect similar performance.

## 3. Complete one shard at a time

**Status:** implemented in `pipeline/run_sharded.py`. It imports the existing
stage functions; `run_baseline.py` remains the reference. Run it with
`python -m pipeline.run_sharded`, optionally passing `--config` or `--reuse-shards`.

```text
shard A: download -> decode selected videos -> sample -> preprocess -> caption -> write
shard B: download -> decode selected videos -> sample -> preprocess -> caption -> write
```

Every stage remains sequential within a shard. One processor loads during the
first preprocessing stage and one Qwen model loads during the first caption
stage. Both remain available across the run. Imports and loads count toward the
stage times and total time. V0 also loads Qwen once per run.

The runner owns the selection metadata and one final JSONL writer. It collects
decoded records across shards, saves `selection.json` once, and checks the complete
expected ID set at the end. Video IDs follow the original global selection order.
All listed shards are downloaded in full; once `max_videos` is reached, later
shards need no video processing. Intermediate files remain available for inspection.

The existing six stage metrics sum durations across shards. `stages.csv` adds a
shard column and records every interval; repeated stages share a color in the plot.
The 250 ms monitor tracks the runner, including its retained model. The new
`time_to_first_caption_seconds` measures pipeline start to the first completed
caption, before that shard's final JSONL writing. A final write interval with an
empty shard name saves the selection manifest and closes the JSONL file. Total
time ends after the final output closes; validation and plotting stay outside
the clock. Use separate data/output directories for comparisons.

First compare the one-shard case with V0, then use the fixed two-shard workload.
Both runners must process the same selected rows and download the same full shards.

**Learn:** changing the unit of scheduling from the complete dataset to a shard.
This can deliver the first result sooner and make intermediate storage easier to
manage. It does not itself overlap work, so a throughput gain is not guaranteed.

**Ready to continue:** both shards contribute their expected captions, one model
instance serves the run, and final output has no duplicates or omissions.

## 4. Overlap downloading with processing

**Implement:** one background downloader while the main runner processes a shard.
A standard-library `ThreadPoolExecutor(max_workers=1)` and one pending download
are enough for the first version. [Python executor documentation](https://docs.python.org/3.11/library/concurrent.futures.html)

```text
download A
process A  ||  download B
process B  ||  download C
```

Start the next download when the current shard enters processing. Await that
download before opening the next shard. Keep one current shard and at most one
prefetched shard; reserve that slot before starting the request. This avoids an
unbounded list of background downloads.

Keep the existing `.part` file followed by rename on completion. Only completed
shard paths may enter processing. Reuse downloaded shards when explicitly running
the local-input experiment. Propagate download failures to the runner.

Now update monitoring to capture the runner and all its worker processes. The
current monitor tracks a single active stage PID; that is insufficient when work
overlaps. Show download and processing activity on separate timeline lanes.

**Learn:** concurrency means activities can progress during each other's waits.
It is different from adding CPU workers that compute simultaneously.

**Measure:** full-run throughput, time to first caption, and whether network
activity overlaps processing. Faster downloading alone may stop helping once
processing determines the run time.

**Ready to continue:** processing starts before all shard downloads finish, no
partial shard is opened, and output identities still match the reference.

## 5. Replace full-stage barriers with per-video handoffs

**Implement in two small steps:**

1. Let CPU preprocessing prepare video B while the main thread captions video A.
   Initially use already sampled files to isolate this change.
2. Extend the producer so a decoded video proceeds through sampling and
   preprocessing immediately, without waiting for every selected video to decode.
   Keep the stage functions separate even if one CPU producer calls them in order.

Start with one CPU producer, one GPU consumer, and one Qwen instance. Keep existing
file handoffs and queue small records containing video metadata and prepared file
paths. The initial ready-input queue can hold two videos.

A bounded queue blocks its producer when full. This slows upstream work when the
GPU cannot consume inputs as quickly, which is called **backpressure**. Python's
`queue.Queue(maxsize=...)` provides this behavior between threads.
[Python queue documentation](https://docs.python.org/3.11/library/queue.html)

Send each video's metadata with its handoff. Today's `selection.json` is written
only after decoding finishes; waiting for that file would preserve the old
barrier. The coordinator collects those records for the final manifest.

Add one end-of-input signal and a shared stop/error mechanism. If a worker fails,
stop the run and unblock other workers; a producer blocked on a full queue must
not leave the runner hanging. Use interruptible queue waits and join workers on
shutdown. This is enough failure handling for the learning version.

Bound unfinished work as well as queue length. A queue of two paths does not
prevent thousands of files from accumulating if the producer writes ahead. Limit
in-flight videos, release intermediates after their last consumer when they are
not being retained for debugging, and keep the downloaded source shards.

**Learn:** streaming between stages, producer/consumer ownership, and bounded
buffers. Each video can progress even while later videos remain upstream.

**Ready to continue:** the timeline shows CPU preparation overlapping captioning,
the GPU can consume the first video before all preprocessing finishes, and memory
or intermediate disk usage does not grow with the entire workload.

### Timing rules once stages overlap

Keep V0 timings unchanged. For the overlapping runner, add an `events.csv` with
`stage`, `worker_id`, `video_id` or shard/batch ID, `start_seconds`, and
`end_seconds`, all relative to one pipeline origin. Have the coordinator write
events so workers do not race to write the same CSV.

Use names such as `decode_work_seconds` for the sum of decode task durations.
Multiple workers' totals can exceed pipeline wall time; they are not a measure of
CPU core time. Queue waiting and initialization are separate from task service
time. Do not add overlapping stage durations to obtain `total_seconds`.
Record GPU task completion after work has finished, retaining the existing
caption-stage synchronization rather than timing only an asynchronous launch.

Use timeline lanes for concurrent tasks and label RSS as pipeline/process RSS
rather than a single active stage. Summed process RSS can count shared pages more
than once. System counters still include other applications. Preserve
`num_videos`, full-run `total_seconds`, and the original throughput formula.

## 6. Add CPU workers where measurements justify them

**Implement:** a small persistent worker pool for the CPU stage limiting delivery
of ready inputs. Try decoding first if the GPU regularly waits for preparation.

Test **1, 2, then 4 workers**, constrained by the Pod's available CPU cores and
memory. Use a `ProcessPoolExecutor` with an explicit `spawn` multiprocessing
context for CPU work. Keep CUDA and the Qwen model in their single owner process.
Keep PyAV's existing one decoding thread per worker for the first sweep.
[Python process-pool documentation](https://docs.python.org/3.11/library/concurrent.futures.html#processpoolexecutor)

Submit only a small bounded number of pending jobs. A worker limit alone does not
limit an eagerly submitted list of videos. Pass paths and lightweight metadata
first; serialization of full decoded frame arrays can erase a speedup.
Have workers read their assigned data without rescanning a whole shard for every
video; a simple reader/dispatcher may pass compressed video bytes when needed.
Measure that transfer cost too.

If preprocessing becomes the bottleneck, test its worker count in a separate
change. Initialize its processor once per worker. Check internal NumPy, Torch,
and tokenizer thread counts if adding workers makes CPU contention worse.

**Learn:** parallelism, worker startup, communication costs, and CPU
oversubscription. More workers can reduce throughput or exhaust RAM.

**Measure:** videos/s, preparation work times, GPU waiting for inputs, peak RAM,
and disk traffic. A persistently full ready-input queue suggests CPU preparation
is already keeping up; more workers there are unlikely to help.

**Ready to continue:** choose the smallest worker count that delivers a repeatable
benefit. Sampling and writing stay simple unless measurements identify them as
meaningful bottlenecks.

## 7. Caption small GPU batches

**Implement in isolation first:** batched preprocessing and captioning with all
inputs ready, then connect the working batch path to the streaming runner.

Try batch sizes **1, 2, then 4**. Use one Qwen instance and one `generate()` call
per batch. Keep the model, BF16, SDPA, prompt, sampled frames, and generation
limits unchanged.

Use Qwen's processor on a list of already sampled video arrays and texts with
padding enabled. Use left padding for generation. Preserve each video's effective
FPS, video grid, and temporal metadata. Existing single-video `.pt` dictionaries
are not safe to batch by blindly applying `torch.stack` to every field.
[Qwen2.5-VL documentation](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_5_vl),
[versioned processor implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py),
[versioned model implementation](https://raw.githubusercontent.com/huggingface/transformers/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)

Map each generated sequence back to its input ID and remove its padded prompt
prefix before decoding. Always process the final partial batch: a five-video run
with batch size two must produce five captions.

In streaming mode, form batches in the CPU preparation path, before calling the
processor. Replace the single-video ready-input handoff with one prepared batch
and its ordered video-ID list. The GPU consumer reads that batch and generates
once; CPU preprocessing can prepare the next batch concurrently. Queue capacity
now counts batches, so include batch size when budgeting RAM. Keep a batch-size-one
fallback.

Use a short maximum wait for filling a batch and flush on end-of-input so a
partial batch cannot wait forever. Record actual batch sizes. Start with FIFO
ordering; only try grouping similar visual-token sizes if variable inputs later
make padding costly.

**Learn:** batching increases useful work per GPU call, but can increase VRAM use
and time to the first result. Different caption lengths can keep a batch working
after shorter sequences finish.

**Measure:** end-to-end videos/s, caption-only videos/s with model load excluded
and clearly labeled, generated tokens, peak VRAM, and time to first caption.
Reserve VRAM headroom for variable videos. The largest batch that fits is not
automatically the fastest.

**Ready to continue:** retain the best measured batch size, flush every tail
batch, and visually inspect captions for the correct videos. Minor wording
differences are possible; swapped IDs or altered sampled inputs are errors.

## 8. Remove expensive intermediate file handoffs

**Implement one handoff change at a time**, keeping a file-based debug option.

1. In the CPU preparation path, decode one video, run the existing sampler on
   its in-memory frames/timestamps, and keep only the selected frames. Stop
   writing and rereading the full decoded RGB array.
2. Pass the small sampled arrays to preprocessing without a sampled `.npy`
   round trip, while keeping the ready-input handoff unchanged.
3. If prepared-input I/O still matters, pass bounded CPU input tensors to the
   single GPU owner instead of saving and loading `.pt` files.

Keep the same sampling indices, timestamps, duration handling, preprocessing,
and number of frames. This first fusion still decodes the selected video portion
fully; it primarily saves storage traffic. It is easier to verify than changing
the decoder and sampling algorithm together.

Measure copies when data crosses processes. In-memory communication is not
automatically free. Keep only a few videos or batches ahead, budget RAM for active
decoders as well as queued items, and release references after consumption.

**Learn:** disk I/O, serialization, memory ownership, and why data movement can
cost more than the arithmetic performed on that data.

**Ready to continue:** selected frames/timestamps and processor inputs match the
reference, memory stays bounded, and throughput or disk/RAM usage improves enough
to justify the change.

## 9. Profile the remaining bottleneck and tune selectively

At this point the useful target structure is:

```mermaid
flowchart LR
    D[Complete shard downloads] --> C[CPU decode workers]
    C --> S[Sample frames]
    S --> P[CPU preprocessing]
    P --> Q[Bounded ready-input queue]
    Q --> G[One Qwen instance with small batches]
    G --> W[One JSONL writer]
```

These are logical stages; cheap adjacent stages can share a worker. Independent
videos and shards overlap. The GPU model remains in one owner process.

Use a short PyTorch profiler capture when the resource timeline cannot explain
the next bottleneck. Profile a few representative batches and benchmark again
with profiling disabled. [PyTorch profiler recipe](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

Choose one experiment from the evidence:

| Observed limit | Next experiment |
|---|---|
| GPU waits for ready inputs | Rebalance CPU workers; try queue capacity 2 then 4 |
| Download delays processing | Compare one versus two download workers, with bounded prefetch |
| CPU-to-GPU transfer is substantial | Try pinned CPU buffers, then investigate transfer overlap |
| Padding or output-length variation dominates batches | Try simple grouping by visual-token size |
| Decode/RGB conversion dominates | Retain/convert only needed frames, checking timestamp equivalence |
| GPU generation dominates and inputs stay ready | Evaluate a compatible attention backend or inference engine in an isolated environment |

Advanced decoder, transfer, or inference changes are separate experiments. Seeking
to sampled frames can still require decoding intervening frames and must preserve
the existing timestamp-based selection. Do not assume a new decoder or GPU decoder
will improve this small workload. Likewise, the code already uses inference mode
and SDPA; they are not future improvements to add again.

Change queue depth, worker count, and batch size one at a time, then repeat the
best combination on the larger workload. A bigger queue can absorb short bursts;
it cannot fix a producer that is consistently slower than its consumer.

## Separate quality-versus-speed experiments

After the scheduling path is understood, you can experiment with fewer frames,
lower pixel limits, a different output-token cap, precision, or quantization.
These change the model's work or behavior. Compare them separately from the
fixed-workload throughput progression and review captions on the same small set
of videos. Lowering resolution is an explicit compute/memory tradeoff in Qwen's
processor. [Qwen resolution controls](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen2_5_vl#notes)

## How to finish each iteration

1. Explain the expected improvement in one sentence before editing.
2. Implement the smallest change that tests it.
3. Run the four-video check and verify IDs, metadata, sampled inputs, and nonempty
   captions. For concurrency, deliberately fail one job and confirm the run exits.
   For batching, include a workload that leaves a partial batch.
4. Run the fixed throughput workload three times for the reference and candidate.
5. Compare videos/s, first-caption time, resource timelines, and peak memory.
6. Record the result and keep or revert the change. Useful learning does not
   require every experiment to improve throughput.

Keep one additional experimental runner and a few explicit settings as they
become needed: worker count, batch size, and queue capacity. Use the existing
stage modules and monitor. Introduce abstractions only when repeated code makes
their purpose clear.

The practical stopping point is a correct, understandable pipeline whose
throughput has stabilized in repeated runs, whose limiting stage stays supplied,
and whose RAM/VRAM/disk usage is manageable. If the GPU is already the limiting
resource, more download or CPU workers will mainly add cost. "Optimal" here is
the best configuration demonstrated for your fixed workload and hardware; it
will need to be measured again if either changes.
