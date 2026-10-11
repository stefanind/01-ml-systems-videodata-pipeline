"""Plot resource samples with the sequential stage boundaries."""

import csv
import json
import math

import matplotlib

matplotlib.use("Agg")  # Save a PNG without needing a desktop/display on RunPod.
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .common import path, read_args


def main():
    """Combine resource samples and stage timings into a shared timeline."""
    config, _ = read_args()
    output = path(config, "output_dir")

    with (output / "resources.csv").open(encoding="utf-8", newline="") as source:
        samples = list(csv.DictReader(source))
    with (output / "stages.csv").open(encoding="utf-8", newline="") as source:
        stages = list(csv.DictReader(source))
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))

    times = [float(row["elapsed_seconds"]) for row in samples]
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(12, 12))

    def line(axis, field, label):
        # A blank sample stays missing in the plot instead of appearing as zero.
        values = []
        for row in samples:
            if row[field]:
                values.append(float(row[field]))
            else:
                values.append(float("nan"))

        if not any(math.isfinite(value) for value in values):
            label += " (unavailable)"
        axis.plot(times, values, label=label)

    # Separate panels keep percentages, memory sizes, and I/O rates readable.
    line(axes[0], "cpu_percent", "System CPU")
    line(axes[0], "gpu_percent", "GPU")
    axes[0].set_ylabel("Utilization (%)")
    axes[0].set_ylim(0, 110)

    line(axes[1], "ram_used_mib", "System RAM used")
    rss_label = "Stage process RSS"
    if metrics.get("execution_mode") == "shard_sequential":
        rss_label = "Pipeline process RSS"
    line(axes[1], "stage_rss_mib", rss_label)
    axes[1].set_ylabel("RAM (MiB)")

    line(axes[2], "gpu_memory_mib", "VRAM used")
    axes[2].set_ylabel("VRAM (MiB)")

    line(axes[3], "net_recv_mb_s", "Receive")
    line(axes[3], "net_sent_mb_s", "Transmit")
    axes[3].set_ylabel("Network (MB/s)")

    line(axes[4], "disk_read_mb_s", "Read")
    line(axes[4], "disk_write_mb_s", "Write")
    axes[4].set_ylabel("Disk (MB/s)")

    # Repeated stages use the same color across shards. The V0 stage order keeps
    # its original colors, and the legend lists each stage only once.
    stage_colors = {}
    for stage in stages:
        name = stage["stage"]
        if name not in stage_colors:
            stage_colors[name] = f"C{len(stage_colors)}"

    # Shade the same stage interval across every panel so resource activity can
    # be matched to the work that was running at that moment.
    for stage in stages:
        left = float(stage["start_seconds"])
        right = float(stage["end_seconds"])
        color = stage_colors[stage["stage"]]
        for axis in axes:
            axis.axvspan(left, right, color=color, alpha=0.08)
            axis.axvline(left, color="grey", linewidth=0.5)

    for axis in axes:
        axis.legend(loc="upper right")
        axis.grid(alpha=0.2)
        axis.set_xlim(0, metrics["total_seconds"])
        axis.set_ylim(bottom=0)

    axes[-1].set_xlabel("Seconds from pipeline start")
    fig.suptitle(
        f"{metrics['num_videos']} videos | "
        f"{metrics['throughput']:.4f} videos/second | {metrics['input_mode']}"
    )
    stage_labels = []
    for name, color in stage_colors.items():
        stage_labels.append(Patch(color=color, alpha=0.3, label=name))
    fig.legend(
        handles=stage_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=6,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output / "resource_timeline.png", dpi=150)
    plt.close(fig)
    print(f"Saved {output / 'resource_timeline.png'}")


if __name__ == "__main__":
    main()
