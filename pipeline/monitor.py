"""Sample system resources every 250 ms while a stage script runs."""

import csv
import threading
import time

import psutil

from .common import path, read_args

FIELDS = [
    "elapsed_seconds",
    "stage",
    "cpu_percent",
    "gpu_percent",
    "gpu_memory_mib",
    "gpu_memory_percent",
    "net_recv_mb_s",
    "net_sent_mb_s",
    "disk_read_mb_s",
    "disk_write_mb_s",
    "ram_used_mib",
    "stage_rss_mib",
]


def disk_counters():
    """Return system disk counters, or None when this environment hides them."""
    try:
        return psutil.disk_io_counters()
    except (OSError, RuntimeError, psutil.Error):
        return None


class ResourceMonitor:
    """Write resource samples in a background thread while stage scripts run."""

    def __init__(self, output, interval=0.25, gpu_index=0):
        if interval <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.output = output
        self.interval = interval

        # The runner updates this pair with the active stage name and child PID.
        self.activity = ("", None)
        self.origin = time.perf_counter()
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.nvml = None
        self.gpu = None

        # NVML reads NVIDIA device counters without loading a model or CUDA tensors.
        try:
            import pynvml
        except ImportError:
            pass
        else:
            try:
                pynvml.nvmlInit()
                self.nvml = pynvml
                self.gpu = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except pynvml.NVMLError:
                pass

        if self.gpu is None:
            print("GPU metrics unavailable; GPU CSV fields will be blank.", flush=True)

    def start(self):
        """Open the CSV and wait until the sampler's initial counters are ready."""
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output.open("w", newline="", encoding="utf-8")
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("Resource sampler failed to start")

    def stop(self):
        """Stop sampling, close the CSV, and release the NVML connection."""
        self.stop_event.set()
        self.thread.join()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()

    def collect(self):
        """Measure counter changes over each actual sample interval."""
        with self.file as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()

            previous_time = time.perf_counter()
            previous_net = psutil.net_io_counters()
            previous_disk = disk_counters()
            psutil.cpu_percent(interval=None)  # Discard the first reading in this thread.
            self.ready.set()
            next_sample = previous_time + self.interval

            while not self.stop_event.wait(max(0, next_sample - time.perf_counter())):
                now = time.perf_counter()
                # Scheduling can drift, so rates use the measured elapsed time
                # rather than assuming that exactly 250 ms passed.
                elapsed = now - previous_time
                net = psutil.net_io_counters()
                disk = disk_counters()
                stage, pid = self.activity

                # None becomes a blank CSV field; missing data is not a zero.
                row = {field: None for field in FIELDS}
                # CPU, RAM, network, and disk counters cover the system visible
                # to the container and may include activity outside this pipeline.
                row.update(
                    elapsed_seconds=now - self.origin,
                    stage=stage,
                    cpu_percent=psutil.cpu_percent(interval=None),
                    ram_used_mib=psutil.virtual_memory().used / 2**20,
                )

                # I/O rates use decimal MB; memory sizes use binary MiB.
                if net is not None and previous_net is not None:
                    received_bytes = net.bytes_recv - previous_net.bytes_recv
                    sent_bytes = net.bytes_sent - previous_net.bytes_sent
                    row["net_recv_mb_s"] = received_bytes / elapsed / 1e6
                    row["net_sent_mb_s"] = sent_bytes / elapsed / 1e6

                if disk is not None and previous_disk is not None:
                    read_bytes = disk.read_bytes - previous_disk.read_bytes
                    written_bytes = disk.write_bytes - previous_disk.write_bytes
                    row["disk_read_mb_s"] = read_bytes / elapsed / 1e6
                    row["disk_write_mb_s"] = written_bytes / elapsed / 1e6

                # RSS is the resident RAM of the stage process and its children,
                # which helps distinguish pipeline usage from system RAM usage.
                if pid is not None:
                    try:
                        process = psutil.Process(pid)
                        processes = [process, *process.children(recursive=True)]
                        rss_bytes = sum(item.memory_info().rss for item in processes)
                        row["stage_rss_mib"] = rss_bytes / 2**20
                    except psutil.Error:
                        pass  # The stage may have exited between samples.

                if self.gpu is not None:
                    # GPU activity and VRAM occupancy are separate device metrics.
                    # Memory percent means capacity used, not memory bandwidth.
                    try:
                        utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu)
                        row["gpu_percent"] = utilization.gpu
                    except self.nvml.NVMLError:
                        pass
                    try:
                        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.gpu)
                        row["gpu_memory_mib"] = memory.used / 2**20
                        row["gpu_memory_percent"] = 100 * memory.used / memory.total
                    except self.nvml.NVMLError:
                        pass

                writer.writerow(row)
                output.flush()
                previous_time = now
                previous_net = net
                previous_disk = disk
                next_sample += self.interval
                # Skip missed deadlines instead of producing a burst of samples.
                if next_sample < time.perf_counter():
                    next_sample = time.perf_counter() + self.interval


if __name__ == "__main__":
    config, _ = read_args()
    monitor = ResourceMonitor(
        path(config, "output_dir", "resources.csv"),
        config["sample_interval_seconds"],
        config["gpu_index"],
    )
    monitor.start()
    print("Monitoring; press Ctrl+C to stop.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
