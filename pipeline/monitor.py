"""Sample system resources every 250 ms while a stage script runs."""

import csv
import threading
import time

import psutil

from .common import path, read_args

FIELDS = [
    "elapsed_seconds", "stage", "cpu_percent", "gpu_percent",
    "gpu_memory_mib", "gpu_memory_percent", "net_recv_mb_s", "net_sent_mb_s",
    "disk_read_mb_s", "disk_write_mb_s", "ram_used_mib", "stage_rss_mib",
]


def disk_counters():
    try:
        return psutil.disk_io_counters()
    except (OSError, RuntimeError, psutil.Error):
        return None


class ResourceMonitor:
    def __init__(self, output, interval=0.25, gpu_index=0):
        if interval <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self.output, self.interval = output, interval
        self.activity = ("", None)
        self.origin = time.perf_counter()
        self.stop_event, self.ready = threading.Event(), threading.Event()
        self.nvml, self.gpu = None, None
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
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.output.open("w", newline="", encoding="utf-8")
        self.thread = threading.Thread(target=self.collect, daemon=True)
        self.thread.start()
        if not self.ready.wait(timeout=5):
            raise RuntimeError("Resource sampler failed to start")

    def stop(self):
        self.stop_event.set()
        self.thread.join()
        if self.nvml is not None:
            self.nvml.nvmlShutdown()

    def collect(self):
        with self.file as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            previous_time = time.perf_counter()
            previous_net, previous_disk = psutil.net_io_counters(), disk_counters()
            psutil.cpu_percent(interval=None)  # Discard the first reading in this thread.
            self.ready.set()
            next_sample = previous_time + self.interval
            while not self.stop_event.wait(max(0, next_sample - time.perf_counter())):
                now = time.perf_counter()
                elapsed = now - previous_time
                net, disk = psutil.net_io_counters(), disk_counters()
                stage, pid = self.activity
                row = {field: None for field in FIELDS}
                row.update(elapsed_seconds=now - self.origin, stage=stage,
                           cpu_percent=psutil.cpu_percent(interval=None),
                           ram_used_mib=psutil.virtual_memory().used / 2**20)
                if net is not None and previous_net is not None:
                    row["net_recv_mb_s"] = (net.bytes_recv - previous_net.bytes_recv) / elapsed / 1e6
                    row["net_sent_mb_s"] = (net.bytes_sent - previous_net.bytes_sent) / elapsed / 1e6
                if disk is not None and previous_disk is not None:
                    row["disk_read_mb_s"] = (disk.read_bytes - previous_disk.read_bytes) / elapsed / 1e6
                    row["disk_write_mb_s"] = (disk.write_bytes - previous_disk.write_bytes) / elapsed / 1e6
                if pid is not None:
                    try:
                        process = psutil.Process(pid)
                        processes = [process, *process.children(recursive=True)]
                        row["stage_rss_mib"] = sum(item.memory_info().rss for item in processes) / 2**20
                    except psutil.Error:
                        pass  # The stage may have exited between samples.
                if self.gpu is not None:
                    try:
                        row["gpu_percent"] = self.nvml.nvmlDeviceGetUtilizationRates(self.gpu).gpu
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
                previous_time, previous_net, previous_disk = now, net, disk
                next_sample += self.interval
                if next_sample < time.perf_counter():
                    next_sample = time.perf_counter() + self.interval


if __name__ == "__main__":
    config, _ = read_args()
    monitor = ResourceMonitor(path(config, "output_dir", "resources.csv"),
                              config["sample_interval_seconds"], config["gpu_index"])
    monitor.start()
    print("Monitoring; press Ctrl+C to stop.", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
