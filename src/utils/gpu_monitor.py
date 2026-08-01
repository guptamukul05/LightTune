"""
src/utils/gpu_monitor.py

Lightweight GPU usage sampler. Runs in a background thread, polls nvidia-smi
via torch.cuda, and logs memory/utilization at fixed intervals to CSV.
"""

import csv
import threading
import time

import torch


class GPUMonitor:
    def __init__(self, log_path: str, interval_seconds: int = 30):
        self.log_path = log_path
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def _poll(self):
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "gpu_id", "mem_allocated_mb", "mem_reserved_mb"])
            while not self._stop_event.is_set():
                ts = time.time()
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
                    reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
                    writer.writerow([ts, i, round(allocated, 2), round(reserved, 2)])
                f.flush()
                self._stop_event.wait(self.interval_seconds)

    def start(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
