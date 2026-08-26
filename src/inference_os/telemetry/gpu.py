"""GPU telemetry collection, point-in-time sampling, and statistical summaries."""

import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

QuerySampleFn = Callable[[int], Optional["GPUSample"]]
ClockFn = Callable[[], int]


@dataclass(frozen=True, slots=True)
class GPUSample:
    """Instantaneous snapshot of GPU memory and compute metrics."""

    timestamp_ns: int
    memory_used_mb: int
    memory_total_mb: int
    utilization_gpu_pct: int
    utilization_memory_pct: Optional[int] = None


@dataclass(frozen=True, slots=True)
class GPUTelemetrySummary:
    """Aggregated GPU memory and utilization statistics across a benchmark."""

    sample_count: int
    peak_memory_used_mb: int
    avg_memory_used_mb: float
    peak_utilization_gpu_pct: int
    avg_utilization_gpu_pct: float
    total_memory_mb: int


def parse_gpu_sample_output(
    output: str,
    timestamp_ns: Optional[int] = None,
) -> Optional[GPUSample]:
    """Parse comma-separated nvidia-smi sample output."""
    lines = [line.strip() for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return None

    # Format: memory.used, memory.total, utilization.gpu, utilization.memory
    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 3:
        return None

    try:
        mem_used = int(float(parts[0]))
        mem_total = int(float(parts[1]))
        util_gpu = int(float(parts[2]))
        util_mem = int(float(parts[3])) if len(parts) > 3 and parts[3] else None
    except ValueError:
        return None

    ts = timestamp_ns if timestamp_ns is not None else time.perf_counter_ns()
    return GPUSample(
        timestamp_ns=ts,
        memory_used_mb=mem_used,
        memory_total_mb=mem_total,
        utilization_gpu_pct=util_gpu,
        utilization_memory_pct=util_mem,
    )


def query_gpu_sample(
    device_index: int = 0,
    clock_fn: ClockFn = time.perf_counter_ns,
) -> Optional[GPUSample]:
    """Query current GPU metrics for device_index via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return None

    try:
        res = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return None
        return parse_gpu_sample_output(res.stdout, timestamp_ns=clock_fn())
    except Exception:
        return None


def compute_gpu_summary(
    samples: Sequence[GPUSample],
) -> Optional[GPUTelemetrySummary]:
    """Compute aggregate peak and average metrics across collected GPU samples."""
    if not samples:
        return None

    n = len(samples)
    peak_mem = max(s.memory_used_mb for s in samples)
    avg_mem = sum(s.memory_used_mb for s in samples) / n

    peak_util = max(s.utilization_gpu_pct for s in samples)
    avg_util = sum(s.utilization_gpu_pct for s in samples) / n
    total_mem = samples[0].memory_total_mb

    return GPUTelemetrySummary(
        sample_count=n,
        peak_memory_used_mb=peak_mem,
        avg_memory_used_mb=avg_mem,
        peak_utilization_gpu_pct=peak_util,
        avg_utilization_gpu_pct=avg_util,
        total_memory_mb=total_mem,
    )


class GPUTelemetrySampler:
    """Asynchronous background sampler for periodic GPU telemetry capture."""

    def __init__(
        self,
        interval_seconds: float = 0.1,
        device_index: int = 0,
        query_fn: Optional[QuerySampleFn] = None,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.device_index = device_index
        self._query_fn = query_fn if query_fn is not None else query_gpu_sample
        self._samples: list[GPUSample] = []
        self._task: Optional[asyncio.Task[None]] = None
        self._running = False

    async def _sample_loop(self) -> None:
        while self._running:
            try:
                sample = self._query_fn(self.device_index)
                if sample is not None:
                    self._samples.append(sample)
            except Exception:
                pass
            await asyncio.sleep(self.interval_seconds)

    async def start(self) -> None:
        """Start the background polling task."""
        if not self._running:
            self._running = True
            self._samples.clear()
            self._task = asyncio.create_task(self._sample_loop())

    async def stop(self) -> None:
        """Stop the background polling task."""
        if self._running:
            self._running = False
            if self._task is not None:
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
                self._task = None

    async def __aenter__(self) -> "GPUTelemetrySampler":
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        await self.stop()

    def get_samples(self) -> list[GPUSample]:
        """Return shallow copy of collected samples."""
        return list(self._samples)

    def get_summary(self) -> Optional[GPUTelemetrySummary]:
        """Return aggregate summary over collected samples."""
        return compute_gpu_summary(self._samples)
