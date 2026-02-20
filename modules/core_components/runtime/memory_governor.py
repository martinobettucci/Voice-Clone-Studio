"""Global memory admission control for heavy workloads."""

from __future__ import annotations

import gc
import json
import logging
import os
import resource
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import psutil
import torch

from modules.core_components.ai_models.model_utils import empty_device_cache


LOGGER = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time memory stats used for admission and UI."""

    timestamp: float
    pid: int
    rss_bytes: int
    rss_pct_of_total: float
    system_available_bytes: int
    system_total_bytes: int
    gpu_allocated_bytes: int
    gpu_reserved_bytes: int
    gpu_total_bytes: int
    gpu_reserved_pct_of_total: float
    active_heavy_jobs: int
    active_heavy_job_names: tuple[str, ...]
    admitted_jobs_total: int
    rejected_jobs_total: int
    completed_jobs_total: int


class MemoryAdmissionError(RuntimeError):
    """Raised when strict memory admission rejects a new heavy job."""

    def __init__(self, message: str, snapshot: MemorySnapshot):
        super().__init__(message)
        self.snapshot = snapshot


class MemoryGovernor:
    """Single-lane heavy workload governor with strict memory admission checks."""

    def __init__(self, config: dict[str, Any] | None = None):
        self._lock = threading.RLock()
        self._active_heavy_jobs = 0
        self._active_job_names: set[str] = set()
        self._admitted_jobs_total = 0
        self._rejected_jobs_total = 0
        self._completed_jobs_total = 0
        self._process = psutil.Process(os.getpid())
        self._config = {}
        self.update_config(config or {})

    def update_config(self, config: dict[str, Any]) -> None:
        """Refresh thresholds from config + environment overrides."""
        self._config = config or {}
        self.max_active_heavy_jobs = 1
        self.memory_max_rss_pct = float(
            os.getenv(
                "VCS_MEMORY_MAX_RSS_PCT",
                self._config.get("memory_max_rss_pct", 70.0),
            )
        )
        self.memory_min_available_mb = int(
            os.getenv(
                "VCS_MEMORY_MIN_AVAILABLE_MB",
                self._config.get("memory_min_available_mb", 2048),
            )
        )
        self.memory_max_gpu_reserved_pct = float(
            os.getenv(
                "VCS_MEMORY_MAX_GPU_RESERVED_PCT",
                self._config.get("memory_max_gpu_reserved_pct", 90.0),
            )
        )
        self.heavy_job_timeout_s = int(
            os.getenv(
                "VCS_HEAVY_JOB_TIMEOUT_S",
                self._config.get("heavy_job_timeout_s", 7200),
            )
        )

    def snapshot(self) -> MemorySnapshot:
        """Capture process/system/device memory stats."""
        vm = psutil.virtual_memory()
        rss_bytes = int(self._process.memory_info().rss)
        rss_pct = (rss_bytes / vm.total * 100.0) if vm.total else 0.0

        gpu_allocated = 0
        gpu_reserved = 0
        gpu_total = 0
        gpu_reserved_pct = 0.0

        if torch.cuda.is_available():
            try:
                device_idx = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_idx)
                gpu_total = int(props.total_memory)
                gpu_allocated = int(torch.cuda.memory_allocated(device_idx))
                gpu_reserved = int(torch.cuda.memory_reserved(device_idx))
                if gpu_total > 0:
                    gpu_reserved_pct = (gpu_reserved / gpu_total) * 100.0
            except Exception:
                pass

        with self._lock:
            active_heavy_jobs = self._active_heavy_jobs
            active_heavy_job_names = tuple(sorted(self._active_job_names))
            admitted_jobs_total = self._admitted_jobs_total
            rejected_jobs_total = self._rejected_jobs_total
            completed_jobs_total = self._completed_jobs_total

        return MemorySnapshot(
            timestamp=time.time(),
            pid=self._process.pid,
            rss_bytes=rss_bytes,
            rss_pct_of_total=rss_pct,
            system_available_bytes=int(vm.available),
            system_total_bytes=int(vm.total),
            gpu_allocated_bytes=gpu_allocated,
            gpu_reserved_bytes=gpu_reserved,
            gpu_total_bytes=gpu_total,
            gpu_reserved_pct_of_total=gpu_reserved_pct,
            active_heavy_jobs=active_heavy_jobs,
            active_heavy_job_names=active_heavy_job_names,
            admitted_jobs_total=admitted_jobs_total,
            rejected_jobs_total=rejected_jobs_total,
            completed_jobs_total=completed_jobs_total,
        )

    def reject_reason(self, snapshot: MemorySnapshot) -> str | None:
        """Return deterministic rejection reason, or None if request is admissible."""
        if snapshot.active_heavy_jobs >= self.max_active_heavy_jobs:
            return "Another heavy task is already running. Strict mode allows only one heavy task at a time."

        if snapshot.system_available_bytes < self.memory_min_available_mb * 1024 * 1024:
            return (
                "System memory is too low to safely start this task "
                f"(< {self.memory_min_available_mb} MB available)."
            )

        if snapshot.rss_pct_of_total > self.memory_max_rss_pct:
            return (
                "Process memory usage is above the strict safety threshold "
                f"({snapshot.rss_pct_of_total:.1f}% > {self.memory_max_rss_pct:.1f}%)."
            )

        if snapshot.gpu_total_bytes > 0 and snapshot.gpu_reserved_pct_of_total > self.memory_max_gpu_reserved_pct:
            return (
                "GPU memory reservation is above the strict safety threshold "
                f"({snapshot.gpu_reserved_pct_of_total:.1f}% > {self.memory_max_gpu_reserved_pct:.1f}%)."
            )

        return None

    @staticmethod
    def format_snapshot(snapshot: MemorySnapshot) -> str:
        """Human-readable memory summary for UI."""
        rss_gb = snapshot.rss_bytes / (1024 ** 3)
        avail_gb = snapshot.system_available_bytes / (1024 ** 3)
        if snapshot.gpu_total_bytes > 0:
            gpu_reserved_gb = snapshot.gpu_reserved_bytes / (1024 ** 3)
            gpu_total_gb = snapshot.gpu_total_bytes / (1024 ** 3)
            gpu_text = f"GPU reserved: {gpu_reserved_gb:.2f}/{gpu_total_gb:.2f} GB ({snapshot.gpu_reserved_pct_of_total:.1f}%)"
        else:
            gpu_text = "GPU reserved: N/A"
        return (
            f"Heavy jobs: {snapshot.active_heavy_jobs} | "
            f"Active names: {', '.join(snapshot.active_heavy_job_names) or 'none'} | "
            f"RSS: {rss_gb:.2f} GB ({snapshot.rss_pct_of_total:.1f}%) | "
            f"RAM available: {avail_gb:.2f} GB | "
            f"{gpu_text}"
        )

    @staticmethod
    def _peak_rss_bytes() -> int:
        """Best-effort process peak RSS in bytes (POSIX ru_maxrss)."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            peak = int(usage.ru_maxrss)
            if peak <= 0:
                return 0
            # Linux/BSD report KB, macOS reports bytes.
            if sys.platform == "darwin":
                return peak
            return peak * 1024
        except Exception:
            return 0

    def _log_event(
        self,
        event: str,
        job_name: str,
        snapshot: MemorySnapshot,
        reason: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "event": event,
            "job_name": job_name,
            "timestamp": snapshot.timestamp,
            "pid": snapshot.pid,
            "active_heavy_jobs": snapshot.active_heavy_jobs,
            "active_heavy_job_names": list(snapshot.active_heavy_job_names),
            "admitted_jobs_total": snapshot.admitted_jobs_total,
            "rejected_jobs_total": snapshot.rejected_jobs_total,
            "completed_jobs_total": snapshot.completed_jobs_total,
            "rss_bytes": snapshot.rss_bytes,
            "rss_pct_of_total": round(snapshot.rss_pct_of_total, 2),
            "system_available_bytes": snapshot.system_available_bytes,
            "gpu_reserved_bytes": snapshot.gpu_reserved_bytes,
            "gpu_reserved_pct_of_total": round(snapshot.gpu_reserved_pct_of_total, 2),
        }
        if reason:
            payload["reason"] = reason
        if extra:
            payload.update(extra)
        LOGGER.info("memory_governor %s", json.dumps(payload, sort_keys=True))

    def _enter_heavy_job(self, job_name: str, timeout_s: int | None = None) -> dict[str, Any]:
        """Run admission checks and mark a heavy job as active."""
        timeout = int(timeout_s) if timeout_s is not None else int(self.heavy_job_timeout_s)
        start = time.monotonic()
        peak_rss_before = self._peak_rss_bytes()

        with self._lock:
            admission_snapshot = self.snapshot()
            reason = self.reject_reason(admission_snapshot)
            if reason:
                self._rejected_jobs_total += 1
                self._log_event("reject", job_name, admission_snapshot, reason=reason)
                raise MemoryAdmissionError(reason, admission_snapshot)

            self._active_heavy_jobs += 1
            self._active_job_names.add(str(job_name))
            self._admitted_jobs_total += 1
            running_snapshot = self.snapshot()
            self._log_event("admit", job_name, running_snapshot)
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

        return {
            "timeout": timeout,
            "start": start,
            "peak_rss_before": peak_rss_before,
        }

    def _exit_heavy_job(self, job_name: str, ctx: dict[str, Any]) -> None:
        """Finalize heavy job accounting and cleanup."""
        gc.collect()
        empty_device_cache()

        elapsed = time.monotonic() - float(ctx.get("start", time.monotonic()))
        timeout = int(ctx.get("timeout", self.heavy_job_timeout_s))
        if timeout > 0 and elapsed > timeout:
            LOGGER.warning("memory_governor job '%s' exceeded timeout (%ss > %ss)", job_name, int(elapsed), timeout)

        peak_rss_before = int(ctx.get("peak_rss_before", 0))
        peak_rss_after = self._peak_rss_bytes()
        job_peak_rss = max(
            peak_rss_before,
            peak_rss_after,
            int(self._process.memory_info().rss),
        )
        job_peak_gpu_reserved = 0
        if torch.cuda.is_available():
            try:
                device_idx = torch.cuda.current_device()
                job_peak_gpu_reserved = int(torch.cuda.max_memory_reserved(device_idx))
            except Exception:
                job_peak_gpu_reserved = 0

        with self._lock:
            self._active_heavy_jobs = max(0, self._active_heavy_jobs - 1)
            self._active_job_names.discard(str(job_name))
            self._completed_jobs_total += 1
            exit_snapshot = self.snapshot()
            self._log_event(
                "exit",
                job_name,
                exit_snapshot,
                extra={
                    "elapsed_s": round(elapsed, 3),
                    "job_peak_rss_bytes": job_peak_rss,
                    "job_peak_gpu_reserved_bytes": job_peak_gpu_reserved,
                },
            )

    def run_heavy(
        self,
        job_name: str,
        fn: Callable[[], Any],
        tenant_id: str | None = None,
        timeout_s: int | None = None,
    ) -> Any:
        """Run a heavy function with strict admission and serialized concurrency."""
        del tenant_id  # Reserved for future per-tenant policy.

        timeout = int(timeout_s) if timeout_s is not None else int(self.heavy_job_timeout_s)
        ctx = self._enter_heavy_job(job_name=job_name, timeout_s=timeout)

        try:
            result = fn()
            return result
        finally:
            self._exit_heavy_job(job_name=job_name, ctx=ctx)

    def run_heavy_stream(
        self,
        job_name: str,
        fn: Callable[[], Any],
        tenant_id: str | None = None,
        timeout_s: int | None = None,
    ):
        """Run a heavy streaming function while holding admission until stream completes."""
        del tenant_id  # Reserved for future per-tenant policy.

        timeout = int(timeout_s) if timeout_s is not None else int(self.heavy_job_timeout_s)
        ctx = self._enter_heavy_job(job_name=job_name, timeout_s=timeout)

        try:
            stream = fn()
            if stream is None:
                return

            if isinstance(stream, (str, bytes, bytearray)):
                yield stream
                return

            for item in stream:
                yield item
        finally:
            self._exit_heavy_job(job_name=job_name, ctx=ctx)


_GOVERNOR: MemoryGovernor | None = None
_GOVERNOR_LOCK = threading.Lock()


def get_memory_governor(config: dict[str, Any] | None = None) -> MemoryGovernor:
    """Get singleton memory governor and refresh config thresholds."""
    global _GOVERNOR
    with _GOVERNOR_LOCK:
        if _GOVERNOR is None:
            _GOVERNOR = MemoryGovernor(config=config or {})
        else:
            _GOVERNOR.update_config(config or {})
        return _GOVERNOR
