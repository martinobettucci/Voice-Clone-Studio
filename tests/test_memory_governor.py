import threading
import time

import pytest

import modules.core_components.runtime.memory_governor as memory_governor_module
from modules.core_components.runtime.memory_governor import (
    MemoryAdmissionError,
    MemoryGovernor,
    MemorySnapshot,
)


def _snapshot(**overrides):
    now = time.time()
    data = {
        "timestamp": now,
        "pid": 1234,
        "rss_bytes": 2 * 1024 * 1024 * 1024,
        "rss_pct_of_total": 25.0,
        "system_available_bytes": 10 * 1024 * 1024 * 1024,
        "system_total_bytes": 32 * 1024 * 1024 * 1024,
        "gpu_allocated_bytes": 0,
        "gpu_reserved_bytes": 0,
        "gpu_total_bytes": 0,
        "gpu_reserved_pct_of_total": 0.0,
        "active_heavy_jobs": 0,
    }
    data.update(overrides)
    return MemorySnapshot(**data)


def test_governor_rejects_second_concurrent_heavy_job():
    governor = MemoryGovernor(
        {
            "memory_max_rss_pct": 99.0,
            "memory_min_available_mb": 1,
            "memory_max_gpu_reserved_pct": 99.0,
        }
    )

    started = threading.Event()
    release = threading.Event()

    def long_job():
        started.set()
        release.wait(timeout=2.0)
        return "done"

    t = threading.Thread(target=lambda: governor.run_heavy("job_1", long_job), daemon=True)
    t.start()
    assert started.wait(timeout=1.0), "first heavy job did not start in time"

    with pytest.raises(MemoryAdmissionError) as exc_info:
        governor.run_heavy("job_2", lambda: "never")

    release.set()
    t.join(timeout=2.0)
    assert "one heavy task at a time" in str(exc_info.value)


def test_governor_rejects_when_available_ram_below_threshold():
    governor = MemoryGovernor(
        {
            "memory_max_rss_pct": 99.0,
            "memory_min_available_mb": 10_000_000,  # force rejection on normal machines
            "memory_max_gpu_reserved_pct": 99.0,
        }
    )

    with pytest.raises(MemoryAdmissionError) as exc_info:
        governor.run_heavy("too_low_ram", lambda: "never")

    assert "System memory is too low" in str(exc_info.value)


def test_governor_runs_cleanup_hooks_on_success_and_failure(monkeypatch):
    governor = MemoryGovernor(
        {
            "memory_max_rss_pct": 99.0,
            "memory_min_available_mb": 1,
            "memory_max_gpu_reserved_pct": 99.0,
        }
    )

    calls = {"gc": 0, "cache": 0}

    def _fake_gc():
        calls["gc"] += 1
        return 0

    def _fake_cache():
        calls["cache"] += 1

    monkeypatch.setattr(memory_governor_module.gc, "collect", _fake_gc)
    monkeypatch.setattr(memory_governor_module, "empty_device_cache", _fake_cache)

    assert governor.run_heavy("ok_job", lambda: "ok") == "ok"

    def _failing_job():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        governor.run_heavy("failing_job", _failing_job)

    assert calls["gc"] == 2
    assert calls["cache"] == 2


def test_governor_reject_reason_reports_gpu_threshold():
    governor = MemoryGovernor(
        {
            "memory_max_rss_pct": 99.0,
            "memory_min_available_mb": 1,
            "memory_max_gpu_reserved_pct": 80.0,
        }
    )
    snapshot = _snapshot(
        gpu_total_bytes=10 * 1024 * 1024 * 1024,
        gpu_reserved_bytes=9 * 1024 * 1024 * 1024,
        gpu_reserved_pct_of_total=90.0,
    )

    reason = governor.reject_reason(snapshot)
    assert reason is not None
    assert "GPU memory reservation" in reason
