import time

from modules.core_components.runtime.memory_governor import MemorySnapshot
from modules.core_components.runtime.resource_monitor import build_resource_monitor_payload


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
        "active_heavy_job_names": (),
        "admitted_jobs_total": 0,
        "rejected_jobs_total": 0,
        "completed_jobs_total": 0,
    }
    data.update(overrides)
    return MemorySnapshot(**data)


def _metric(payload, title):
    for metric in payload["metrics"]:
        if metric["title"] == title:
            return metric
    raise AssertionError(f"Missing metric '{title}'")


def test_resource_monitor_connected_tenant_shows_quota_usage():
    payload = build_resource_monitor_payload(
        _snapshot(),
        memory_max_rss_pct=70.0,
        memory_min_available_mb=2048,
        memory_max_gpu_reserved_pct=90.0,
        max_active_heavy_jobs=1,
        tenant_id="tenant_a",
        tenant_usage_summary={"used_bytes": 1024**3, "limit_bytes": 5 * 1024**3, "percent": 20.0},
        tenant_unresolved_reason=None,
    )
    metric = _metric(payload, "Tenant Quota Usage")
    assert "1.0 GB / 5.0 GB (20.0%)" in metric["value"]
    assert "Tenant: tenant_a" in metric["detail"]


def test_resource_monitor_missing_tenant_shows_unresolved_reason():
    payload = build_resource_monitor_payload(
        _snapshot(),
        memory_max_rss_pct=70.0,
        memory_min_available_mb=2048,
        memory_max_gpu_reserved_pct=90.0,
        max_active_heavy_jobs=1,
        tenant_id=None,
        tenant_usage_summary=None,
        tenant_unresolved_reason="Missing tenant header 'X-Tenant-Id' and no default tenant fallback.",
    )
    metric = _metric(payload, "Tenant Quota Usage")
    assert metric["value"] == "Tenant unresolved"
    assert "Missing tenant header 'X-Tenant-Id'" in metric["detail"]


def test_resource_monitor_no_cuda_shows_vram_na():
    payload = build_resource_monitor_payload(
        _snapshot(gpu_total_bytes=0, gpu_allocated_bytes=0, gpu_reserved_bytes=0),
        memory_max_rss_pct=70.0,
        memory_min_available_mb=2048,
        memory_max_gpu_reserved_pct=90.0,
        max_active_heavy_jobs=1,
        tenant_id="tenant_a",
        tenant_usage_summary={"used_bytes": 0, "limit_bytes": 5 * 1024**3, "percent": 0.0},
        tenant_unresolved_reason=None,
    )
    metric = _metric(payload, "App VRAM (allocated / reserved)")
    assert metric["value"] == "N/A"
    assert "CUDA not available" in metric["detail"]
