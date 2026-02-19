"""Pure helpers for building resource monitor UI payloads."""

from __future__ import annotations

import time
from typing import Any

from modules.core_components.runtime.memory_governor import MemorySnapshot
from modules.core_components.tenant_storage import format_bytes


def _bytes_to_gib(value: int) -> float:
    return float(value) / float(1024 ** 3)


def _clamp_pct(value: float) -> float:
    return max(0.0, min(100.0, float(value)))


def _level_from_upper_limit(value: float, limit: float) -> str:
    if value >= limit:
        return "danger"
    if value >= limit * 0.85:
        return "warn"
    return "ok"


def _level_from_tenant_pct(value: float) -> str:
    return _level_from_upper_limit(value, 100.0)


def build_resource_monitor_payload(
    snapshot: MemorySnapshot,
    *,
    memory_max_rss_pct: float,
    memory_min_available_mb: int,
    memory_max_gpu_reserved_pct: float,
    max_active_heavy_jobs: int,
    tenant_id: str | None,
    tenant_usage_summary: dict[str, Any] | None,
    tenant_unresolved_reason: str | None,
) -> dict[str, Any]:
    """Build a UI-agnostic payload for the resource monitor."""
    rss_gib = _bytes_to_gib(snapshot.rss_bytes)
    min_available_bytes = int(memory_min_available_mb) * 1024 * 1024
    available_ram_pct = (
        (snapshot.system_available_bytes / snapshot.system_total_bytes) * 100.0
        if snapshot.system_total_bytes
        else 0.0
    )
    active_names = ", ".join(snapshot.active_heavy_job_names) if snapshot.active_heavy_job_names else "none"

    status_class = "ok"
    status_label = "Healthy"

    hard_issues = []
    if snapshot.rss_pct_of_total > memory_max_rss_pct:
        hard_issues.append("App RAM is above configured admission threshold.")
    if snapshot.system_available_bytes < min_available_bytes:
        hard_issues.append("System free RAM is below configured minimum.")
    if snapshot.gpu_total_bytes > 0 and snapshot.gpu_reserved_pct_of_total > memory_max_gpu_reserved_pct:
        hard_issues.append("App VRAM reserved is above configured admission threshold.")

    soft_issues = []
    if snapshot.active_heavy_jobs >= max_active_heavy_jobs:
        soft_issues.append("A heavy job is currently running.")
    if tenant_unresolved_reason:
        soft_issues.append("Tenant could not be resolved for quota usage.")

    if hard_issues:
        status_class = "danger"
        status_label = "Action Needed"
    elif soft_issues:
        status_class = "warn"
        status_label = "Watch"

    metrics: list[dict[str, Any]] = [
        {
            "title": "Running Heavy Jobs",
            "value": f"{snapshot.active_heavy_jobs} ({active_names})",
            "detail": (
                f"Scheduler mode: strict single-lane (max {max_active_heavy_jobs} heavy job at a time). "
                f"Session totals: admitted {snapshot.admitted_jobs_total}, completed {snapshot.completed_jobs_total}, "
                f"rejected {snapshot.rejected_jobs_total}."
            ),
            "pct": (
                (snapshot.active_heavy_jobs / max_active_heavy_jobs) * 100.0
                if max_active_heavy_jobs > 0
                else 0.0
            ),
            "level": "warn" if snapshot.active_heavy_jobs > 0 else "ok",
        },
        {
            "title": "App RAM (RSS)",
            "value": f"{rss_gib:.2f} GB ({snapshot.rss_pct_of_total:.1f}% of system RAM)",
            "detail": f"Admission threshold: {memory_max_rss_pct:.1f}% max RSS.",
            "pct": snapshot.rss_pct_of_total,
            "level": _level_from_upper_limit(snapshot.rss_pct_of_total, memory_max_rss_pct),
        },
    ]

    if snapshot.gpu_total_bytes > 0:
        metrics.append(
            {
                "title": "App VRAM (allocated / reserved)",
                "value": (
                    f"{_bytes_to_gib(snapshot.gpu_allocated_bytes):.2f} / "
                    f"{_bytes_to_gib(snapshot.gpu_reserved_bytes):.2f} GB "
                    f"({snapshot.gpu_reserved_pct_of_total:.1f}% of device total)"
                ),
                "detail": (
                    "PyTorch allocator stats for this process only. "
                    f"Admission threshold: {memory_max_gpu_reserved_pct:.1f}% reserved."
                ),
                "pct": snapshot.gpu_reserved_pct_of_total,
                "level": _level_from_upper_limit(snapshot.gpu_reserved_pct_of_total, memory_max_gpu_reserved_pct),
            }
        )
    else:
        metrics.append(
            {
                "title": "App VRAM (allocated / reserved)",
                "value": "N/A",
                "detail": "CUDA not available.",
                "pct": None,
                "level": "warn",
            }
        )

    if tenant_id and tenant_usage_summary:
        used = int(tenant_usage_summary.get("used_bytes", 0))
        limit = int(tenant_usage_summary.get("limit_bytes", 0))
        pct = float(tenant_usage_summary.get("percent", 0.0))
        metrics.append(
            {
                "title": "Tenant Quota Usage",
                "value": f"{format_bytes(used)} / {format_bytes(limit)} ({pct:.1f}%)",
                "detail": f"Tenant: {tenant_id}. Quota tracks samples + datasets only.",
                "pct": pct,
                "level": _level_from_tenant_pct(pct),
            }
        )
    else:
        reason = tenant_unresolved_reason or "Tenant resolution unavailable for this request."
        metrics.append(
            {
                "title": "Tenant Quota Usage",
                "value": "Tenant unresolved",
                "detail": reason,
                "pct": None,
                "level": "warn",
            }
        )

    return {
        "status_class": status_class,
        "status_label": status_label,
        "updated_at_text": time.strftime("%H:%M:%S", time.localtime(snapshot.timestamp)),
        "metrics": [
            {
                **metric,
                "pct": None if metric["pct"] is None else _clamp_pct(metric["pct"]),
            }
            for metric in metrics
        ],
        "guide_hint": "See Help Guide > Resource Monitor for definitions and formulas.",
        "system_available_pct": _clamp_pct(available_ram_pct),
    }
