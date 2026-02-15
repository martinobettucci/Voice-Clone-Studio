"""Tenant-scoped filesystem helpers, quotas, and dataset/sample operations."""

from __future__ import annotations

from dataclasses import dataclass
import os
import shutil
from pathlib import Path
from typing import Iterable

from .tenant_context import TenantContext, resolve_tenant_context


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


@dataclass(frozen=True)
class TenantPaths:
    """Tenant-resolved paths for all user-generated resources."""

    tenant_id: str
    samples_dir: Path
    datasets_dir: Path
    output_dir: Path
    trained_models_dir: Path
    temp_dir: Path


@dataclass(frozen=True)
class BaseStoragePaths:
    """Base roots from config before tenant isolation is applied."""

    samples_dir: Path
    datasets_dir: Path
    output_dir: Path
    trained_models_dir: Path
    temp_dir: Path


def _tenant_path(base_dir: Path, tenant_id: str) -> Path:
    return base_dir / "tenants" / tenant_id


def get_tenant_paths(base_dirs: BaseStoragePaths, tenant_id: str) -> TenantPaths:
    """Resolve tenant-specific storage paths from configured base dirs."""
    return TenantPaths(
        tenant_id=tenant_id,
        samples_dir=_tenant_path(base_dirs.samples_dir, tenant_id),
        datasets_dir=_tenant_path(base_dirs.datasets_dir, tenant_id),
        output_dir=_tenant_path(base_dirs.output_dir, tenant_id),
        trained_models_dir=_tenant_path(base_dirs.trained_models_dir, tenant_id),
        temp_dir=_tenant_path(base_dirs.temp_dir, tenant_id),
    )


def ensure_tenant_dirs(paths: TenantPaths) -> None:
    """Create tenant directories if missing."""
    for folder in (
        paths.samples_dir,
        paths.datasets_dir,
        paths.output_dir,
        paths.trained_models_dir,
        paths.temp_dir,
    ):
        folder.mkdir(parents=True, exist_ok=True)


def sanitize_filename(value: str, keep_extension: bool = True) -> str:
    """Sanitize filename to a filesystem-safe representation."""
    value = (value or "").strip()
    if not value:
        return "file"

    suffix = ""
    stem = value
    if keep_extension:
        p = Path(value)
        suffix = p.suffix
        stem = p.stem

    safe = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in stem)
    safe = safe.strip("._") or "file"

    if keep_extension and suffix:
        safe_suffix = "".join(c for c in suffix if c.isalnum() or c == ".").lower()
        if not safe_suffix.startswith("."):
            safe_suffix = "." + safe_suffix
        return safe + safe_suffix

    return safe


def safe_join(root: Path, *parts: str) -> Path:
    """Join paths safely, preventing traversal outside root."""
    root_resolved = root.resolve()
    candidate = root.joinpath(*parts).resolve()
    if os.path.commonpath([str(root_resolved), str(candidate)]) != str(root_resolved):
        raise ValueError("Path traversal attempt detected")
    return candidate


def collision_safe_path(root: Path, filename: str) -> Path:
    """Get unique non-colliding file path by suffixing `_N` when needed."""
    candidate = safe_join(root, filename)
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for idx in range(1, 10000):
        named = safe_join(root, f"{stem}_{idx}{suffix}")
        if not named.exists():
            return named

    raise RuntimeError("Could not allocate unique filename")


def is_allowed_media(filename: str) -> bool:
    return Path(filename).suffix.lower() in MEDIA_EXTENSIONS


def _dir_size_bytes(directory: Path) -> int:
    total = 0
    if not directory.exists():
        return total
    for path in directory.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                pass
    return total


class TenantStorageService:
    """Stateful service for tenant header resolution + storage operations."""

    def __init__(
        self,
        base_paths: BaseStoragePaths,
        tenant_header_name: str = "X-Tenant-Id",
        tenant_file_limit_mb: int = 200,
        tenant_media_quota_gb: int = 5,
        default_tenant_id: str | None = None,
    ) -> None:
        self.base_paths = base_paths
        self.tenant_header_name = tenant_header_name
        self.tenant_file_limit_mb = int(tenant_file_limit_mb)
        self.tenant_media_quota_gb = int(tenant_media_quota_gb)
        self.default_tenant_id = (default_tenant_id or "").strip() or None

    @property
    def file_limit_bytes(self) -> int:
        return self.tenant_file_limit_mb * 1024 * 1024

    @property
    def media_quota_bytes(self) -> int:
        return self.tenant_media_quota_gb * 1024 * 1024 * 1024

    def update_limits(self, file_limit_mb: int, media_quota_gb: int) -> None:
        self.tenant_file_limit_mb = int(file_limit_mb)
        self.tenant_media_quota_gb = int(media_quota_gb)

    def update_header_name(self, header_name: str) -> None:
        self.tenant_header_name = header_name or "X-Tenant-Id"

    def update_default_tenant(self, default_tenant_id: str | None) -> None:
        self.default_tenant_id = (default_tenant_id or "").strip() or None

    def resolve_tenant_context(self, request, required: bool = True) -> TenantContext | None:
        return resolve_tenant_context(
            request=request,
            header_name=self.tenant_header_name,
            required=required,
            default_tenant_id=self.default_tenant_id,
        )

    def get_tenant_paths(self, request, required: bool = True) -> TenantPaths | None:
        context = self.resolve_tenant_context(request=request, required=required)
        if context is None:
            return None
        paths = get_tenant_paths(self.base_paths, context.tenant_id)
        ensure_tenant_dirs(paths)
        return paths

    def get_media_usage_bytes(self, paths: TenantPaths) -> int:
        return _dir_size_bytes(paths.samples_dir) + _dir_size_bytes(paths.datasets_dir)

    def compute_usage_summary(self, paths: TenantPaths) -> dict:
        used = self.get_media_usage_bytes(paths)
        limit = self.media_quota_bytes
        pct = 0.0 if limit <= 0 else (used / limit) * 100.0
        return {
            "used_bytes": used,
            "limit_bytes": limit,
            "percent": min(max(pct, 0.0), 100.0),
        }

    def validate_uploads(self, paths: TenantPaths, file_paths: Iterable[str | Path]) -> tuple[bool, str]:
        normalized = []
        for path in file_paths or []:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return False, f"Upload source not found: {p}"
            if not is_allowed_media(p.name):
                return False, f"Unsupported media format: {p.name}"
            size = p.stat().st_size
            if size > self.file_limit_bytes:
                return False, (
                    f"File '{p.name}' exceeds per-file limit of {self.tenant_file_limit_mb} MB"
                )
            normalized.append(p)

        if not normalized:
            return False, "No files selected"

        incoming = sum(p.stat().st_size for p in normalized)
        used = self.get_media_usage_bytes(paths)
        if used + incoming > self.media_quota_bytes:
            return False, (
                f"Upload would exceed tenant media quota ({self.tenant_media_quota_gb} GB). "
                f"Current usage: {used / (1024**3):.2f} GB"
            )

        return True, "ok"

    def validate_generated_sizes(
        self,
        paths: TenantPaths,
        file_sizes_bytes: Iterable[int],
        label: str = "Generated media",
        reclaimed_bytes: int = 0,
    ) -> tuple[bool, str]:
        """Validate per-file and tenant quota limits for in-app generated media."""
        sizes = [max(int(s), 0) for s in (file_sizes_bytes or [])]
        if not sizes:
            return False, "No generated files to save"

        for idx, size in enumerate(sizes, start=1):
            if size > self.file_limit_bytes:
                return False, (
                    f"{label} item #{idx} exceeds per-file limit of {self.tenant_file_limit_mb} MB"
                )

        incoming = sum(sizes)
        used = max(self.get_media_usage_bytes(paths) - max(int(reclaimed_bytes), 0), 0)
        if used + incoming > self.media_quota_bytes:
            return False, (
                f"{label} would exceed tenant media quota ({self.tenant_media_quota_gb} GB). "
                f"Current usage: {used / (1024**3):.2f} GB"
            )

        return True, "ok"

    def list_dataset_folders(self, paths: TenantPaths) -> list[str]:
        if not paths.datasets_dir.exists():
            return []
        return sorted([d.name for d in paths.datasets_dir.iterdir() if d.is_dir()])

    def create_dataset_folder(self, paths: TenantPaths, folder_name: str) -> Path:
        safe = sanitize_filename(folder_name, keep_extension=False)
        target = safe_join(paths.datasets_dir, safe)
        target.mkdir(parents=True, exist_ok=False)
        return target

    def delete_dataset_folder(self, paths: TenantPaths, folder_name: str) -> None:
        safe = sanitize_filename(folder_name, keep_extension=False)
        target = safe_join(paths.datasets_dir, safe)
        if not target.exists() or not target.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder_name}")
        shutil.rmtree(target)

    def list_dataset_files(self, paths: TenantPaths, folder: str | None = None) -> list[str]:
        scan_dir = paths.datasets_dir
        if folder and folder not in {"(No folders)", "(Select Dataset)"}:
            scan_dir = safe_join(paths.datasets_dir, sanitize_filename(folder, keep_extension=False))

        if not scan_dir.exists():
            return []

        files = sorted(
            [p for p in scan_dir.iterdir() if p.is_file() and p.suffix.lower() in {".wav", ".mp3"}],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return [p.name for p in files]

    def bulk_add_dataset_files(
        self,
        paths: TenantPaths,
        folder: str,
        file_paths: Iterable[str | Path],
    ) -> list[Path]:
        safe_folder = sanitize_filename(folder, keep_extension=False)
        target_dir = safe_join(paths.datasets_dir, safe_folder)
        target_dir.mkdir(parents=True, exist_ok=True)

        ok, msg = self.validate_uploads(paths, file_paths)
        if not ok:
            raise ValueError(msg)

        saved = []
        for src in file_paths:
            src_path = Path(src)
            filename = sanitize_filename(src_path.name, keep_extension=True)
            dst = collision_safe_path(target_dir, filename)
            shutil.copy2(src_path, dst)
            saved.append(dst)
        return saved

    def list_samples(self, paths: TenantPaths) -> list[Path]:
        if not paths.samples_dir.exists():
            return []
        return sorted(paths.samples_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)

    def delete_samples(self, paths: TenantPaths, sample_names: Iterable[str]) -> int:
        deleted = 0
        for name in sample_names:
            stem = sanitize_filename(Path(name).stem, keep_extension=False)
            wav = safe_join(paths.samples_dir, f"{stem}.wav")
            meta = safe_join(paths.samples_dir, f"{stem}.json")
            if wav.exists():
                wav.unlink()
                deleted += 1
            if meta.exists():
                meta.unlink()
            for cache_name in (f"{stem}_0.6B.pt", f"{stem}_1.7B.pt", f"{stem}_luxtts.pt"):
                cache = safe_join(paths.samples_dir, cache_name)
                if cache.exists():
                    cache.unlink()
        return deleted


def format_bytes(value: int) -> str:
    """Human-readable bytes formatter."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(max(value, 0))
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "0 B"
