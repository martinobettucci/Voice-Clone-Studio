from pathlib import Path

import pytest

from modules.core_components.tenant_storage import (
    BaseStoragePaths,
    TenantStorageService,
    collision_safe_path,
    sanitize_filename,
    safe_join,
)


class _Req:
    def __init__(self, tenant_id: str | None):
        self.headers = {}
        if tenant_id is not None:
            self.headers["X-Tenant-Id"] = tenant_id


def _base(tmp_path: Path) -> BaseStoragePaths:
    return BaseStoragePaths(
        samples_dir=tmp_path / "samples",
        datasets_dir=tmp_path / "datasets",
        output_dir=tmp_path / "output",
        trained_models_dir=tmp_path / "models",
        temp_dir=tmp_path / "temp",
    )


def _service(tmp_path: Path, file_limit_mb: int = 1, quota_gb: int = 1):
    return TenantStorageService(
        base_paths=_base(tmp_path),
        tenant_header_name="X-Tenant-Id",
        tenant_file_limit_mb=file_limit_mb,
        tenant_media_quota_gb=quota_gb,
    )


def test_sanitize_filename_and_safe_join(tmp_path):
    assert sanitize_filename("my file?.wav") == "my_file_.wav"

    root = tmp_path / "root"
    root.mkdir()
    p = safe_join(root, "a", "b.txt")
    assert str(p).startswith(str(root.resolve()))

    with pytest.raises(ValueError):
        safe_join(root, "..", "escape.txt")


def test_collision_safe_path_adds_suffix(tmp_path):
    root = tmp_path / "x"
    root.mkdir()
    p1 = collision_safe_path(root, "clip.wav")
    p1.write_bytes(b"abc")
    p2 = collision_safe_path(root, "clip.wav")
    assert p2.name.startswith("clip_")
    assert p2.suffix == ".wav"


def test_validate_uploads_enforces_file_limit(tmp_path):
    svc = _service(tmp_path, file_limit_mb=1, quota_gb=1)
    req = _Req("tenant1")
    paths = svc.get_tenant_paths(req, required=True)

    big = tmp_path / "big.wav"
    big.write_bytes(b"0" * (2 * 1024 * 1024))

    ok, msg = svc.validate_uploads(paths, [big])
    assert not ok
    assert "per-file limit" in msg


def test_validate_uploads_enforces_quota(tmp_path):
    svc = _service(tmp_path, file_limit_mb=5, quota_gb=1)
    req = _Req("tenant1")
    paths = svc.get_tenant_paths(req, required=True)

    # Force very low quota for test determinism without large fixture files.
    svc.tenant_media_quota_gb = 0

    incoming = tmp_path / "new.wav"
    incoming.write_bytes(b"0" * (512 * 1024))

    ok, msg = svc.validate_uploads(paths, [incoming])
    assert not ok
    assert "quota" in msg.lower()


def test_validate_generated_sizes_enforces_file_limit(tmp_path):
    svc = _service(tmp_path, file_limit_mb=1, quota_gb=1)
    paths = svc.get_tenant_paths(_Req("tenant1"), required=True)

    ok, msg = svc.validate_generated_sizes(paths, [2 * 1024 * 1024], label="Generated media")
    assert not ok
    assert "per-file limit" in msg


def test_validate_generated_sizes_enforces_quota(tmp_path):
    svc = _service(tmp_path, file_limit_mb=10, quota_gb=1)
    paths = svc.get_tenant_paths(_Req("tenant1"), required=True)

    svc.tenant_media_quota_gb = 0
    ok, msg = svc.validate_generated_sizes(paths, [1024], label="Generated media")
    assert not ok
    assert "quota" in msg.lower()


def test_tenant_isolation_between_dataset_lists(tmp_path):
    svc = _service(tmp_path, file_limit_mb=5, quota_gb=1)
    paths_a = svc.get_tenant_paths(_Req("tenant_a"), required=True)
    paths_b = svc.get_tenant_paths(_Req("tenant_b"), required=True)

    folder = paths_a.datasets_dir / "set1"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "a.wav").write_bytes(b"abc")

    assert svc.list_dataset_files(paths_a, "set1") == ["a.wav"]
    assert svc.list_dataset_folders(paths_b) == []


def test_default_tenant_fallback_is_used_when_header_missing(tmp_path):
    svc = TenantStorageService(
        base_paths=_base(tmp_path),
        tenant_header_name="X-Tenant-Id",
        tenant_file_limit_mb=5,
        tenant_media_quota_gb=1,
        default_tenant_id="legacy",
    )
    paths = svc.get_tenant_paths(_Req(None), required=True)
    assert paths.tenant_id == "legacy"
