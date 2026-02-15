from pathlib import Path

from modules.core_components.tenant_storage import BaseStoragePaths, TenantStorageService


class _Req:
    def __init__(self, tenant_id: str):
        self.headers = {"X-Tenant-Id": tenant_id}


def test_bulk_add_dataset_files_is_tenant_scoped(tmp_path: Path):
    svc = TenantStorageService(
        base_paths=BaseStoragePaths(
            samples_dir=tmp_path / "samples",
            datasets_dir=tmp_path / "datasets",
            output_dir=tmp_path / "output",
            trained_models_dir=tmp_path / "models",
            temp_dir=tmp_path / "temp",
        ),
        tenant_header_name="X-Tenant-Id",
        tenant_file_limit_mb=10,
        tenant_media_quota_gb=1,
    )

    src = tmp_path / "clip.wav"
    src.write_bytes(b"123")

    paths_a = svc.get_tenant_paths(_Req("a"), required=True)
    paths_b = svc.get_tenant_paths(_Req("b"), required=True)

    saved = svc.bulk_add_dataset_files(paths_a, "dataset_1", [src])
    assert len(saved) == 1
    assert saved[0].exists()

    files_a = svc.list_dataset_files(paths_a, "dataset_1")
    files_b = svc.list_dataset_files(paths_b, "dataset_1")

    assert files_a == [saved[0].name]
    assert files_b == []
