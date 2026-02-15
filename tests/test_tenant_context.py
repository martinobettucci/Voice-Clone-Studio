from types import SimpleNamespace

import pytest

from modules.core_components.tenant_context import (
    TenantResolutionError,
    resolve_tenant_id,
    validate_tenant_id,
)


def test_validate_tenant_id_accepts_expected_patterns():
    assert validate_tenant_id("tenant1")
    assert validate_tenant_id("tenant-1")
    assert validate_tenant_id("tenant_1")
    assert validate_tenant_id("tenant.1")


def test_validate_tenant_id_rejects_invalid_patterns():
    assert not validate_tenant_id("")
    assert not validate_tenant_id("../oops")
    assert not validate_tenant_id("tenant with spaces")
    assert not validate_tenant_id("_starts_with_underscore")


def test_resolve_tenant_id_missing_header_raises_when_required():
    req = SimpleNamespace(headers={})
    with pytest.raises(TenantResolutionError):
        resolve_tenant_id(req, header_name="X-Tenant-Id", required=True)


def test_resolve_tenant_id_reads_header_case_insensitive():
    req = SimpleNamespace(headers={"x-tenant-id": "tenant_a"})
    assert resolve_tenant_id(req, header_name="X-Tenant-Id", required=True) == "tenant_a"


def test_resolve_tenant_id_uses_default_tenant_when_header_missing():
    req = SimpleNamespace(headers={})
    assert resolve_tenant_id(
        req,
        header_name="X-Tenant-Id",
        required=True,
        default_tenant_id="legacy",
    ) == "legacy"


def test_resolve_tenant_id_rejects_invalid_default_tenant():
    req = SimpleNamespace(headers={})
    with pytest.raises(TenantResolutionError):
        resolve_tenant_id(
            req,
            header_name="X-Tenant-Id",
            required=True,
            default_tenant_id="../invalid",
        )


def test_resolve_tenant_id_invalid_value_raises():
    req = SimpleNamespace(headers={"X-Tenant-Id": "../../etc"})
    with pytest.raises(TenantResolutionError):
        resolve_tenant_id(req, header_name="X-Tenant-Id", required=True)
