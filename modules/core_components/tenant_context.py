"""Tenant resolution and validation helpers for multi-tenant SaaS mode."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


TENANT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")


class TenantResolutionError(ValueError):
    """Raised when tenant id cannot be resolved from request headers."""


@dataclass(frozen=True)
class TenantContext:
    """Resolved tenant context from an incoming request."""

    tenant_id: str
    header_name: str


def validate_tenant_id(tenant_id: str) -> bool:
    """Return True if tenant id is valid for filesystem-safe routing."""
    return bool(tenant_id and TENANT_ID_PATTERN.fullmatch(tenant_id))


def _extract_headers(request: Any) -> Mapping[str, str]:
    """Extract headers from Gradio/FastAPI request-like objects."""
    if request is None:
        return {}

    headers = getattr(request, "headers", None)
    if headers is None and hasattr(request, "request"):
        headers = getattr(request.request, "headers", None)

    if headers is None:
        return {}

    return headers


def resolve_tenant_id(
    request: Any,
    header_name: str = "X-Tenant-Id",
    required: bool = True,
    default_tenant_id: str | None = None,
) -> str | None:
    """Resolve tenant id from request headers and validate format.

    Args:
        request: Gradio/FastAPI request object.
        header_name: Header containing tenant id.
        required: If True, raises TenantResolutionError when missing/invalid.
        default_tenant_id: Optional fallback tenant id when header is missing.
    """
    headers = _extract_headers(request)

    tenant_id = None
    if headers:
        tenant_id = headers.get(header_name) or headers.get(header_name.lower())

    if not tenant_id:
        if default_tenant_id:
            fallback_tenant = str(default_tenant_id).strip()
            if validate_tenant_id(fallback_tenant):
                return fallback_tenant
            if required:
                raise TenantResolutionError(
                    f"Configured default tenant id '{fallback_tenant}' is invalid. "
                    f"Allowed pattern: {TENANT_ID_PATTERN.pattern}"
                )
            return None

        if required:
            raise TenantResolutionError(
                f"Missing required tenant header '{header_name}'. Configure your reverse proxy to inject it."
            )
        return None

    tenant_id = str(tenant_id).strip()
    if not validate_tenant_id(tenant_id):
        if required:
            raise TenantResolutionError(
                f"Invalid tenant id '{tenant_id}'. Allowed pattern: {TENANT_ID_PATTERN.pattern}"
            )
        return None

    return tenant_id


def resolve_tenant_context(
    request: Any,
    header_name: str = "X-Tenant-Id",
    required: bool = True,
    default_tenant_id: str | None = None,
) -> TenantContext | None:
    """Resolve request into a validated TenantContext."""
    tenant_id = resolve_tenant_id(
        request=request,
        header_name=header_name,
        required=required,
        default_tenant_id=default_tenant_id,
    )
    if not tenant_id:
        return None
    return TenantContext(tenant_id=tenant_id, header_name=header_name)


def tenant_error_message(header_name: str) -> str:
    """Human-readable error to present in UI controls."""
    return (
        f"Missing or invalid tenant header '{header_name}'. "
        "Please access this app through the configured reverse proxy/auth gateway."
    )
