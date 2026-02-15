#!/usr/bin/env python3
"""Migrate existing shared assets into tenant-scoped storage.

Default behavior copies data into `tenants/<tenant_id>` and leaves source intact.
Use `--move` to move instead of copy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def load_config(project_root: Path) -> dict:
    config_path = project_root / "config.json"
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def iter_source_items(base_dir: Path):
    if not base_dir.exists():
        return []
    items = []
    for child in base_dir.iterdir():
        if child.name == "tenants":
            continue
        items.append(child)
    return items


def migrate_base(base_dir: Path, tenant_id: str, move: bool):
    target = base_dir / "tenants" / tenant_id
    target.mkdir(parents=True, exist_ok=True)

    migrated = []
    skipped = []
    for src in iter_source_items(base_dir):
        dst = target / src.name
        if dst.exists():
            skipped.append((src, "already exists in tenant target"))
            continue

        if move:
            shutil.move(str(src), str(dst))
        else:
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        migrated.append((src, dst))

    return migrated, skipped


def main():
    parser = argparse.ArgumentParser(description="Migrate shared assets into tenant storage")
    parser.add_argument("--tenant", default="legacy", help="Target tenant id (default: legacy)")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument("--project-root", default=".", help="Project root path")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    cfg = load_config(project_root)

    folder_keys = [
        ("samples_folder", "samples"),
        ("datasets_folder", "datasets"),
        ("output_folder", "output"),
        ("trained_models_folder", "models"),
    ]

    print(f"Project root: {project_root}")
    print(f"Tenant target: {args.tenant}")
    print(f"Mode: {'move' if args.move else 'copy (non-destructive)'}")
    print()

    total_migrated = 0
    total_skipped = 0

    for key, default_name in folder_keys:
        rel = cfg.get(key, default_name)
        base_dir = project_root / rel
        migrated, skipped = migrate_base(base_dir, args.tenant, args.move)

        print(f"[{key}] {base_dir}")
        print(f"  migrated: {len(migrated)}")
        print(f"  skipped: {len(skipped)}")
        for src, reason in skipped[:5]:
            print(f"    - {src.name}: {reason}")

        total_migrated += len(migrated)
        total_skipped += len(skipped)

    print()
    print("Migration complete")
    print(f"Total migrated: {total_migrated}")
    print(f"Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
