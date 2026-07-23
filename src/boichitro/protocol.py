from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


CODE_PATTERNS = ("src/**/*.py", "tools/*.py")
CONFIG_PATTERNS = ("configs/**/*.yaml",)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def matched_files(project: Path, patterns: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in project.glob(pattern) if path.is_file())
    return sorted(paths, key=lambda path: path.relative_to(project).as_posix())


def file_manifest(project: Path, paths: Iterable[Path]) -> dict[str, str]:
    return {
        path.relative_to(project).as_posix(): sha256_file(path)
        for path in sorted(paths, key=lambda value: value.relative_to(project).as_posix())
    }


def manifest_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_hash))
    return digest.hexdigest()


def protocol_fingerprints(project: Path) -> dict[str, object]:
    code_files = file_manifest(project, matched_files(project, CODE_PATTERNS))
    config_files = file_manifest(project, matched_files(project, CONFIG_PATTERNS))
    return {
        "code_sha256": manifest_sha256(code_files),
        "config_sha256": manifest_sha256(config_files),
        "code_files": code_files,
        "config_files": config_files,
    }


def freeze_manifest_path(project: Path, protocol_id: str) -> Path:
    return project / "reports/protocol" / f"{protocol_id}_freeze.json"


def require_protocol_freeze(project: Path, protocol_id: str) -> dict:
    path = freeze_manifest_path(project, protocol_id)
    if not path.exists():
        raise RuntimeError(
            f"Locked evaluation forbidden before protocol freeze: missing {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN" or payload.get("protocol_id") != protocol_id:
        raise RuntimeError(f"Invalid protocol freeze manifest: {path}")
    current = protocol_fingerprints(project)
    for key in ("code_sha256", "config_sha256"):
        if payload.get(key) != current[key]:
            raise RuntimeError(
                f"Protocol changed after freeze ({key}); create a new protocol before test access"
            )
    return payload


def require_frozen_artifact(project: Path, freeze: dict, path: Path) -> str:
    relative = path.relative_to(project).as_posix()
    expected = freeze.get("selected_artifacts", {}).get(relative)
    if expected is None:
        raise RuntimeError(f"Artifact was not registered at protocol freeze: {relative}")
    current = sha256_file(path)
    if current != expected:
        raise RuntimeError(f"Artifact changed after protocol freeze: {relative}")
    return current
