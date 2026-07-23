#!/usr/bin/env python3
"""Fetch immutable, licensed external sources for the Boichitro data build.

Mendeley releases are pinned by dataset ID and version. Hugging Face releases
are pinned by commit SHA. For the very large ASR Parquet release, only the two
text metadata columns are materialized through range reads; upstream shard
hashes remain recorded in the snapshot metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT / "configs" / "external_sources.yaml"
RAW_ROOT = PROJECT / "data" / "external" / "raw"
PROCESSED_ROOT = PROJECT / "data" / "external" / "processed"
REPORT_ROOT = PROJECT / "reports"
USER_AGENT = "boichitro-moe-data-audit/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def get_json(url: str, accept: str = "application/json") -> Any:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=300) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def compact_hf_info(info: Any) -> dict[str, Any]:
    siblings = []
    for sibling in info.siblings or []:
        lfs = getattr(sibling, "lfs", None)
        siblings.append(
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs_sha256": getattr(lfs, "sha256", None) if lfs else None,
                "lfs_size": getattr(lfs, "size", None) if lfs else None,
            }
        )
    card = info.card_data.to_dict() if info.card_data else None
    return {
        "repo_id": info.id,
        "sha": info.sha,
        "private": info.private,
        "gated": info.gated,
        "created_at": info.created_at.isoformat() if info.created_at else None,
        "last_modified": (
            info.last_modified.isoformat() if info.last_modified else None
        ),
        "downloads": info.downloads,
        "likes": info.likes,
        "tags": info.tags or [],
        "card_data": card,
        "files": siblings,
    }


def fetch_mendeley(entry: dict[str, Any]) -> dict[str, Any]:
    source_id = entry["source_id"]
    dataset_id = entry["dataset_id"]
    version = int(entry["version"])
    base = f"https://data.mendeley.com"
    # The public page API omits licence fields unless they are requested. The
    # anonymous datasets-v2 endpoint returns the complete snapshot plus the
    # explicitly selected licence object.
    metadata = get_json(
        f"{base}/api/datasets-v2/datasets/{dataset_id}"
        f"?fields=licence.*&version={version}"
    )
    files = get_json(
        f"{base}/public-api/datasets/{dataset_id}/files?folder_id=root&version={version}",
        accept="application/vnd.mendeley-public-dataset.1+json",
    )
    actual_license = (metadata.get("data_licence") or {}).get("short_name")
    if actual_license != entry["expected_license"]:
        raise RuntimeError(
            f"{source_id}: expected license {entry['expected_license']!r}, "
            f"received {actual_license!r}"
        )
    if metadata.get("version") != version:
        raise RuntimeError(f"{source_id}: Mendeley returned the wrong version")

    output_dir = RAW_ROOT / "mendeley" / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    file_by_name = {item["filename"]: item for item in files}
    downloaded = []
    for filename in entry.get("selected_files", []):
        if filename not in file_by_name:
            raise RuntimeError(f"{source_id}: selected file is absent: {filename}")
        remote = file_by_name[filename]
        details = remote["content_details"]
        destination = output_dir / filename
        if not destination.exists() or sha256_file(destination) != details["sha256_hash"]:
            download(details["download_url"], destination)
        actual_sha = sha256_file(destination)
        if actual_sha != details["sha256_hash"]:
            raise RuntimeError(
                f"{source_id}: SHA-256 mismatch for {filename}: {actual_sha}"
            )
        downloaded.append(
            {
                "filename": filename,
                "path": str(destination.relative_to(PROJECT)),
                "size": destination.stat().st_size,
                "sha256": actual_sha,
                "mendeley_file_id": remote["id"],
            }
        )

    snapshot = {
        "registry_entry": entry,
        "metadata": metadata,
        "remote_files": files,
        "downloaded_files": downloaded,
        "landing_page": f"https://data.mendeley.com/datasets/{dataset_id}/{version}",
    }
    write_json(output_dir / "snapshot_metadata.json", snapshot)
    return {
        "source_id": source_id,
        "provider": "mendeley",
        "decision": entry["decision"],
        "role": entry["role"],
        "license": actual_license,
        "doi": metadata.get("doi", {}).get("id"),
        "downloaded_files": downloaded,
    }


def extract_asr_text(entry: dict[str, Any], info: Any) -> dict[str, Any]:
    repo_id = entry["repo_id"]
    revision = entry["revision"]
    columns = entry["selected_columns"]
    fs = HfFileSystem()
    shards = sorted(
        sibling.rfilename
        for sibling in info.siblings or []
        if sibling.rfilename.endswith(".parquet")
    )
    tables = []
    for shard in shards:
        remote_path = f"datasets/{repo_id}@{revision}/{shard}"
        with fs.open(remote_path, "rb", block_size=1024 * 1024) as stream:
            table = pq.read_table(stream, columns=columns)
        split = "eval" if "/eval-" in f"/{shard}" else "train"
        table = table.append_column("split_original", pa.array([split] * len(table)))
        table = table.append_column("upstream_shard", pa.array([shard] * len(table)))
        table = table.append_column(
            "upstream_row_index", pa.array(range(len(table)), type=pa.int64())
        )
        tables.append(table)
    combined = pa.concat_tables(tables)
    frame = combined.to_pandas()
    frame["transcriptions"] = frame["transcriptions"].astype("string")
    frame["district"] = frame["district"].astype("string")
    frame = frame.sort_values(
        ["split_original", "upstream_shard", "upstream_row_index"],
        kind="stable",
    ).reset_index(drop=True)
    output_dir = PROCESSED_ROOT / "huggingface" / entry["source_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "transcripts_text_only.parquet"
    frame.to_parquet(destination, index=False, compression="zstd")
    return {
        "path": str(destination.relative_to(PROJECT)),
        "rows": len(frame),
        "sha256": sha256_file(destination),
        "district_counts": {
            str(key): int(value)
            for key, value in frame.groupby("district", dropna=False).size().items()
        },
        "split_counts": {
            str(key): int(value)
            for key, value in frame.groupby("split_original").size().items()
        },
        "upstream_shards": shards,
    }


def fetch_huggingface(entry: dict[str, Any], api: HfApi) -> dict[str, Any]:
    source_id = entry["source_id"]
    repo_id = entry["repo_id"]
    revision = entry["revision"]
    info = api.dataset_info(repo_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"{source_id}: Hugging Face returned the wrong commit")
    license_tags = {
        tag.removeprefix("license:")
        for tag in info.tags or []
        if tag.startswith("license:")
    }
    if entry["expected_license"] not in license_tags:
        raise RuntimeError(
            f"{source_id}: expected license tag {entry['expected_license']!r}; "
            f"received {sorted(license_tags)!r}"
        )

    output_dir = RAW_ROOT / "huggingface" / source_id
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for filename in entry.get("selected_files", []):
        cached = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )
        destination = output_dir / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        downloaded.append(
            {
                "filename": filename,
                "path": str(destination.relative_to(PROJECT)),
                "size": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        )
    processed = None
    if entry.get("selected_columns"):
        processed = extract_asr_text(entry, info)

    snapshot = {
        "registry_entry": entry,
        "metadata": compact_hf_info(info),
        "downloaded_files": downloaded,
        "processed_extract": processed,
        "landing_page": f"https://huggingface.co/datasets/{repo_id}/tree/{revision}",
    }
    write_json(output_dir / "snapshot_metadata.json", snapshot)
    return {
        "source_id": source_id,
        "provider": "huggingface",
        "decision": entry["decision"],
        "role": entry["role"],
        "license": entry["expected_license"],
        "revision": revision,
        "downloaded_files": downloaded,
        "processed_extract": processed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    registry = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for entry in registry["mendeley"]:
        print(f"Mendeley: {entry['source_id']}", flush=True)
        results.append(fetch_mendeley(entry))
    api = HfApi()
    for entry in registry["huggingface"]:
        print(f"Hugging Face: {entry['source_id']}", flush=True)
        results.append(fetch_huggingface(entry, api))
    report = {
        "registry_version": registry["registry_version"],
        "frozen_on": registry["frozen_on"],
        "sources": results,
        "huggingface_rejections": registry["huggingface_rejections"],
    }
    write_json(REPORT_ROOT / "external_source_acquisition.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
