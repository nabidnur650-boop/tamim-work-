#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]


def command(arguments: list[str]) -> str:
    result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()


def tree_hash(paths: list[Path]) -> tuple[str, list[dict[str, object]]]:
    digest = hashlib.sha256()
    entries = []
    files = sorted(
        file
        for root in paths
        for file in root.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts
    )
    for path in files:
        relative = path.relative_to(PROJECT).as_posix()
        content = path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_hash))
        entries.append({"path": relative, "sha256": file_hash, "bytes": len(content)})
    return digest.hexdigest(), entries


def main() -> None:
    output_dir = PROJECT / "reports/reproducibility"
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze = command([sys.executable, "-m", "pip", "freeze"])
    (output_dir / "environment_pip_freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    code_sha, files = tree_hash(
        [PROJECT / "src", PROJECT / "tools", PROJECT / "configs", PROJECT / "tests"]
    )
    cuda_properties = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        cuda_properties = {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device": cuda_properties,
        "nvidia_smi": command(["nvidia-smi"]),
        "memory": command(["free", "-h"]),
        "disk": command(["df", "-h", str(PROJECT)]),
        "git_head": command(["git", "-C", str(PROJECT), "rev-parse", "HEAD"]),
        "git_status": command(["git", "-C", str(PROJECT), "status", "--short"]),
        "code_tree_sha256": code_sha,
        "code_files": files,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "code_files"}, indent=2))


if __name__ == "__main__":
    main()
