#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.tokenization import sha256_file  # noqa: E402


def main() -> None:
    config = yaml.safe_load(
        (PROJECT / "configs/pretraining_corpus.yaml").read_text(encoding="utf-8")
    )
    rows = []
    for filename in config["train_files"]:
        path = PROJECT / config["local_raw_dir"] / filename
        expected = config["source_lfs_sha256"][filename]
        actual = sha256_file(path)
        rows.append(
            {
                "source_file": filename,
                "path": str(path.relative_to(PROJECT)),
                "bytes": path.stat().st_size,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    report = {"status": "PASS" if all(row["match"] for row in rows) else "FAIL", "files": rows}
    output = PROJECT / "reports/model/pretraining_source_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
