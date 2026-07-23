#!/usr/bin/env python3
"""Validate paired Q1 figures, captions, source data, hashes, and raster quality."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "figures/q1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = ROOT / "figure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    figures: list[dict[str, Any]] = manifest["figures"]
    checks: list[dict[str, Any]] = []

    def add(figure_id: str, check: str, passed: bool, evidence: str) -> None:
        checks.append(
            {
                "figure_id": figure_id,
                "check": check,
                "passed": bool(passed),
                "evidence": evidence,
            }
        )

    ids = [row["id"] for row in figures]
    add("bundle", "unique figure IDs", len(ids) == len(set(ids)), f"{len(ids)} entries")
    add(
        "bundle",
        "minimum figure count",
        len(figures) >= int(manifest["minimum_required"]),
        f"{len(figures)}/{manifest['minimum_required']}",
    )
    captions_path = ROOT / "FIGURE_CAPTIONS.md"
    captions = captions_path.read_text(encoding="utf-8") if captions_path.exists() else ""

    for row in figures:
        figure_id = str(row["id"])
        png = PROJECT / row["png"]
        pdf = PROJECT / row["pdf"]
        add(figure_id, "paired files", png.exists() and pdf.exists(), f"{png.name}; {pdf.name}")
        if not png.exists() or not pdf.exists():
            continue
        add(figure_id, "nontrivial file size", png.stat().st_size > 20_000 and pdf.stat().st_size > 3_000, f"PNG {png.stat().st_size}; PDF {pdf.stat().st_size} bytes")
        add(figure_id, "recorded hashes", sha256(png) == row.get("png_sha256") and sha256(pdf) == row.get("pdf_sha256"), "SHA-256 matches manifest")
        with Image.open(png) as image:
            width, height = image.size
            dpi = image.info.get("dpi", (0, 0))
            effective_dpi = min(float(dpi[0]), float(dpi[1])) if len(dpi) >= 2 else 0
            add(figure_id, "raster dimensions", width >= 1600 and height >= 900, f"{width}×{height} pixels")
            add(figure_id, "raster DPI", effective_dpi >= 590, f"{effective_dpi:.1f} DPI")
        with pdf.open("rb") as handle:
            magic = handle.read(5)
        add(figure_id, "valid PDF header", magic == b"%PDF-", repr(magic))
        add(figure_id, "caption indexed", figure_id in captions and len(str(row.get("caption", ""))) >= 40, str(row.get("title", "")))
        source_data = row.get("source_data")
        diagrams = {"fig_architecture", "fig_protocol_flow"}
        source_ok = bool(source_data and (PROJECT / source_data).exists()) or figure_id in diagrams
        add(figure_id, "source data or method specification", source_ok, str(source_data or "method diagram"))
        add(
            figure_id,
            "evidence label",
            row.get("evidence")
            in {"development_only", "prior_test_descriptive", "locked_test"},
            str(row.get("evidence")),
        )

    failed = [row for row in checks if not row["passed"]]
    result = {
        "status": "PASS" if not failed else "FAIL",
        "figure_pairs": len(figures),
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "failed_checks": failed,
        "checks": checks,
    }
    json_path = ROOT / "VALIDATION_REPORT.json"
    md_path = ROOT / "VALIDATION_REPORT.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Q1 figure bundle validation",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Figure pairs: **{len(figures)}**  ",
        f"Checks passed: **{result['checks_passed']}/{result['checks_total']}**",
        "",
    ]
    if failed:
        lines.extend(["## Failed checks", ""])
        lines.extend(f"- `{row['figure_id']}` — {row['check']}: {row['evidence']}" for row in failed)
    else:
        lines.append("All manifest, pairing, hash, raster, PDF, caption, source-data, and evidence-label checks passed.")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": f"{result['checks_passed']}/{result['checks_total']}", "report": str(md_path)}, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
