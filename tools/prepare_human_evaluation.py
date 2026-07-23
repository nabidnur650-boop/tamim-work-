#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare blinded native-speaker evaluation packets.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/human_evaluation.yaml"
    )
    parser.add_argument("--allow-identical-rerun", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_rank(value: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).digest()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = PROJECT / "human_evaluation" / str(config["protocol_id"])
    manifest_path = output / "preparation_manifest.json"
    prediction_hashes = {
        system: sha256(PROJECT / relative_path)
        for system, relative_path in config["systems"].items()
    }
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("prediction_sha256") != prediction_hashes:
            raise RuntimeError("Refusing to overwrite human packets from different predictions")
        if not args.allow_identical_rerun:
            print(f"Human-evaluation packets already prepared: {output}")
            return

    systems = {}
    required = ["row_id", "semantic_group_id", "dialect", "source_id", "source", "reference", "prediction"]
    for system, relative_path in config["systems"].items():
        frame = pd.read_parquet(PROJECT / relative_path)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{system} predictions lack {sorted(missing)}")
        systems[system] = frame[required].copy()
    row_sets = {system: set(frame["row_id"].astype(str)) for system, frame in systems.items()}
    if len({frozenset(values) for values in row_sets.values()}) != 1:
        raise ValueError("Human-evaluation systems do not contain identical row IDs")

    reference = next(iter(systems.values()))
    sampled_indices = []
    for dialect, group in reference.groupby("dialect", sort=True, observed=True):
        ranked = sorted(
            group.index,
            key=lambda index: stable_rank(
                str(reference.loc[index, "row_id"]), int(config["sampling_seed"])
            ),
        )
        requested = int(config["examples_per_dialect"])
        if len(ranked) < requested:
            raise ValueError(f"Dialect {dialect} has only {len(ranked)} rows, need {requested}")
        sampled_indices.extend(ranked[:requested])
    sample = reference.loc[sampled_indices].copy()
    sampled_ids = set(sample["row_id"].astype(str))

    items = []
    master = []
    for system, frame in systems.items():
        indexed = frame.set_index(frame["row_id"].astype(str), drop=False)
        for row in sample.itertuples(index=False):
            prediction = indexed.loc[str(row.row_id), "prediction"]
            blind_id = hashlib.sha256(
                f"{config['protocol_id']}|{row.row_id}|{system}".encode("utf-8")
            ).hexdigest()[:20]
            items.append(
                {
                    "blind_item_id": blind_id,
                    "row_id": str(row.row_id),
                    "dialect_source_text": str(row.source),
                    "candidate_standard_bangla": str(prediction),
                }
            )
            master.append(
                {
                    "blind_item_id": blind_id,
                    "row_id": str(row.row_id),
                    "semantic_group_id": str(row.semantic_group_id),
                    "dialect": str(row.dialect),
                    "source_id": str(row.source_id),
                    "system": system,
                    "reference": str(row.reference),
                    "prediction": str(prediction),
                }
            )
    if {row["row_id"] for row in items} != sampled_ids:
        raise RuntimeError("Internal human sample mismatch")

    output.mkdir(parents=True, exist_ok=True)
    master_frame = pd.DataFrame(master).sort_values("blind_item_id")
    master_frame.to_csv(output / "BLINDING_KEY_DO_NOT_SHARE_WITH_RATERS.csv", index=False)
    item_frame = pd.DataFrame(items).set_index("blind_item_id", drop=False)
    rater_count = int(config["raters"])
    ratings_per_item = int(config["ratings_per_item"])
    assignments = {rater: [] for rater in range(1, rater_count + 1)}
    for blind_id in sorted(item_frame.index):
        offset = int.from_bytes(stable_rank(blind_id, int(config["sampling_seed"]))[:4], "big")
        selected_raters = [((offset + step) % rater_count) + 1 for step in range(ratings_per_item)]
        for rater in selected_raters:
            assignments[rater].append(blind_id)
    for rater, blind_ids in assignments.items():
        ordered = sorted(
            blind_ids,
            key=lambda value: stable_rank(value, int(config["sampling_seed"]) + rater),
        )
        packet = item_frame.loc[ordered, [
            "blind_item_id",
            "dialect_source_text",
            "candidate_standard_bangla",
        ]].copy()
        packet.insert(0, "rater_id", f"R{rater:02d}")
        for dimension in config["rating_dimensions"]:
            packet[dimension] = ""
        for dimension in config["binary_dimensions"]:
            packet[dimension] = ""
        packet["error_category_optional"] = ""
        packet["rater_comment_optional"] = ""
        packet.to_csv(output / f"rater_packet_{rater:02d}.csv", index=False)

    instructions = """# Native-speaker normalization evaluation

Rate each candidate independently. Do not try to infer the system identity.

- meaning_preservation: 1 = meaning lost/changed; 5 = all source meaning preserved.
- standard_bangla_fluency: 1 = not acceptable Standard Bangla; 5 = fully natural.
- overall_quality: 1 = unusable; 5 = publication-quality normalization.
- hallucination_or_unsupported_content: enter 1 if content was added without support, else 0.

Use `error_category_optional` for concise categories such as omission, meaning_change,
hallucination, dialect_not_normalized, grammar, named_entity, number, or other. Raters
must work independently and must not access the blinding key.
"""
    (output / "RATER_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    manifest = {
        "status": "READY_FOR_EXTERNAL_NATIVE_RATING",
        "protocol_id": config["protocol_id"],
        "systems": list(config["systems"]),
        "sampled_unique_examples": len(sample),
        "blind_system_items": len(items),
        "raters": rater_count,
        "ratings_per_item": ratings_per_item,
        "expected_completed_ratings": len(items) * ratings_per_item,
        "prediction_sha256": prediction_hashes,
        "test_policy": config["test_policy"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
