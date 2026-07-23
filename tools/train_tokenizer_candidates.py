#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import tokenizers
import yaml

PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from boichitro.tokenization import (  # noqa: E402
    CandidateSpec,
    SPECIAL_TOKENS,
    assert_special_tokens,
    build_tokenizer,
    corpus_texts,
    evaluate_tokenizer,
    intrinsic_selection_score,
    load_tokenizer,
    nfc,
    sha256_file,
    temperature_balanced_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate leakage-safe tokenizer candidates.")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/tokenizer_screen.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild candidate tokenizers if present.")
    return parser.parse_args()


def frame_hash(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame[["row_id", "dialect", "text_model"]].itertuples(index=False):
        digest.update(f"{row.row_id}\t{row.dialect}\t{nfc(row.text_model)}\n".encode("utf-8"))
    return digest.hexdigest()


def load_selection_frame(data_dir: Path) -> pd.DataFrame:
    norm = pd.read_parquet(data_dir / "normalization_validation.parquet")
    norm_src = norm[["row_id", "dialect", "source_id", "source_text_model"]].rename(
        columns={"source_text_model": "text_model"}
    )
    norm_tgt = norm[["row_id", "dialect", "source_id", "target_text_model"]].rename(
        columns={"target_text_model": "text_model"}
    )
    norm_tgt["row_id"] = norm_tgt["row_id"].astype(str) + ":target"

    identification = pd.read_parquet(data_dir / "identification_evaluation.parquet")
    identification = identification.loc[identification["split"] == "validation"]
    identification = identification[["row_id", "dialect", "source_id", "text_model"]]
    frame = pd.concat([norm_src, norm_tgt, identification], ignore_index=True)
    frame["text_model"] = frame["text_model"].map(nfc)
    frame = frame.loc[frame["text_model"].str.len() > 0]
    frame = frame.drop_duplicates(subset=["text_model"], keep="first").reset_index(drop=True)
    return frame


def select_shortlist(results: pd.DataFrame, count: int) -> list[str]:
    ordered = results.sort_values(["passes_hard_gates", "intrinsic_score"], ascending=[False, True])
    chosen: list[str] = []
    # Protect the comparison from eliminating a family solely on intrinsic compression.
    for _, family_frame in ordered.groupby("family", sort=False):
        passing = family_frame.loc[family_frame["passes_hard_gates"]]
        if not passing.empty:
            chosen.append(str(passing.iloc[0]["candidate_id"]))
    for candidate_id in ordered.loc[ordered["passes_hard_gates"], "candidate_id"]:
        if candidate_id not in chosen:
            chosen.append(str(candidate_id))
        if len(chosen) >= count:
            break
    return chosen


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    manifest_path = PROJECT / "data/manifests/tokenizer_train_v1.parquet"
    data_dir = PROJECT / "data/final/v1"
    output_root = PROJECT / "artifacts/tokenizers/candidates"
    report_dir = PROJECT / "reports/tokenizer"
    figure_dir = PROJECT / "figures/model"
    output_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(manifest_path)
    train = train.loc[train["text_model"].notna()].copy()
    train["text_model"] = train["text_model"].map(nfc)
    train = train.loc[train["text_model"].str.len() > 0].reset_index(drop=True)
    selection = load_selection_frame(data_dir)
    overlap = set(train["text_model"]) & set(selection["text_model"])
    if overlap:
        raise AssertionError(f"Tokenizer train/evaluation exact-text leakage: {len(overlap)}")

    balanced = temperature_balanced_frame(
        train,
        seed=seed,
        alpha=float(config["balance_alpha"]),
        target_rows=len(train),
    )
    corpora = {"natural": corpus_texts(train), "balanced": corpus_texts(balanced)}
    candidates: list[CandidateSpec] = []
    for family, family_config in config["families"].items():
        for vocab_size in config["vocab_sizes"]:
            candidates.append(CandidateSpec(family, int(vocab_size), str(family_config["corpus"])))

    summary_rows: list[dict[str, object]] = []
    by_dialect_frames: list[pd.DataFrame] = []
    started = time.time()
    for index, spec in enumerate(candidates, start=1):
        candidate_dir = output_root / spec.candidate_id
        if args.force and candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        print(f"[{index:02d}/{len(candidates):02d}] {spec.candidate_id}", flush=True)
        candidate_started = time.time()
        if not (candidate_dir / "tokenizer.json").exists():
            metadata = build_tokenizer(spec, corpora[spec.corpus], candidate_dir)
        else:
            metadata = json.loads((candidate_dir / "metadata.json").read_text(encoding="utf-8"))
        tokenizer = load_tokenizer(candidate_dir)
        assert_special_tokens(tokenizer)
        per_row, by_dialect, metrics = evaluate_tokenizer(tokenizer, selection)
        actual_vocab = int(metadata["actual_vocab_size"])
        score = intrinsic_selection_score(metrics, actual_vocab)
        passes = bool(
            metrics["unk_rate"] <= 2e-5
            and metrics["dialect_cost_ratio"] <= 2.0
            and metrics["canonical_roundtrip_exact_rate"] >= 0.98
        )
        row = {
            "candidate_id": spec.candidate_id,
            "family": spec.family,
            "corpus": spec.corpus,
            "requested_vocab_size": spec.vocab_size,
            "actual_vocab_size": actual_vocab,
            **metrics,
            "intrinsic_score": score,
            "passes_hard_gates": passes,
            "elapsed_seconds": time.time() - candidate_started,
            "tokenizer_sha256": sha256_file(candidate_dir / "tokenizer.json"),
        }
        summary_rows.append(row)
        by_dialect.insert(0, "candidate_id", spec.candidate_id)
        by_dialect_frames.append(by_dialect)
        per_row.drop(columns=["text"]).to_parquet(candidate_dir / "selection_metrics_by_row.parquet", index=False)
        by_dialect.to_csv(candidate_dir / "selection_metrics_by_dialect.csv", index=False)
        (candidate_dir / "selection_metrics.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"  score={score:.6f} tpc={metrics['tokens_per_character']:.5f} "
            f"worst={metrics['worst_dialect_tokens_per_character']:.5f} "
            f"ratio={metrics['dialect_cost_ratio']:.3f} unk={metrics['unk_rate']:.2e} "
            f"canonical_roundtrip={metrics['canonical_roundtrip_exact_rate']:.4f}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["passes_hard_gates", "intrinsic_score"], ascending=[False, True]
    )
    by_dialect_all = pd.concat(by_dialect_frames, ignore_index=True)
    shortlist = select_shortlist(summary, int(config["intrinsic_shortlist"]))
    summary["proxy_shortlisted"] = summary["candidate_id"].isin(shortlist)
    summary.to_csv(report_dir / "tokenizer_intrinsic_screen.csv", index=False)
    by_dialect_all.to_csv(report_dir / "tokenizer_intrinsic_by_dialect.csv", index=False)

    audit = {
        "status": "INTRINSIC_SCREEN_COMPLETE_PROXY_REQUIRED",
        "created_utc": pd.Timestamp.now("UTC").isoformat(),
        "seed": seed,
        "train_manifest": str(manifest_path.relative_to(PROJECT)),
        "train_manifest_sha256": sha256_file(manifest_path),
        "train_rows": len(train),
        "train_unique_texts": int(train["text_model"].nunique()),
        "train_frame_sha256": frame_hash(train),
        "balanced_frame_sha256": frame_hash(balanced),
        "selection_rows": len(selection),
        "selection_frame_sha256": frame_hash(selection),
        "exact_text_overlap": len(overlap),
        "candidate_count": len(candidates),
        "shortlist": shortlist,
        "special_tokens": list(SPECIAL_TOKENS),
        "software": {
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
            "pandas": pd.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    (report_dir / "tokenizer_intrinsic_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plotting = summary.sort_values("intrinsic_score")
    fig, ax = plt.subplots(figsize=(11, 5.8))
    colors = ["#0072B2" if value else "#999999" for value in plotting["passes_hard_gates"]]
    ax.bar(plotting["candidate_id"], plotting["intrinsic_score"], color=colors)
    ax.set_ylabel("Pre-registered intrinsic score (lower is better)")
    ax.tick_params(axis="x", rotation=55, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_t01_tokenizer_intrinsic_screen.png", dpi=240)
    fig.savefig(figure_dir / "fig_t01_tokenizer_intrinsic_screen.pdf")
    plt.close(fig)

    print("\nShortlist for causal-LM proxy:")
    for candidate_id in shortlist:
        print(f"  - {candidate_id}")
    print(f"Reports: {report_dir}")


if __name__ == "__main__":
    main()
