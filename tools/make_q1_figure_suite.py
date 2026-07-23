#!/usr/bin/env python3
"""Build the auditable Q1 main/supplementary figure suite.

Every figure is emitted as a high-resolution PNG and a vector PDF.  The exact
plotting frame is also exported, and the manifest distinguishes development,
previously accessed descriptive test, and locked-test evidence so that no
preliminary or legacy curve can be mistaken for confirmatory results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import make_paper_figures as paper


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "figures/q1"
DATA = OUTPUT / "source_data"
MODEL_ORDER = ["M0", "M1", "M2", "M3"]
MODEL_PALETTE = {
    "M0": "#4477AA",
    "M1": "#EE6677",
    "M2": "#228833",
    "M3": "#AA3377",
}
TASK_PALETTE = {"normalization": "#4477AA", "identification": "#EE6677"}
SPLIT_ORDER = ["train", "validation", "test", "test_iid", "test_ood"]
RECORDS: list[dict[str, Any]] = []


CORE_METADATA = {
    "fig_tokenizer_tradeoff": (
        "Tokenizer efficiency–quality frontier",
        "Development-only proxy bits per character versus intrinsic tokenization cost. Error bars show seed standard deviation; the star marks the frozen tokenizer.",
        "tokenizer",
    ),
    "fig_compute_pareto": (
        "Active-compute systems frontier",
        "Measured GB10 training throughput versus total parameters for compute-matched dense and sparse architectures.",
        "systems",
    ),
    "fig_classical_id_floor": (
        "Classical dialect-identification floors",
        "Regional macro-F1 of fixed character n-gram classifiers across validation, IID, source-OOD, and external-transcript tracks.",
        "baseline",
    ),
    "fig_classical_norm_floor": (
        "Fair normalization floors and oracle disclosure",
        "Macro chrF++ for source-blind copy/rewrite controls; the legacy gold-dialect rewrite is explicitly labeled as an oracle rather than a deployable comparator.",
        "baseline",
    ),
    "fig_continuation_lr_stability": (
        "Continuation learning-rate stability",
        "Validation BPC change from the mature-checkpoint LR pilot; the rejected warm restart is retained as a negative result.",
        "foundation",
    ),
    "fig_stage_s_retention_tradeoff": (
        "Normalization–retention trade-off",
        "Validation macro chrF++ versus replay-NLL degradation during registered Stage-S schedule selection. The dashed line is the preregistered 5% guard.",
        "task-development",
    ),
    "fig_architecture": (
        "Boichitro-MoE architecture",
        "Source-blind task input, dense prefix, sparse decoder blocks, router curricula, auxiliary heads, and hybrid optimization.",
        "method",
    ),
    "fig_protocol_flow": (
        "Leakage-safe experimental protocol",
        "Registered development sequence and immutable freeze boundary preceding one scripted locked evaluation.",
        "method",
    ),
    "fig_training_curves": (
        "Foundation and continuation learning curves",
        "Smoothed training loss and held-out BPC for the dense foundation and matched 200M-token continuations.",
        "foundation",
    ),
    "fig_upcycling_recovery": (
        "Dense-to-MoE recovery pilot",
        "Validation-only comparison of bank release and initialization strategies under a matched 20M-token budget.",
        "foundation",
    ),
    "fig_switch_router_recovery": (
        "Switch-router failure and repair",
        "Held-out BPC and expert-load variation for the disclosed collapsed run and two registered repair candidates.",
        "foundation",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="locked_test_v1")
    parser.add_argument("--suffix", default="base")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    parser.add_argument("--minimum", type=int, default=30)
    return parser.parse_args()


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=0.95)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_axes(axes: Iterable[plt.Axes] | np.ndarray | plt.Axes) -> None:
    values = np.asarray(axes, dtype=object).reshape(-1)
    for axis in values:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.75)
        axis.grid(axis="x", visible=False)


def export_frame(figure_id: str, frame: pd.DataFrame) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{figure_id}.csv"
    frame.to_csv(path, index=False)
    return path


def save(
    fig: plt.Figure,
    figure_id: str,
    title: str,
    caption: str,
    category: str,
    sources: list[str],
    frame: pd.DataFrame | None = None,
    evidence: str = "development_only",
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data_path = export_frame(figure_id, frame) if frame is not None else None
    png = OUTPUT / f"{figure_id}.png"
    pdf = OUTPUT / f"{figure_id}.pdf"
    fig.savefig(
        png,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        dpi=600,
        metadata={"Title": title, "Description": caption},
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        metadata={"Title": title, "Subject": caption},
    )
    plt.close(fig)
    RECORDS.append(
        {
            "id": figure_id,
            "title": title,
            "caption": caption,
            "category": category,
            "evidence": evidence,
            "png": str(png.relative_to(PROJECT)),
            "pdf": str(pdf.relative_to(PROJECT)),
            "source_data": str(data_path.relative_to(PROJECT)) if data_path else None,
            "sources": sources,
            "png_bytes": png.stat().st_size,
            "pdf_bytes": pdf.stat().st_size,
            "png_sha256": sha256(png),
            "pdf_sha256": sha256(pdf),
        }
    )


def register_core(names: list[str], evidence: str = "development_only") -> None:
    alternate_source_data = {
        "fig_switch_router_recovery": DATA / "fig_switch_router_validation.csv",
    }
    alternate_sources = {
        "fig_switch_router_recovery": [
            "figures/q1/source_data/fig_switch_router_validation.csv",
            "figures/q1/source_data/fig_switch_router_load.csv",
        ],
    }
    for name in names:
        png = OUTPUT / f"{name}.png"
        pdf = OUTPUT / f"{name}.pdf"
        if not png.exists() or not pdf.exists():
            continue
        title, caption, category = CORE_METADATA[name]
        figure_evidence = (
            "prior_test_descriptive"
            if name in {"fig_classical_id_floor", "fig_classical_norm_floor"}
            else evidence
        )
        source = DATA / f"{name}.csv"
        if not source.exists():
            json_source = DATA / f"{name}.json"
            source = json_source if json_source.exists() else source
        if not source.exists() and name in alternate_source_data:
            source = alternate_source_data[name]
        RECORDS.append(
            {
                "id": name,
                "title": title,
                "caption": caption,
                "category": category,
                "evidence": figure_evidence,
                "png": str(png.relative_to(PROJECT)),
                "pdf": str(pdf.relative_to(PROJECT)),
                "source_data": str(source.relative_to(PROJECT)) if source.exists() else None,
                "sources": alternate_sources.get(
                    name, ["generated by tools/make_paper_figures.py"]
                ),
                "png_bytes": png.stat().st_size,
                "pdf_bytes": pdf.stat().st_size,
                "png_sha256": sha256(png),
                "pdf_sha256": sha256(pdf),
            }
        )


def build_core_figures() -> None:
    created: list[str] = []
    created += paper.preliminary_figures(OUTPUT, DATA)
    created += paper.continuation_lr_stability_figure(OUTPUT, DATA)
    created += paper.stage_s_retention_figure(OUTPUT, DATA)
    created += paper.architecture_figure(OUTPUT, DATA)
    created += paper.protocol_flow_figure(OUTPUT, DATA)
    created += paper.training_curves_figure(OUTPUT, DATA)
    created += paper.upcycling_recovery_figure(OUTPUT, DATA)
    created += paper.switch_router_recovery_figure(OUTPUT, DATA)
    register_core(created)


def data_figures() -> None:
    root = PROJECT / "data/final/v1"
    norm = pd.read_parquet(root / "normalization_all.parquet")
    ident = pd.read_parquet(root / "identification_all.parquet")
    roman = pd.read_parquet(root / "romanized_test_ood.parquet")

    ncounts = norm.groupby(["dialect", "split"], observed=True).size().rename("rows").reset_index()
    icounts = ident.groupby(["dialect", "split"], observed=True).size().rename("rows").reset_index()
    combined = pd.concat(
        [ncounts.assign(task="normalization"), icounts.assign(task="identification")],
        ignore_index=True,
    )
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.8), constrained_layout=True)
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = combined.loc[combined.task.eq(task)]
        pivot = subset.pivot_table(index="dialect", columns="split", values="rows", fill_value=0)
        pivot = pivot.reindex(columns=[c for c in SPLIT_ORDER if c in pivot.columns])
        pivot.plot(kind="bar", stacked=True, ax=axis, width=0.84, colormap="Blues" if task == "normalization" else "Reds")
        axis.set_title(f"{task.capitalize()} rows by dialect and split")
        axis.set_xlabel("")
        axis.set_ylabel("Rows")
        axis.tick_params(axis="x", rotation=0)
        axis.legend(title="Split", ncol=min(5, len(pivot.columns)), frameon=False)
    save(
        fig,
        "fig_data_split_dialect_balance",
        "Dialect and split balance",
        "Final admitted rows by dialect and split for the two supervised tasks; stacked segments expose sparse labels and held-out coverage.",
        "data",
        ["data/final/v1/normalization_all.parquet", "data/final/v1/identification_all.parquet"],
        combined,
    )

    source_rows = pd.concat(
        [
            norm.groupby("source_id").size().rename("rows").reset_index().assign(task="normalization"),
            ident.groupby("source_id").size().rename("rows").reset_index().assign(task="identification"),
        ],
        ignore_index=True,
    )
    source_rows["share"] = source_rows["rows"] / source_rows.groupby("task")["rows"].transform("sum")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = source_rows.loc[source_rows.task.eq(task)].nlargest(10, "rows").sort_values("rows")
        axis.barh(subset.source_id, subset.share * 100, color=TASK_PALETTE[task])
        axis.set_title(f"{task.capitalize()}: ten largest sources")
        axis.set_xlabel("Share of admitted rows (%)")
        axis.set_ylabel("")
    save(
        fig,
        "fig_data_source_composition",
        "Source composition",
        "Largest provenance sources and their share of admitted normalization and identification rows.",
        "data",
        ["data/final/v1/normalization_all.parquet", "data/final/v1/identification_all.parquet"],
        source_rows,
    )

    length_rows = pd.concat(
        [
            norm[["row_id", "dialect", "split", "source_text_model"]]
            .rename(columns={"source_text_model": "text"})
            .assign(task="normalization"),
            ident[["row_id", "dialect", "split", "text_model"]]
            .rename(columns={"text_model": "text"})
            .assign(task="identification"),
        ],
        ignore_index=True,
    )
    length_rows["characters"] = length_rows.text.astype(str).str.len()
    sampled = (
        length_rows.sample(frac=1.0, random_state=1701)
        .groupby(["task", "dialect"], group_keys=False, observed=True)
        .head(600)
        .reset_index(drop=True)
    )
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.8), constrained_layout=True)
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = sampled.loc[sampled.task.eq(task)]
        sns.boxplot(data=subset, x="dialect", y="characters", showfliers=False, color=TASK_PALETTE[task], ax=axis)
        axis.set_title(f"{task.capitalize()} input-length distribution")
        axis.set_xlabel("")
        axis.set_ylabel("Unicode characters")
    save(
        fig,
        "fig_data_text_length",
        "Input-length distributions",
        "Character-length distributions by dialect after frozen cleaning; boxes omit display outliers but source data retain them.",
        "data",
        ["data/final/v1/normalization_all.parquet", "data/final/v1/identification_all.parquet"],
        sampled.drop(columns=["text"]),
    )

    tracks = pd.concat(
        [
            norm.groupby(["evaluation_track", "split"]).size().rename("rows").reset_index().assign(task="normalization"),
            ident.groupby(["evaluation_track", "split"]).size().rename("rows").reset_index().assign(task="identification"),
        ],
        ignore_index=True,
    )
    tracks["track"] = tracks.evaluation_track.replace("", "training/development")
    show = tracks.groupby(["task", "track"], as_index=False).rows.sum()
    show["rank"] = show.groupby("task").rows.rank(method="first", ascending=False)
    show = show.loc[show["rank"].le(12)]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = show.loc[show.task.eq(task)].sort_values("rows")
        axis.barh(subset.track, subset.rows, color=TASK_PALETTE[task])
        axis.set_title(f"{task.capitalize()} evaluation tracks")
        axis.set_xlabel("Rows")
        axis.set_ylabel("")
    save(
        fig,
        "fig_data_evaluation_tracks",
        "Registered evaluation-track composition",
        "Row counts in the frozen IID, source-held-out, external, and challenge tracks; only the twelve largest labels per task are displayed.",
        "data",
        ["data/final/v1/normalization_all.parquet", "data/final/v1/identification_all.parquet"],
        tracks,
    )

    synthetic = (
        norm.assign(origin=np.where(norm.is_synthetic, "synthetic train-only", "authentic"))
        .groupby(["dialect", "split", "origin"], observed=True)
        .size()
        .rename("rows")
        .reset_index()
    )
    fig, axis = plt.subplots(figsize=(8.2, 3.6), constrained_layout=True)
    pivot = synthetic.groupby(["dialect", "origin"], observed=True).rows.sum().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True, color=["#4477AA", "#CCBB44"], width=0.82, ax=axis)
    axis.set_title("Synthetic material is restricted to the normalization training pool")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Rows")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title=None, frameon=False)
    save(
        fig,
        "fig_data_synthetic_footprint",
        "Synthetic-data footprint",
        "Authentic and traceable train-only perturbation rows by dialect; synthetic rows are not evaluation evidence.",
        "data",
        ["data/final/v1/normalization_all.parquet"],
        synthetic,
    )

    roman_rows = roman.assign(
        input_characters=roman.romanized_input_model.astype(str).str.len(),
        target_characters=roman.target_text_model.astype(str).str.len(),
    )
    roman_summary = roman_rows.groupby("dialect", observed=True).agg(
        rows=("row_id", "size"),
        median_input_characters=("input_characters", "median"),
        median_target_characters=("target_characters", "median"),
    ).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    axes[0].bar(roman_summary.dialect, roman_summary.rows, color="#66CCEE")
    axes[0].set_title("Romanized source-held-out rows")
    axes[0].set_ylabel("Rows")
    axes[0].set_xlabel("Dialect")
    axes[1].scatter(roman_summary.median_input_characters, roman_summary.median_target_characters, s=70, color="#AA3377")
    for row in roman_summary.itertuples(index=False):
        axes[1].annotate(row.dialect, (row.median_input_characters, row.median_target_characters), xytext=(3, 3), textcoords="offset points")
    axes[1].set_title("Median sequence lengths")
    axes[1].set_xlabel("Romanized input characters")
    axes[1].set_ylabel("Standard-Bangla target characters")
    save(
        fig,
        "fig_data_romanized_coverage",
        "Romanized challenge coverage",
        "Dialect support and median sequence lengths for the frozen real romanized source-held-out track.",
        "data",
        ["data/final/v1/romanized_test_ood.parquet"],
        roman_summary,
    )

    quality = pd.concat(
        [
            norm.groupby("quality_tier").size().rename("rows").reset_index().assign(task="normalization"),
            ident.groupby("quality_tier").size().rename("rows").reset_index().assign(task="identification"),
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.9), constrained_layout=True)
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = quality.loc[quality.task.eq(task)].sort_values("rows")
        axis.barh(subset.quality_tier, subset.rows, color=TASK_PALETTE[task])
        axis.set_title(f"{task.capitalize()} quality tiers")
        axis.set_xlabel("Rows")
        axis.set_ylabel("")
    save(
        fig,
        "fig_data_quality_tiers",
        "Dataset quality tiers",
        "Frozen provenance/quality tiers for admitted task rows, retaining explicit uncertainty rather than imputing provenance.",
        "data",
        ["data/final/v1/normalization_all.parquet", "data/final/v1/identification_all.parquet"],
        quality,
    )

    coverage = norm.groupby(["source_id", "dialect"], observed=True).size().rename("rows").reset_index()
    top_sources = coverage.groupby("source_id").rows.sum().nlargest(12).index
    heat = coverage.loc[coverage.source_id.isin(top_sources)].pivot(index="source_id", columns="dialect", values="rows").fillna(0)
    fig, axis = plt.subplots(figsize=(9.3, 4.8), constrained_layout=True)
    sns.heatmap(np.log10(heat + 1), cmap="viridis", linewidths=0.25, cbar_kws={"label": "log10(rows + 1)"}, ax=axis)
    axis.set_title("Normalization source × dialect coverage")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Source")
    save(
        fig,
        "fig_data_source_dialect_coverage",
        "Source–dialect coverage matrix",
        "Log-scaled row counts for the twelve largest normalization sources, making missing source–dialect cells explicit.",
        "data",
        ["data/final/v1/normalization_all.parquet"],
        coverage,
    )


def tokenizer_figures() -> None:
    screen = pd.read_csv(PROJECT / "reports/tokenizer/tokenizer_intrinsic_screen.csv")
    dialect = pd.read_csv(PROJECT / "reports/tokenizer/tokenizer_intrinsic_by_dialect.csv")
    proxy = pd.read_csv(PROJECT / "reports/tokenizer/tokenizer_proxy_seed_metrics.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), constrained_layout=True)
    sns.lineplot(data=screen, x="actual_vocab_size", y="tokens_per_character", hue="family", style="corpus", marker="o", ax=axes[0])
    axes[0].set_title("Vocabulary scaling")
    axes[0].set_xlabel("Actual vocabulary size")
    axes[0].set_ylabel("Tokens per character ↓")
    sns.lineplot(data=screen, x="actual_vocab_size", y="mean_fertility", hue="family", style="corpus", marker="o", legend=False, ax=axes[1])
    axes[1].set_title("Word fertility")
    axes[1].set_xlabel("Actual vocabulary size")
    axes[1].set_ylabel("Mean subwords per word ↓")
    save(
        fig,
        "fig_tokenizer_vocab_scaling",
        "Tokenizer vocabulary scaling",
        "Intrinsic token cost and fertility across family, corpus balance, and vocabulary size.",
        "tokenizer",
        ["reports/tokenizer/tokenizer_intrinsic_screen.csv"],
        screen,
    )

    shortlist = screen.loc[screen.proxy_shortlisted.astype(bool), "candidate_id"].tolist()
    heat = dialect.loc[dialect.candidate_id.isin(shortlist)].pivot(index="candidate_id", columns="dialect", values="tokens_per_character")
    fig, axis = plt.subplots(figsize=(9.3, 3.4), constrained_layout=True)
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="mako_r", cbar_kws={"label": "tokens/character"}, ax=axis)
    axis.set_title("Shortlisted tokenizer cost by dialect")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Candidate")
    save(
        fig,
        "fig_tokenizer_dialect_fertility",
        "Tokenizer cost by dialect",
        "Tokens per character for the validation-shortlisted candidates across the frozen dialect inventory.",
        "tokenizer",
        ["reports/tokenizer/tokenizer_intrinsic_by_dialect.csv"],
        dialect.loc[dialect.candidate_id.isin(shortlist)],
    )

    fig, axis = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    colors = np.where(screen.proxy_shortlisted.astype(bool), "#AA3377", "#4477AA")
    axis.scatter(screen.dialect_cost_ratio, screen.dialect_cost_gini, c=colors, s=42, alpha=0.85)
    for row in screen.loc[screen.proxy_shortlisted.astype(bool)].itertuples(index=False):
        axis.annotate(row.candidate_id.replace("_balanced", "-bal").replace("_natural", "-nat"), (row.dialect_cost_ratio, row.dialect_cost_gini), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    axis.set_xlabel("Worst/best dialect token-cost ratio ↓")
    axis.set_ylabel("Dialect token-cost Gini ↓")
    axis.set_title("Tokenizer parity across dialects")
    save(
        fig,
        "fig_tokenizer_fairness",
        "Tokenizer dialect parity",
        "Worst-to-best cost ratio versus Gini dispersion for every screened tokenizer; highlighted candidates entered the proxy study.",
        "tokenizer",
        ["reports/tokenizer/tokenizer_intrinsic_screen.csv"],
        screen,
    )

    proxy_summary = proxy.groupby("candidate_id", as_index=False).agg(
        mean_bpc=("validation_bits_per_character", "mean"),
        sd_bpc=("validation_bits_per_character", "std"),
        mean_throughput=("training_tokens_per_second", "mean"),
        sd_throughput=("training_tokens_per_second", "std"),
        mean_range=("validation_dialect_bpc_range", "mean"),
    )
    proxy_summary = proxy_summary.sort_values("mean_bpc")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.6), constrained_layout=True)
    positions = np.arange(len(proxy_summary))
    axes[0].errorbar(proxy_summary.mean_bpc, positions, xerr=proxy_summary.sd_bpc.fillna(0), fmt="o", color="#4477AA", capsize=3)
    axes[0].set_yticks(positions, proxy_summary.candidate_id)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Validation BPC (mean ± SD) ↓")
    axes[0].set_title("Three-seed proxy quality")
    axes[1].errorbar(proxy_summary.mean_throughput, positions, xerr=proxy_summary.sd_throughput.fillna(0), fmt="o", color="#228833", capsize=3)
    axes[1].set_yticks(positions, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Training tokens/s (mean ± SD) ↑")
    axes[1].set_title("Proxy throughput")
    save(
        fig,
        "fig_tokenizer_proxy_stability",
        "Tokenizer proxy stability",
        "Three-seed proxy-language-model quality and throughput for shortlisted tokenizers.",
        "tokenizer",
        ["reports/tokenizer/tokenizer_proxy_seed_metrics.csv"],
        proxy_summary,
    )


def pretraining_figures() -> None:
    report = json.loads((PROJECT / "reports/model/pretraining_corpus_report.json").read_text(encoding="utf-8"))
    rejection_rows = pd.DataFrame([{"reason": key, "documents": value} for key, value in report["rejections"].items()])
    acceptance = pd.concat(
        [
            pd.DataFrame(
                [
                    {"reason": "accepted", "documents": report["accepted_documents"]},
                    {"reason": "all rejected", "documents": report["raw_documents_seen"] - report["accepted_documents"]},
                ]
            ),
            rejection_rows,
        ],
        ignore_index=True,
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    axes[0].pie(
        acceptance.loc[acceptance.reason.isin(["accepted", "all rejected"]), "documents"],
        labels=["accepted", "rejected"],
        colors=["#228833", "#CC6677"],
        autopct="%1.1f%%",
        startangle=90,
    )
    axes[0].set_title("FineWeb-2 Bengali document gate")
    rejection_rows.sort_values("documents").plot.barh(x="reason", y="documents", color="#CC6677", legend=False, ax=axes[1])
    axes[1].set_title("Rejection reasons")
    axes[1].set_xlabel("Documents")
    axes[1].set_ylabel("")
    save(
        fig,
        "fig_pretraining_acceptance",
        "Pretraining corpus acceptance gate",
        "Accepted share and mutually recorded rejection reasons for the pinned FineWeb-2 Bengali source.",
        "pretraining-data",
        ["reports/model/pretraining_corpus_report.json"],
        acceptance,
    )

    shards = pd.DataFrame(report["shards"])
    shards["shard_index"] = np.arange(len(shards))
    shards["validation_fraction"] = shards.validation_rows / shards.rows
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), constrained_layout=True)
    axes[0].plot(shards.shard_index, shards.tokens / 1e6, color="#4477AA", linewidth=1.1)
    axes[0].axhline(shards.tokens.median() / 1e6, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Tokens per immutable shard")
    axes[0].set_xlabel("Shard index")
    axes[0].set_ylabel("Tokens (millions)")
    axes[1].scatter(shards.tokens / 1e6, shards.validation_fraction * 100, color="#228833", s=26, alpha=0.8)
    axes[1].set_title("Deterministic validation allocation")
    axes[1].set_xlabel("Shard tokens (millions)")
    axes[1].set_ylabel("Validation rows (%)")
    save(
        fig,
        "fig_pretraining_shard_balance",
        "Pretraining shard balance",
        "Token volume and deterministic validation allocation across immutable packed-source shards.",
        "pretraining-data",
        ["reports/model/pretraining_corpus_report.json"],
        shards.drop(columns=[c for c in ("sha256", "path") if c in shards]),
    )


def foundation_figures() -> None:
    benchmark = pd.read_csv(PROJECT / "reports/model/gb10_model_benchmark.csv")
    benchmark["variant"] = benchmark.model_id.str.extract(r"^(M\d)")
    benchmark["active_millions"] = benchmark.active_parameters_per_token / 1e6
    benchmark["inactive_millions"] = benchmark.inactive_expert_parameters_per_token / 1e6
    fig, axis = plt.subplots(figsize=(7.2, 3.7), constrained_layout=True)
    axis.bar(benchmark.variant, benchmark.active_millions, color="#4477AA", label="active/token")
    axis.bar(benchmark.variant, benchmark.inactive_millions, bottom=benchmark.active_millions, color="#BBBBBB", label="inactive experts")
    axis.set_ylabel("Parameters (millions)")
    axis.set_xlabel("Architecture")
    axis.set_title("Total versus token-active parameter capacity")
    axis.legend(frameon=False)
    save(
        fig,
        "fig_model_parameter_decomposition",
        "Model parameter decomposition",
        "Token-active and inactive expert capacity for the dense, Switch, shared-expert MoE, and Boichitro architectures.",
        "systems",
        ["reports/model/gb10_model_benchmark.csv"],
        benchmark,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    for row in benchmark.itertuples(index=False):
        color = MODEL_PALETTE[row.variant]
        axes[0].scatter(row.tokens_per_second, row.peak_memory_gib, s=85, color=color, edgecolor="white")
        axes[0].annotate(row.variant, (row.tokens_per_second, row.peak_memory_gib), xytext=(4, 3), textcoords="offset points")
        axes[1].scatter(row.active_fraction * 100, row.tokens_per_second, s=85, color=color, edgecolor="white")
        axes[1].annotate(row.variant, (row.active_fraction * 100, row.tokens_per_second), xytext=(4, 3), textcoords="offset points")
    axes[0].set_title("Throughput–memory frontier")
    axes[0].set_xlabel("Training tokens/s ↑")
    axes[0].set_ylabel("Peak memory (GiB) ↓")
    axes[1].set_title("Sparsity–throughput relation")
    axes[1].set_xlabel("Active parameter fraction (%)")
    axes[1].set_ylabel("Training tokens/s ↑")
    save(
        fig,
        "fig_model_throughput_memory",
        "Measured systems efficiency",
        "GB10 throughput, peak memory, and active parameter fraction under the same batch and sequence length.",
        "systems",
        ["reports/model/gb10_model_benchmark.csv"],
        benchmark,
    )

    run_map = {"M0": "U_M0_DENSE_200M", "M1": "U_M1_SWITCH_200M", "M2": "U_M2_STANDARD_MOE_200M"}
    frames = []
    for variant, run_id in run_map.items():
        frame = pd.read_json(PROJECT / "runs" / run_id / "1701/train_log.jsonl", lines=True)
        frame["variant"] = variant
        frame["tokens_millions"] = frame.tokens_seen / 1e6
        frames.append(frame)
    train = pd.concat(frames, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    sns.lineplot(data=train, x="tokens_millions", y="lm_loss", hue="variant", palette=MODEL_PALETTE, ax=axes[0])
    axes[0].set_title("Continuation language-model loss")
    axes[0].set_xlabel("Tokens (millions)")
    axes[0].set_ylabel("LM loss")
    sns.lineplot(data=train, x="tokens_millions", y="mtp_loss", hue="variant", palette=MODEL_PALETTE, legend=False, ax=axes[1])
    axes[1].set_title("Multi-token-prediction loss")
    axes[1].set_xlabel("Tokens (millions)")
    axes[1].set_ylabel("MTP loss")
    save(
        fig,
        "fig_foundation_loss_components",
        "Foundation continuation loss components",
        "Logged LM and multi-token-prediction losses for compute-matched 200M-token continuations.",
        "foundation",
        [f"runs/{value}/1701/train_log.jsonl" for value in run_map.values()],
        train,
    )

    routed = train.loc[train.variant.isin(["M1", "M2"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5), constrained_layout=True)
    sns.lineplot(data=routed, x="tokens_millions", y="router_load_cv", hue="variant", palette=MODEL_PALETTE, ax=axes[0])
    axes[0].axhline(0.5, color="#CC6677", linestyle="--", linewidth=0.8)
    axes[0].set_title("Expert-load coefficient of variation")
    axes[0].set_xlabel("Tokens (millions)")
    axes[0].set_ylabel("Load CV ↓")
    sns.lineplot(data=routed, x="tokens_millions", y="router_entropy", hue="variant", palette=MODEL_PALETTE, legend=False, ax=axes[1])
    axes[1].set_title("Router entropy")
    axes[1].set_xlabel("Tokens (millions)")
    axes[1].set_ylabel("Entropy")
    save(
        fig,
        "fig_foundation_router_dynamics",
        "Foundation router dynamics",
        "Expert-load variation and router entropy over matched Switch and shared-expert MoE continuation.",
        "foundation",
        ["runs/U_M1_SWITCH_200M/1701/train_log.jsonl", "runs/U_M2_STANDARD_MOE_200M/1701/train_log.jsonl"],
        routed,
    )

    diagnostics = pd.read_csv(PROJECT / "reports/model/foundation_router_diagnostics.csv")
    diagnostics["variant"] = diagnostics.run_id.map({"U_M1_SWITCH_200M": "M1", "U_M2_STANDARD_MOE_200M": "M2"})
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), constrained_layout=True)
    metrics = [
        ("router_load_cv_mean", "Mean load CV ↓"),
        ("cv_threshold_exceedance_fraction", "Threshold exceedance fraction ↓"),
        ("final_validation_bpc", "Final validation BPC ↓"),
    ]
    for axis, (metric, label) in zip(axes, metrics):
        axis.bar(diagnostics.variant, diagnostics[metric], color=[MODEL_PALETTE[v] for v in diagnostics.variant])
        axis.set_title(label)
        axis.set_xlabel("")
    save(
        fig,
        "fig_foundation_router_stability",
        "Router stability summary",
        "Registered diagnostic summary for load balance, sustained threshold exceedance, and held-out BPC.",
        "foundation",
        ["reports/model/foundation_router_diagnostics.csv"],
        diagnostics,
    )


def complete_task_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = PROJECT / "runs/task/boichitro_q1_v1"
    if not root.exists():
        return rows
    for variant_dir in sorted(root.glob("M?__base")):
        variant = variant_dir.name.split("__", 1)[0]
        for seed_dir in sorted(variant_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.isdigit():
                continue
            stage_s_report = seed_dir / "stage_s/training_report.json"
            stage_id_report = seed_dir / "stage_id/training_report.json"
            if not stage_s_report.exists() or not stage_id_report.exists():
                continue
            s_report = json.loads(stage_s_report.read_text(encoding="utf-8"))
            i_report = json.loads(stage_id_report.read_text(encoding="utf-8"))
            if s_report.get("status") != "COMPLETE" or i_report.get("status") != "COMPLETE":
                continue
            rows.append(
                {
                    "variant": variant,
                    "seed": int(seed_dir.name),
                    "root": seed_dir,
                    "stage_s_report": s_report,
                    "stage_id_report": i_report,
                }
            )
    return rows


def task_figures() -> None:
    runs = complete_task_runs()
    if not runs:
        return
    s_rows: list[dict[str, Any]] = []
    id_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    norm_dialect: list[pd.DataFrame] = []
    id_class: list[pd.DataFrame] = []
    matrices: dict[str, list[np.ndarray]] = {}
    reliability: list[dict[str, Any]] = []
    efficiency: list[dict[str, Any]] = []
    loss_frames: list[pd.DataFrame] = []

    for item in runs:
        variant, seed, root = item["variant"], item["seed"], item["root"]
        s_selection = json.loads((root / "stage_s/best_selection.json").read_text(encoding="utf-8"))
        i_selection = json.loads((root / "stage_id/best_selection.json").read_text(encoding="utf-8"))
        s_id = int(s_selection["validation_id"])
        i_id = int(i_selection["validation_id"])
        for path in sorted((root / "stage_s").glob("validation_metrics_epoch_*.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            step = int(path.stem.rsplit("_", 1)[1])
            s_rows.append({"variant": variant, "seed": seed, "step": step, "selected": step == s_id, **metrics})
        for path in sorted((root / "stage_id").glob("validation_metrics_epoch_*.json")):
            metrics = json.loads(path.read_text(encoding="utf-8"))
            epoch = int(path.stem.rsplit("_", 1)[1])
            id_rows.append({"variant": variant, "seed": seed, "epoch": epoch, "selected": epoch == i_id, **metrics})
        selected_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "macro_chrfpp": s_selection["validation"]["macro_chrfpp"],
                "worst_dialect_chrfpp": s_selection["validation"]["worst_dialect_chrfpp"],
                "replay_degradation_percent": 100 * s_selection["validation"]["replay_relative_degradation"],
                "regional_macro_f1": i_selection["validation"]["regional_macro_f1"],
                "worst_present_dialect_f1": i_selection["validation"]["worst_present_dialect_f1"],
                "ece_15": i_selection["validation"]["ece_15"],
                "stage_s_validation_id": s_id,
                "stage_id_validation_id": i_id,
            }
        )
        by_dialect_path = root / f"stage_s/validation_by_dialect_epoch_{s_id:02d}.csv"
        frame = pd.read_csv(by_dialect_path).assign(variant=variant, seed=seed)
        norm_dialect.append(frame)
        by_class_path = root / f"stage_id/validation_by_class_epoch_{i_id:02d}.csv"
        frame = pd.read_csv(by_class_path).assign(variant=variant, seed=seed)
        id_class.append(frame)
        matrix_path = root / f"stage_id/validation_confusion_epoch_{i_id:02d}.npy"
        matrix = np.load(matrix_path).astype(float)
        matrix = matrix / np.maximum(1, matrix.sum(axis=1, keepdims=True))
        matrices.setdefault(variant, []).append(matrix)
        prediction_path = root / f"stage_id/validation_predictions_epoch_{i_id:02d}.parquet"
        predictions = pd.read_parquet(prediction_path)
        probabilities = np.asarray(predictions.probabilities.tolist(), dtype=float)
        confidence = probabilities.max(axis=1)
        correct = predictions.prediction_id.to_numpy() == predictions.label_id.to_numpy()
        bins = np.minimum((confidence * 10).astype(int), 9)
        for index in range(10):
            mask = bins == index
            if mask.any():
                reliability.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "bin": index,
                        "count": int(mask.sum()),
                        "confidence": float(confidence[mask].mean()),
                        "accuracy": float(correct[mask].mean()),
                    }
                )
        for stage, report in (("stage_s", item["stage_s_report"]), ("stage_id", item["stage_id_report"])):
            efficiency.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "stage": stage,
                    "elapsed_minutes": report["elapsed_seconds"] / 60,
                    "tokens_per_second": report["tokens_per_second"],
                    "peak_memory_gib": report["peak_memory_gib"],
                }
            )
        for stage in ("stage_a", "stage_s", "stage_id"):
            log_path = root / stage / "train_log.jsonl"
            if log_path.exists():
                frame = pd.read_json(log_path, lines=True)
                frame["variant"] = variant
                frame["seed"] = seed
                frame["stage"] = stage
                frame["progress"] = frame.global_step / max(1, frame.global_step.max())
                loss_frames.append(frame)

    s_frame = pd.DataFrame(s_rows)
    id_frame = pd.DataFrame(id_rows)
    selected = pd.DataFrame(selected_rows)
    norm_by_dialect = pd.concat(norm_dialect, ignore_index=True)
    id_by_class = pd.concat(id_class, ignore_index=True)
    reliability_frame = pd.DataFrame(reliability)
    efficiency_frame = pd.DataFrame(efficiency)
    loss_frame = pd.concat(loss_frames, ignore_index=True)

    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    sns.lineplot(data=s_frame, x="step", y="macro_chrfpp", hue="variant", palette=MODEL_PALETTE, errorbar="sd", marker="o", ax=axis)
    axis.set_title("Stage-S validation trajectories")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Macro chrF++ ↑")
    save(
        fig,
        "fig_task_stage_s_trajectories",
        "Normalization validation trajectories",
        "Mean and seed dispersion of macro chrF++ at registered Stage-S validation checkpoints for completed main runs.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_s/validation_metrics_epoch_*.json"],
        s_frame,
    )

    fig, axis = plt.subplots(figsize=(7.0, 3.9), constrained_layout=True)
    for variant, subset in s_frame.groupby("variant", observed=True):
        axis.scatter(100 * subset.replay_relative_degradation, subset.macro_chrfpp, s=35, alpha=0.65, color=MODEL_PALETTE[variant], label=variant)
    axis.axvline(5, color="#CC6677", linestyle="--", linewidth=1)
    axis.set_xlabel("Replay-NLL degradation (%) ↓")
    axis.set_ylabel("Macro chrF++ ↑")
    axis.set_title("Main-run task quality under the retention guard")
    axis.legend(frameon=False)
    replay_export = s_frame.assign(replay_degradation_percent=100 * s_frame.replay_relative_degradation)
    save(
        fig,
        "fig_task_replay_tradeoff",
        "Main-run retention trade-off",
        "All main-run normalization checkpoints in development space; the dashed line is the fixed replay guard.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_s/validation_metrics_epoch_*.json"],
        replay_export,
    )

    fig, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    sns.lineplot(data=id_frame, x="epoch", y="regional_macro_f1", hue="variant", palette=MODEL_PALETTE, errorbar="sd", marker="o", ax=axis)
    axis.set_title("Causal identification validation trajectories")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Regional macro-F1 ↑")
    save(
        fig,
        "fig_task_stage_id_trajectories",
        "Identification validation trajectories",
        "Mean and seed dispersion of regional macro-F1 during causal identification specialization.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_id/validation_metrics_epoch_*.json"],
        id_frame,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    sns.stripplot(data=selected, x="variant", y="macro_chrfpp", hue="variant", palette=MODEL_PALETTE, legend=False, jitter=0.08, size=7, ax=axes[0])
    sns.pointplot(data=selected, x="variant", y="macro_chrfpp", color="black", errorbar="sd", markers="_", linestyles="none", ax=axes[0])
    axes[0].set_title("Selected normalization checkpoints")
    axes[0].set_ylabel("Macro chrF++")
    axes[0].set_xlabel("")
    sns.stripplot(data=selected, x="variant", y="regional_macro_f1", hue="variant", palette=MODEL_PALETTE, legend=False, jitter=0.08, size=7, ax=axes[1])
    sns.pointplot(data=selected, x="variant", y="regional_macro_f1", color="black", errorbar="sd", markers="_", linestyles="none", ax=axes[1])
    axes[1].set_title("Selected identification checkpoints")
    axes[1].set_ylabel("Regional macro-F1")
    axes[1].set_xlabel("")
    save(
        fig,
        "fig_task_seed_stability",
        "Main-run seed stability",
        "Individual seeds and mean ± standard deviation at the validation-selected checkpoints.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/best_selection.json"],
        selected,
    )

    norm_heat = norm_by_dialect.groupby(["variant", "dialect"], observed=True).chrfpp.mean().unstack()
    fig, axis = plt.subplots(figsize=(8.8, 3.0), constrained_layout=True)
    sns.heatmap(norm_heat, annot=True, fmt=".1f", cmap="mako", cbar_kws={"label": "chrF++"}, ax=axis)
    axis.set_title("Validation normalization by dialect (seed mean)")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Model")
    save(
        fig,
        "fig_task_norm_dialect_heatmap",
        "Validation normalization by dialect",
        "Seed-mean chrF++ at each run's selected normalization checkpoint; blank cells denote unsupported dialects.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_s/validation_by_dialect_epoch_*.csv"],
        norm_by_dialect,
    )

    id_heat = id_by_class.groupby(["variant", "dialect"], observed=True).f1.mean().unstack()
    fig, axis = plt.subplots(figsize=(9.0, 3.0), constrained_layout=True)
    sns.heatmap(id_heat, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="crest", cbar_kws={"label": "F1"}, ax=axis)
    axis.set_title("Validation identification by dialect (seed mean)")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Model")
    save(
        fig,
        "fig_task_id_class_heatmap",
        "Validation identification by class",
        "Seed-mean per-class F1 at validation-selected causal identification checkpoints.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_id/validation_by_class_epoch_*.csv"],
        id_by_class,
    )

    variants = sorted(matrices)
    fig, axes = plt.subplots(1, len(variants), figsize=(5.1 * len(variants), 4.1), constrained_layout=True, squeeze=False)
    labels = id_by_class.sort_values("label_id").dialect.drop_duplicates().tolist()
    matrix_export = []
    for axis, variant in zip(axes[0], variants):
        matrix = np.mean(matrices[variant], axis=0)
        sns.heatmap(matrix, vmin=0, vmax=1, cmap="mako", xticklabels=labels, yticklabels=labels, cbar=variant == variants[-1], ax=axis)
        axis.set_title(f"{variant}: row-normalized confusion")
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
        for true_index, true_label in enumerate(labels):
            for pred_index, pred_label in enumerate(labels):
                matrix_export.append({"variant": variant, "true": true_label, "predicted": pred_label, "rate": matrix[true_index, pred_index]})
    save(
        fig,
        "fig_task_id_confusion",
        "Validation identification confusion structure",
        "Seed-mean row-normalized confusion at selected causal checkpoints, shown only for computationally complete main variants.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_id/validation_confusion_epoch_*.npy"],
        pd.DataFrame(matrix_export),
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    sns.lineplot(data=reliability_frame, x="confidence", y="accuracy", hue="variant", palette=MODEL_PALETTE, marker="o", errorbar="sd", ax=axes[0])
    axes[0].plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Reliability diagram")
    axes[0].set_xlabel("Mean calibrated confidence")
    axes[0].set_ylabel("Empirical accuracy")
    sns.stripplot(data=selected, x="variant", y="ece_15", hue="variant", palette=MODEL_PALETTE, legend=False, jitter=0.08, size=7, ax=axes[1])
    axes[1].set_title("Expected calibration error")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("ECE-15 ↓")
    save(
        fig,
        "fig_task_calibration",
        "Validation calibration",
        "Reliability curves and ECE-15 for temperature-calibrated causal identification checkpoints.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/stage_id/validation_predictions_epoch_*.parquet"],
        reliability_frame,
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "elapsed_minutes", "Wall time (minutes) ↓"),
        (axes[1], "tokens_per_second", "Training tokens/s ↑"),
        (axes[2], "peak_memory_gib", "Peak memory (GiB) ↓"),
    ):
        sns.barplot(data=efficiency_frame, x="variant", y=metric, hue="stage", errorbar="sd", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        if axis is not axes[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    axes[0].legend(title="Stage", frameon=False)
    save(
        fig,
        "fig_task_training_efficiency",
        "Task-stage training efficiency",
        "Wall time, throughput, and peak memory for completed normalization and identification stages.",
        "systems",
        ["runs/task/boichitro_q1_v1/*/training_report.json"],
        efficiency_frame,
    )

    plot_loss = loss_frame.loc[loss_frame.stage.isin(["stage_a", "stage_s", "stage_id"])].copy()
    plot_loss["smoothed_loss"] = plot_loss.groupby(["variant", "seed", "stage"], observed=True).loss.transform(lambda x: x.rolling(5, min_periods=1).mean())
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), constrained_layout=True)
    for axis, stage in zip(axes, ("stage_a", "stage_s", "stage_id")):
        subset = plot_loss.loc[plot_loss.stage.eq(stage)]
        sns.lineplot(data=subset, x="progress", y="smoothed_loss", hue="variant", palette=MODEL_PALETTE, errorbar="sd", ax=axis)
        axis.set_title(stage.replace("stage_", "Stage ").upper())
        axis.set_xlabel("Fraction of stage")
        axis.set_ylabel("Smoothed total loss")
        if axis is not axes[0] and axis.get_legend() is not None:
            axis.get_legend().remove()
    save(
        fig,
        "fig_task_stage_loss_components",
        "Task-stage optimization curves",
        "Seed-aggregated smoothed training loss over adaptation, normalization, and causal identification stages.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/train_log.jsonl"],
        plot_loss,
    )

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    sns.scatterplot(data=selected, x="stage_s_validation_id", y="macro_chrfpp", hue="variant", palette=MODEL_PALETTE, s=75, ax=axes[0])
    axes[0].set_title("Selected normalization checkpoint")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Macro chrF++")
    sns.scatterplot(data=selected, x="stage_id_validation_id", y="regional_macro_f1", hue="variant", palette=MODEL_PALETTE, s=75, legend=False, ax=axes[1])
    axes[1].set_title("Selected identification checkpoint")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Regional macro-F1")
    save(
        fig,
        "fig_validation_checkpoint_selection",
        "Validation checkpoint selection",
        "Selected checkpoint locations and objectives across seeds, documenting early-selection variation without test access.",
        "task-development",
        ["runs/task/boichitro_q1_v1/*/best_selection.json"],
        selected,
    )


def classical_dialect_figures() -> None:
    norm_rows = []
    fair_models = {
        "N_COPY_SOURCE_BLIND": "Copy (source-blind)",
        "N_WORD_REWRITE_POOLED_SOURCE_BLIND": "Pooled rewrite (source-blind)",
        "N_WORD_REWRITE_INFERRED_SUPPORTED_DIALECT": "Inferred rewrite (source-blind)",
        "N_WORD_REWRITE_ORACLE_GOLD_DIALECT": "Rewrite (gold-dialect oracle)",
    }
    for model, label in fair_models.items():
        path = PROJECT / f"metrics/development_source_blind/{model}_validation_by_dialect.csv"
        if path.exists():
            norm_rows.append(
                pd.read_csv(path).assign(model=label, split="validation")
            )
    for model, label in (
        ("N_COPY", "Copy (source-blind)"),
        ("N_WORD_REWRITE", "Rewrite (gold-dialect oracle)"),
    ):
        for split in ("iid_test", "source_ood", "zero_shot_raj"):
            path = PROJECT / f"metrics/normalization/{model}_{split}_by_dialect.csv"
            if path.exists():
                norm_rows.append(pd.read_csv(path).assign(model=label, split=split))
    normalization = pd.concat(norm_rows, ignore_index=True)
    heat = normalization.groupby(["model", "split", "dialect"], observed=True).chrfpp.mean().unstack()
    fig, axis = plt.subplots(figsize=(9.2, 4.2), constrained_layout=True)
    sns.heatmap(heat, annot=True, fmt=".1f", cmap="mako", cbar_kws={"label": "chrF++"}, ax=axis)
    axis.set_title("Source-blind controls and disclosed gold-dialect oracle")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Model / track")
    save(
        fig,
        "fig_classical_norm_dialect",
        "Classical normalization error structure",
        "Per-dialect chrF++ for source-blind controls. Legacy word-rewrite test values use gold dialect and are labeled only as oracle evidence.",
        "baseline",
        [
            "metrics/development_source_blind/*_by_dialect.csv",
            "metrics/normalization/*_by_dialect.csv",
        ],
        normalization,
        evidence="prior_test_descriptive",
    )

    id_rows = []
    for split in ("iid_test", "source_ood", "external_transcript"):
        path = PROJECT / f"metrics/identification/ID_CHAR_TFIDF_SVM_{split}_per_class.csv"
        if path.exists():
            id_rows.append(pd.read_csv(path).assign(split=split))
    identification = pd.concat(id_rows, ignore_index=True)
    heat = identification.pivot_table(index="split", columns="dialect", values="f1")
    fig, axis = plt.subplots(figsize=(9.0, 3.0), constrained_layout=True)
    sns.heatmap(heat, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="crest", cbar_kws={"label": "F1"}, ax=axis)
    axis.set_title("Character-SVM identification by class")
    axis.set_xlabel("Dialect")
    axis.set_ylabel("Track")
    save(
        fig,
        "fig_classical_id_class",
        "Classical identification error structure",
        "Per-class F1 for the selected character n-gram SVM across IID, source-OOD, and external-transcript tracks.",
        "baseline",
        ["metrics/identification/ID_CHAR_TFIDF_SVM_*_per_class.csv"],
        identification,
        evidence="prior_test_descriptive",
    )


def development_fusion_figure() -> None:
    norm_path = PROJECT / "reports/model/normalization_fusion_selection_v2.json"
    id_path = PROJECT / "reports/model/id_fusion_selection.json"
    if not norm_path.exists() or not id_path.exists():
        return
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    ident = json.loads(id_path.read_text(encoding="utf-8"))
    if (
        norm.get("status") != "COMPLETE_VALIDATION_ONLY"
        or ident.get("status") != "COMPLETE_VALIDATION_ONLY"
    ):
        return
    rows = []
    for run in norm["runs"]:
        variant, seed = str(run["variant"]), int(run["seed"])
        raw = json.loads(
            (
                PROJECT
                / f"runs/task/boichitro_q1_v1/{variant}__base/{seed}/stage_s/best_selection.json"
            ).read_text(encoding="utf-8")
        )["validation"]["macro_chrfpp"]
        run_id = f"{variant}-{seed}"
        rows.extend(
            [
                {"task": "normalization", "run": run_id, "system": "Raw neural", "value": float(raw)},
                {
                    "task": "normalization",
                    "run": run_id,
                    "system": "Fair rewrite",
                    "value": float(norm["baseline"]["macro_chrfpp"]),
                },
                {
                    "task": "normalization",
                    "run": run_id,
                    "system": "Fixed fusion",
                    "value": float(run["macro_chrfpp"]),
                },
            ]
        )
    id_baselines = pd.read_csv(
        PROJECT / "reports/model/classical_identification_results.csv"
    )
    svm = float(
        id_baselines.loc[
            id_baselines["model_id"].eq("ID_CHAR_TFIDF_SVM")
            & id_baselines["split"].eq("validation"),
            "regional_macro_f1",
        ].iloc[0]
    )
    for run in ident["runs"]:
        variant, seed = str(run["variant"]), int(run["seed"])
        raw = json.loads(
            (
                PROJECT
                / f"runs/task/boichitro_q1_v1/{variant}__base/{seed}/stage_id/best_selection.json"
            ).read_text(encoding="utf-8")
        )["validation"]["regional_macro_f1"]
        run_id = f"{variant}-{seed}"
        rows.extend(
            [
                {"task": "identification", "run": run_id, "system": "Raw neural", "value": float(raw)},
                {"task": "identification", "run": run_id, "system": "Character SVM", "value": svm},
                {
                    "task": "identification",
                    "run": run_id,
                    "system": "Fixed fusion",
                    "value": float(run["regional_macro_f1"]),
                },
            ]
        )
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5), constrained_layout=True)
    contracts = (
        ("normalization", ["Raw neural", "Fair rewrite", "Fixed fusion"], "Macro chrF++"),
        (
            "identification",
            ["Raw neural", "Character SVM", "Fixed fusion"],
            "Regional macro-F1",
        ),
    )
    for axis, (task, order, ylabel) in zip(axes, contracts):
        subset = frame.loc[frame["task"].eq(task)]
        for _, group in subset.groupby("run", sort=True):
            aligned = group.set_index("system").reindex(order)
            axis.plot(order, aligned["value"], color="#BBBBBB", linewidth=0.7, alpha=0.7)
            axis.scatter(order, aligned["value"], color="#4C78A8", s=15, alpha=0.75)
        sns.pointplot(
            data=subset,
            x="system",
            y="value",
            order=order,
            color="#8B1A1A",
            errorbar="sd",
            markers="D",
            linestyles="none",
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel(ylabel)
        axis.set_title(task.capitalize())
        axis.tick_params(axis="x", rotation=15)
    save(
        fig,
        "fig_development_source_blind_fusion",
        "Development-only source-blind fusion gains",
        "Six paired M0/M1 runs. Normalization uses semantic-group-held-out selector predictions; identification uses the fixed calibrated neural/SVM blend. Settings were selected on development data, so this is exploratory evidence. Diamonds show mean ± seed SD.",
        "task-development",
        [
            "reports/model/normalization_fusion_selection_v2.json",
            "reports/model/id_fusion_selection.json",
            "reports/model/development_fusion_uncertainty.json",
        ],
        frame,
    )


def optional_locked_figures(protocol: str, suffix: str, seeds: list[int]) -> None:
    functions = [
        lambda: paper.main_result_figures(OUTPUT, DATA, protocol, suffix, seeds),
        lambda: paper.source_blind_system_fusion_figure(
            OUTPUT, DATA, protocol, suffix, seeds
        ),
        lambda: paper.routing_figure(OUTPUT, DATA, protocol, suffix, seeds),
        lambda: paper.routing_specialization_figure(OUTPUT, DATA, protocol),
        lambda: paper.robustness_figure(OUTPUT, DATA),
        lambda: paper.factorial_pilot_figure(OUTPUT, DATA),
        lambda: paper.ablation_delta_figure(OUTPUT, DATA, protocol, seeds),
        lambda: paper.calibration_confusion_figure(OUTPUT, DATA, protocol, suffix, seeds),
        lambda: paper.external_baseline_figure(OUTPUT, DATA, protocol, seeds),
        lambda: paper.task_efficiency_figure(OUTPUT, DATA, protocol, seeds),
        lambda: paper.statistical_effects_figure(OUTPUT, DATA, protocol),
        lambda: paper.human_evaluation_figure(OUTPUT, DATA),
    ]
    titles = {
        "fig_main_results": ("Locked main results", "Locked-test normalization and identification results across registered models and tracks."),
        "fig_source_blind_system_fusion": (
            "Source-blind system fusion",
            "Raw neural architecture evidence, fixed development-selected fusion, and fair source-blind baselines on locked source-OOD tracks.",
        ),
        "fig_dialect_heatmap": ("Locked dialect results", "Seed-mean source-OOD normalization by dialect."),
        "fig_routing_heatmaps": ("Locked routing loads", "Representative expert-load distributions from locked source-OOD inference."),
        "fig_routing_specialization": ("Expert specialization", "Dialect and task routing specialization across sparse layers."),
        "fig_robustness_curves": ("Perturbation robustness", "Locked performance under registered perturbation families and severities."),
        "fig_fusion_robustness": (
            "Source-blind fusion robustness",
            "M3 raw neural, fixed fused, and fair baseline performance across registered perturbation families and severities.",
        ),
        "fig_factorial_pilot": ("Factorial pilot effects", "Development-only main effects from the registered 2^4 routing study."),
        "fig_ablation_deltas": ("Confirmatory ablation deltas", "Locked source-OOD deltas from full M3 across registered ablations."),
        "fig_calibration_confusion": ("Locked calibration and confusion", "M3 reliability and source-OOD confusion after frozen calibration."),
        "fig_external_baselines": ("External model baselines", "Source-OOD comparison with pinned external normalization and identification models."),
        "fig_task_efficiency": ("End-to-end task efficiency", "Locked source-OOD quality versus measured inference throughput."),
        "fig_confirmatory_effects": ("Confirmatory statistical effects", "Hierarchical paired-bootstrap effects and confidence intervals."),
        "fig_human_evaluation": ("Blinded native-speaker evaluation", "Native-speaker ratings with confidence intervals; present only after completed rating sheets."),
    }
    alternate_source_data = {
        "fig_main_results": DATA / "fig_main_normalization.csv",
        "fig_routing_heatmaps": DATA / "fig_routing_load.csv",
        "fig_factorial_pilot": DATA / "fig_factorial_pilot_cells.csv",
        "fig_calibration_confusion": DATA / "fig_calibration_reliability.csv",
    }
    alternate_sources = {
        "fig_main_results": [
            "figures/q1/source_data/fig_main_normalization.csv",
            "figures/q1/source_data/fig_main_identification.csv",
        ],
        "fig_factorial_pilot": [
            "figures/q1/source_data/fig_factorial_pilot_cells.csv",
            "figures/q1/source_data/fig_factorial_pilot_effects.csv",
        ],
        "fig_calibration_confusion": [
            "figures/q1/source_data/fig_calibration_reliability.csv",
            "figures/q1/source_data/fig_source_ood_confusion.csv",
        ],
    }
    seen = {record["id"] for record in RECORDS}
    for function in functions:
        for name in function():
            if name in seen:
                continue
            png, pdf = OUTPUT / f"{name}.png", OUTPUT / f"{name}.pdf"
            if not png.exists() or not pdf.exists():
                continue
            title, caption = titles[name]
            source = DATA / f"{name}.csv"
            if not source.exists() and name in alternate_source_data:
                source = alternate_source_data[name]
            RECORDS.append(
                {
                    "id": name,
                    "title": title,
                    "caption": caption,
                    "category": "locked-results" if name != "fig_factorial_pilot" else "task-development",
                    "evidence": "locked_test" if name != "fig_factorial_pilot" else "development_only",
                    "png": str(png.relative_to(PROJECT)),
                    "pdf": str(pdf.relative_to(PROJECT)),
                    "source_data": str(source.relative_to(PROJECT)) if source.exists() else None,
                    "sources": alternate_sources.get(
                        name, ["registered publication pipeline outputs"]
                    ),
                    "png_bytes": png.stat().st_size,
                    "pdf_bytes": pdf.stat().st_size,
                    "png_sha256": sha256(png),
                    "pdf_sha256": sha256(pdf),
                }
            )
            seen.add(name)


def write_manifest(minimum: int, protocol: str, seeds: list[int]) -> None:
    unique = {record["id"]: record for record in RECORDS}
    records = list(unique.values())
    for index, record in enumerate(records, start=1):
        record["figure_number"] = index
    if len(records) < minimum:
        raise RuntimeError(f"Only {len(records)} complete PNG/PDF figure pairs; required {minimum}")
    manifest = {
        "status": "PASS",
        "figure_pairs": len(records),
        "minimum_required": minimum,
        "protocol": protocol,
        "seeds": seeds,
        "format_contract": {
            "png_dpi": 600,
            "pdf": "vector with TrueType fonts",
            "paired_exports_required": True,
            "source_data_required_for_empirical_figures": True,
        },
        "evidence_counts": pd.Series([row["evidence"] for row in records]).value_counts().to_dict(),
        "figures": records,
    }
    (OUTPUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = ["# Q1 figure captions", "", f"Complete PNG/PDF pairs: **{len(records)}**", ""]
    for record in records:
        lines.extend(
            [
                f"## Figure {record['figure_number']}. {record['title']}",
                "",
                record["caption"],
                "",
                f"Evidence status: `{record['evidence']}`. Figure ID: `{record['id']}`.",
                "",
            ]
        )
    (OUTPUT / "FIGURE_CAPTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "PASS", "figure_pairs": len(records), "manifest": str(OUTPUT / "figure_manifest.json")}, indent=2))


def main() -> None:
    args = parse_args()
    style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    build_core_figures()
    data_figures()
    tokenizer_figures()
    pretraining_figures()
    foundation_figures()
    task_figures()
    classical_dialect_figures()
    development_fusion_figure()
    optional_locked_figures(args.protocol, args.suffix, args.seeds)
    write_manifest(args.minimum, args.protocol, args.seeds)


if __name__ == "__main__":
    main()
