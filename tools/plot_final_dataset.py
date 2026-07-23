#!/usr/bin/env python3
"""Create camera-ready descriptive figures for Boichitro Data v1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT = Path(__file__).resolve().parents[1]
FINAL = PROJECT / "data" / "final" / "v1"
OUTPUT = PROJECT / "figures" / "data"
REPORT = PROJECT / "reports" / "DATA_FIGURES.md"
DIALECT_ORDER = [
    "BAR",
    "CHI",
    "KHU",
    "KIS",
    "MYM",
    "NAR",
    "NOA",
    "NSD",
    "RAJ",
    "RAN",
    "SYL",
    "TAN",
    "STD",
]


def save(fig: plt.Figure, stem: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def normalization_composition(frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame["partition"] = np.select(
        [
            frame["is_synthetic"],
            frame["split"].eq("train"),
            frame["split"].eq("validation"),
            frame["split"].eq("test"),
            frame["split"].eq("test_ood"),
        ],
        ["Synthetic train", "Authentic train", "Validation", "IID test", "Source OOD"],
        default="Other",
    )
    partitions = [
        "Authentic train",
        "Synthetic train",
        "Validation",
        "IID test",
        "Source OOD",
    ]
    dialects = [label for label in DIALECT_ORDER if label in set(frame["dialect"])]
    table = (
        frame.groupby(["dialect", "partition"]).size().unstack(fill_value=0)
        .reindex(index=dialects, columns=partitions, fill_value=0)
    )
    colors = ["#1b6ca8", "#80b1d3", "#fdb462", "#fb8072", "#6a3d9a"]
    fig, ax = plt.subplots(figsize=(10.8, 5.4))
    table.plot(kind="bar", stacked=True, color=colors, width=0.78, ax=ax)
    ax.set_title("Boichitro normalization corpus by dialect and evaluation role", pad=46)
    ax.set_xlabel("Dialect label")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title=None,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "figure_d1_normalization_composition")


def identification_composition(frame: pd.DataFrame) -> None:
    splits = ["train", "validation", "test", "test_external", "test_ood"]
    labels = [label for label in DIALECT_ORDER if label in set(frame["dialect"])]
    table = (
        frame.groupby(["dialect", "split"]).size().unstack(fill_value=0)
        .reindex(index=labels, columns=splits, fill_value=0)
    )
    colors = ["#238b45", "#a1d99b", "#fdae6b", "#de2d26", "#756bb1"]
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    table.plot(kind="bar", stacked=True, color=colors, width=0.78, ax=ax)
    ax.set_title("Thirteen-label identification corpus after conflict removal", pad=46)
    ax.set_xlabel("12 regional labels + Standard Bangla")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        title=None,
        frameon=False,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "figure_d2_identification_composition")


def source_dialect_heatmap(frame: pd.DataFrame) -> None:
    authentic = frame[~frame["is_synthetic"]]
    table = authentic.groupby(["source_id", "dialect"]).size().unstack(fill_value=0)
    columns = [label for label in DIALECT_ORDER if label in table.columns]
    table = table.reindex(columns=columns)
    display = np.log10(table + 1)
    height = max(5.0, 0.48 * len(table) + 1.8)
    fig, ax = plt.subplots(figsize=(9.7, height))
    sns.heatmap(
        display,
        cmap="mako",
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "log10(rows + 1)"},
        ax=ax,
    )
    for y, source in enumerate(table.index):
        for x, dialect in enumerate(table.columns):
            value = int(table.loc[source, dialect])
            if value:
                ax.text(x + 0.5, y + 0.5, f"{value:,}", ha="center", va="center", fontsize=7.2, color="white" if display.loc[source, dialect] > 2.7 else "black")
    ax.set_title("Authentic normalization coverage by source and dialect", pad=12)
    ax.set_xlabel("Dialect label")
    ax.set_ylabel("Immutable source release")
    fig.tight_layout()
    save(fig, "figure_d3_source_dialect_coverage")


def exclusion_profile(frame: pd.DataFrame) -> None:
    counts = (
        frame.assign(decision=frame["task"] + ": " + frame["reason"])
        .groupby("decision")
        .size()
        .sort_values(ascending=False)
        .head(12)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    bars = ax.barh(counts.index, counts.values, color="#4472a8")
    ax.bar_label(bars, labels=[f"{value:,}" for value in counts.values], padding=4, fontsize=8)
    ax.set_title("Largest exclusion decisions in the frozen data gate", pad=12)
    ax.set_xlabel("Row-level decisions (log scale)")
    ax.set_xscale("log")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    fig.tight_layout()
    save(fig, "figure_d4_exclusion_profile")


def main() -> None:
    sns.set_theme(context="paper", style="white", font_scale=1.08)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    normalization = pd.read_parquet(FINAL / "normalization_all.parquet")
    identification = pd.read_parquet(FINAL / "identification_all.parquet")
    exclusions = pd.read_parquet(FINAL / "excluded_rows.parquet")
    normalization_composition(normalization)
    identification_composition(identification)
    source_dialect_heatmap(normalization)
    exclusion_profile(exclusions)
    REPORT.write_text(
        """# Dataset figure index

All figures are generated from the frozen Parquet manifests by
`tools/plot_final_dataset.py`. PNG files are 320 dpi; matching vector PDFs are
provided for manuscript typesetting.

- Figure D1: normalization rows by dialect, authenticity, and evaluation role.
- Figure D2: final 13-label identification distribution after conflict removal.
- Figure D3: authentic normalization source-by-dialect coverage (cell labels are raw counts; color is log-scaled).
- Figure D4: the twelve largest row-level exclusion decisions (log-scaled axis).

These are descriptive data figures, not model-result figures.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
