#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
MODEL_ORDER = ["M0", "M1", "M2", "M3"]
PALETTE = {"M0": "#4C78A8", "M1": "#F58518", "M2": "#54A24B", "M3": "#B279A2"}
DIALECT_LABELS = ["BAR", "CHI", "KHU", "KIS", "MYM", "NAR", "NOA", "NSD", "RAJ", "RAN", "STD", "SYL", "TAN"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create camera-ready experiment figures and source tables.")
    parser.add_argument("--protocol", default="locked_test_v1")
    parser.add_argument("--suffix", default="base")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1701, 2903, 4307])
    return parser.parse_args()


def style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: plt.Figure, output: Path, name: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def preliminary_figures(output: Path, data_dir: Path) -> list[str]:
    created = []
    tokenizer = pd.read_csv(PROJECT / "reports/tokenizer/tokenizer_proxy_summary.csv")
    tokenizer.to_csv(data_dir / "fig_tokenizer_tradeoff.csv", index=False)
    fig, axis = plt.subplots(figsize=(5.0, 3.2))
    display_names = {
        "wordpiece_natural_32k": "WP-natural 32k",
        "wordpiece_balanced_32k": "WP-balanced 32k",
        "unigram_balanced_32k": "Unigram 32k",
        "byte_bpe_balanced_16k": "Byte-BPE 7.2k",
    }
    offsets = {
        "wordpiece_natural_32k": (12, -7),
        "wordpiece_balanced_32k": (12, 14),
        "unigram_balanced_32k": (8, 8),
        "byte_bpe_balanced_16k": (-76, 8),
    }
    for row in tokenizer.itertuples(index=False):
        selected = bool(row.selected)
        axis.errorbar(
            row.tokens_per_character,
            row.mean_bpc,
            yerr=row.std_bpc,
            fmt="*" if selected else "o",
            markersize=11 if selected else 6,
            color="#B279A2" if selected else "#4C78A8",
            capsize=3,
        )
        axis.annotate(
            display_names[row.candidate_id],
            (row.tokens_per_character, row.mean_bpc),
            xytext=offsets[row.candidate_id],
            textcoords="offset points",
            fontsize=7,
        )
    axis.set_xlabel("Tokens per character (lower is better)")
    axis.set_ylabel("Held-out proxy BPC (lower is better)")
    axis.set_title("Tokenizer efficiency–quality frontier")
    save(fig, output, "fig_tokenizer_tradeoff")
    created.append("fig_tokenizer_tradeoff")

    benchmark = pd.read_csv(PROJECT / "reports/model/gb10_model_benchmark.csv")
    benchmark.to_csv(data_dir / "fig_compute_pareto.csv", index=False)
    fig, axis = plt.subplots(figsize=(5.0, 3.2))
    labels = [value.split("_", 1)[0] for value in benchmark["model_id"]]
    label_offsets = {"M0": (4, 3), "M1": (4, 3), "M2": (8, 8), "M3": (8, -15)}
    for label, row in zip(labels, benchmark.itertuples(index=False)):
        axis.scatter(
            row.tokens_per_second,
            row.total_parameters / 1e6,
            s=65,
            color=PALETTE[label],
            edgecolor="white",
            linewidth=0.7,
        )
        axis.annotate(
            label,
            (row.tokens_per_second, row.total_parameters / 1e6),
            xytext=label_offsets[label],
            textcoords="offset points",
        )
    axis.set_xlabel("Training throughput (tokens/s; higher is better)")
    axis.set_ylabel("Total parameters (millions)")
    axis.set_title("GB10 sparse-model systems trade-off")
    save(fig, output, "fig_compute_pareto")
    created.append("fig_compute_pareto")

    identification = pd.read_csv(PROJECT / "reports/model/classical_identification_results.csv")
    identification = identification.loc[
        identification["model_id"].isin(["ID_CHAR_TFIDF_SVM", "ID_CHAR_TFIDF_SGD"])
    ].copy()
    summary = (
        identification.groupby(["model_id", "split"], as_index=False)["regional_macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(data_dir / "fig_classical_id_floor.csv", index=False)
    fig, axis = plt.subplots(figsize=(6.2, 3.2))
    sns.barplot(
        data=summary,
        x="split",
        y="mean",
        hue="model_id",
        order=["validation", "iid_test", "source_ood", "external_transcript"],
        ax=axis,
    )
    axis.set_ylabel("Regional macro-F1")
    axis.set_xlabel("")
    axis.tick_params(axis="x", rotation=15)
    axis.set_title("Classical identification floors expose source shift")
    axis.legend(title=None, frameon=False)
    save(fig, output, "fig_classical_id_floor")
    created.append("fig_classical_id_floor")

    normalization = pd.read_csv(PROJECT / "reports/model/classical_normalization_results.csv")
    normalization["display_id"] = normalization["model_id"].map(
        {
            "N_COPY": "Copy (source-blind)",
            "N_WORD_REWRITE": "Rewrite (gold-dialect oracle)",
        }
    )
    fair_path = PROJECT / "reports/model/source_blind_normalization_baselines.csv"
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        fair = fair.loc[
            fair["model_id"].isin(
                [
                    "N_WORD_REWRITE_POOLED_SOURCE_BLIND",
                    "N_WORD_REWRITE_INFERRED_SUPPORTED_DIALECT",
                ]
            )
        ].copy()
        fair["display_id"] = fair["model_id"].map(
            {
                "N_WORD_REWRITE_POOLED_SOURCE_BLIND": "Pooled rewrite (source-blind)",
                "N_WORD_REWRITE_INFERRED_SUPPORTED_DIALECT": "Inferred-dialect rewrite (source-blind)",
            }
        )
        normalization = pd.concat([normalization, fair], ignore_index=True, sort=False)
    normalization.to_csv(data_dir / "fig_classical_norm_floor.csv", index=False)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.2, 3.4),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )
    validation = normalization.loc[normalization["split"].eq("validation")].copy()
    validation["short_label"] = validation["display_id"].map(
        {
            "Copy (source-blind)": "Copy",
            "Pooled rewrite (source-blind)": "Pooled\nrewrite",
            "Inferred-dialect rewrite (source-blind)": "Inferred\nrewrite",
            "Rewrite (gold-dialect oracle)": "Gold-dialect\noracle",
        }
    )
    short_order = ["Copy", "Pooled\nrewrite", "Inferred\nrewrite", "Gold-dialect\noracle"]
    palette = ["#4C78A8", "#59A14F", "#B55A60", "#D08C60"]
    sns.barplot(
        data=validation,
        x="short_label",
        y="macro_chrfpp",
        order=short_order,
        hue="short_label",
        hue_order=short_order,
        palette=palette,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_ylabel("Macro chrF++")
    axes[0].set_xlabel("")
    axes[0].set_title("Deployable validation controls")

    legacy = normalization.loc[
        normalization["model_id"].isin(["N_COPY", "N_WORD_REWRITE"])
        & normalization["split"].isin(["iid_test", "source_ood", "zero_shot_raj"])
    ].copy()
    legacy["track"] = legacy["split"].map(
        {
            "iid_test": "IID test",
            "source_ood": "Source OOD",
            "zero_shot_raj": "RAJ zero-shot",
        }
    )
    legacy["system"] = legacy["model_id"].map(
        {"N_COPY": "Copy", "N_WORD_REWRITE": "Gold-dialect oracle"}
    )
    sns.barplot(
        data=legacy,
        x="track",
        y="macro_chrfpp",
        hue="system",
        order=["IID test", "Source OOD", "RAJ zero-shot"],
        hue_order=["Copy", "Gold-dialect oracle"],
        palette=["#4C78A8", "#D08C60"],
        ax=axes[1],
    )
    axes[1].set_ylabel("")
    axes[1].set_xlabel("")
    axes[1].set_title("Legacy test diagnostics (disclosed)")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(
        title=None,
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
    )
    fig.suptitle("Fair source-blind controls and the disclosed gold-dialect oracle")
    fig.subplots_adjust(top=0.78, bottom=0.23, wspace=0.18)
    save(fig, output, "fig_classical_norm_floor")
    created.append("fig_classical_norm_floor")
    return created


def continuation_lr_stability_figure(output: Path, data_dir: Path) -> list[str]:
    path = PROJECT / "reports/model/continuation_lr_pilot_selection.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE_VALIDATION_ONLY":
        return []
    rows = [
        {
            "candidate": row["id"],
            "muon_lr": float(row["muon_lr"]),
            "adamw_lr": float(row["adamw_lr"]),
            "final_relative_bpc_change_percent": 100.0
            * float(row["final_relative_bpc_regression"]),
            "eligible": bool(row["eligible"]),
            "selected": row["id"] == payload["selected_candidate"],
            "schedule": "no restart warmup",
        }
        for row in payload["candidates"]
    ]
    trigger = payload["rejected_high_lr_trigger"]
    rows.append(
        {
            "candidate": "rejected_high_lr_restart",
            "muon_lr": float(trigger["muon_lr"]),
            "adamw_lr": float(trigger["adamw_lr"]),
            "final_relative_bpc_change_percent": 100.0
            * float(trigger["relative_bpc_regression"]),
            "eligible": False,
            "selected": False,
            "schedule": "2% restart warmup",
        }
    )
    frame = pd.DataFrame(rows).sort_values("muon_lr")
    frame.to_csv(data_dir / "fig_continuation_lr_stability.csv", index=False)
    fig, axis = plt.subplots(figsize=(5.6, 3.3))
    for row in frame.itertuples(index=False):
        color = "#54A24B" if row.eligible else "#E45756"
        marker = "*" if row.selected else ("X" if "restart" in row.candidate else "o")
        axis.scatter(
            row.muon_lr,
            row.final_relative_bpc_change_percent,
            color=color,
            marker=marker,
            s=110 if row.selected else 58,
            zorder=3,
        )
        axis.annotate(
            (
                f"{row.muon_lr:g} selected"
                if row.selected
                else (
                    f"{row.muon_lr:g} restart"
                    if "restart" in row.candidate
                    else f"{row.muon_lr:g}"
                )
            ),
            (row.muon_lr, row.final_relative_bpc_change_percent),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("Muon continuation learning rate (log scale)")
    axis.set_ylabel("Final validation BPC change (%)\n(lower is better)")
    axis.set_title("Mature-checkpoint continuation requires a lower LR")
    save(fig, output, "fig_continuation_lr_stability")
    return ["fig_continuation_lr_stability"]


def stage_s_retention_figure(output: Path, data_dir: Path) -> list[str]:
    path = PROJECT / "reports/model/stage_s_retention_pilot_selection.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE_VALIDATION_ONLY":
        return []
    rows = []
    for candidate in payload["candidates"]:
        for validation in candidate["validation_curve"]:
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source": candidate["source"],
                    "optimizer_step": int(validation["optimizer_step"]),
                    "macro_chrfpp": float(validation["macro_chrfpp"]),
                    "worst_dialect_chrfpp": float(
                        validation["worst_dialect_chrfpp"]
                    ),
                    "replay_degradation_percent": 100.0
                    * float(validation["replay_relative_degradation"]),
                    "replay_guard_pass": bool(validation["replay_guard_pass"]),
                    "selected_schedule": candidate["candidate_id"]
                    == payload["selected_candidate_id"],
                    "selected_checkpoint": (
                        candidate["candidate_id"]
                        == payload["selected_candidate_id"]
                        and int(validation["optimizer_step"])
                        == int(payload["selected_validation"]["optimizer_step"])
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "fig_stage_s_retention_tradeoff.csv", index=False)
    colors = {
        "ret25_balanced": "#4C78A8",
        "ret25_conservative": "#72B7B2",
        "ret35_balanced": "#B279A2",
        "ret35_conservative": "#54A24B",
        "rejected_default": "#E45756",
    }
    labels = {
        "ret25_balanced": "25% replay, balanced LR",
        "ret25_conservative": "25% replay, conservative LR",
        "ret35_balanced": "35% replay, balanced LR (selected)",
        "ret35_conservative": "35% replay, conservative LR",
        "rejected_default": "10% replay, original LR (rejected)",
    }
    fig, (safe_axis, rejected_axis) = plt.subplots(
        1, 2, figsize=(9.2, 3.5), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    safe = frame.loc[frame["source"].eq("registered_candidate")]
    for candidate_id, subset in safe.groupby("candidate_id", sort=False):
        subset = subset.sort_values("optimizer_step")
        safe_axis.plot(
            subset["replay_degradation_percent"],
            subset["macro_chrfpp"],
            marker="o",
            markersize=4,
            linewidth=1.2,
            color=colors[candidate_id],
            label=labels[candidate_id],
        )
    selected = frame.loc[frame["selected_checkpoint"]].iloc[0]
    safe_axis.scatter(
        selected["replay_degradation_percent"],
        selected["macro_chrfpp"],
        marker="*",
        s=155,
        color=colors[str(selected["candidate_id"])],
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    safe_axis.annotate(
        f"selected\n{selected['macro_chrfpp']:.2f} chrF++",
        (selected["replay_degradation_percent"], selected["macro_chrfpp"]),
        xytext=(9, -28),
        textcoords="offset points",
        fontsize=7,
    )
    safe_axis.axvline(5.0, color="#E45756", linestyle="--", linewidth=1.0)
    safe_axis.set_xlim(0.0, 5.3)
    safe_axis.set_xlabel("Replay NLL degradation (%)")
    safe_axis.set_ylabel("Validation macro chrF++")
    safe_axis.set_title("Registered replay-safe schedules")
    safe_axis.legend(frameon=False, fontsize=7, loc="lower right")

    rejected = frame.loc[frame["candidate_id"].eq("rejected_default")].sort_values(
        "optimizer_step"
    )
    rejected_axis.plot(
        rejected["replay_degradation_percent"],
        rejected["macro_chrfpp"],
        marker="X",
        markersize=6,
        linestyle="--",
        linewidth=1.2,
        color=colors["rejected_default"],
    )
    rejected_axis.axvspan(
        5.0,
        max(18.0, rejected["replay_degradation_percent"].max() + 0.5),
        color="#E45756",
        alpha=0.08,
    )
    rejected_axis.text(
        0.5,
        0.05,
        "All checkpoints violate\nthe preregistered 5% guard",
        transform=rejected_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#9C2F2F",
    )
    rejected_axis.set_xlim(10.0, 18.0)
    rejected_axis.set_xlabel("Replay NLL degradation (%)")
    rejected_axis.set_ylabel("Validation macro chrF++")
    rejected_axis.set_title("Original schedule: task gain with forgetting")
    fig.suptitle(
        "Stage-S schedule selection is constrained by language-model retention",
        y=0.99,
    )
    fig.subplots_adjust(top=0.78, wspace=0.25)
    save(fig, output, "fig_stage_s_retention_tradeoff")
    return ["fig_stage_s_retention_tradeoff"]


def architecture_figure(output: Path, data_dir: Path) -> list[str]:
    specification = {
        "tokenizer": "Frozen WP-natural 32k",
        "backbone": "16-layer, d=512, GQA 8Q/2KV, RoPE, QK norm, RMSNorm",
        "dense_prefix_layers": 4,
        "moe_layers": 12,
        "experts": "1 shared + 8 routed, top-2, width 768, dropless",
        "early_router": "causal lexical/dialect curriculum (layers 5–8)",
        "late_router": "task-conditioned bias (layers 13–16)",
        "optimization": "Muon matrices + AdamW embeddings/norms/routers/heads",
    }
    (data_dir / "fig_architecture.json").write_text(
        json.dumps(specification, indent=2) + "\n", encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(10.5, 4.5))
    axis.set_xlim(0, 10.5)
    axis.set_ylim(0, 5.0)
    axis.axis("off")

    def box(x, y, width, height, text, color, fontsize=8):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#333333",
            linewidth=0.9,
        )
        axis.add_patch(patch)
        axis.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=fontsize)
        return patch

    def arrow(start, end, style="-|>", color="#444444"):
        axis.add_patch(
            FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=11, color=color, linewidth=1.0)
        )

    box(0.15, 2.05, 1.25, 0.85, "Source-blind input\n+ task token\n+ <dial_unknown>", "#DCEAF7")
    box(1.75, 2.05, 1.25, 0.85, "Frozen own\nWordPiece tokenizer\n32k vocabulary", "#DCEAF7")
    box(3.35, 2.05, 1.15, 0.85, "4 dense\ndecoder blocks", "#E8E8E8")
    box(4.85, 2.05, 1.45, 0.85, "MoE layers 5–8\ncausal dialect-evidence\ncurriculum", "#DDF1E1")
    box(6.65, 2.05, 1.25, 0.85, "MoE layers 9–12\nloss-free dynamic\nload bias", "#DDF1E1")
    box(8.25, 2.05, 1.45, 0.85, "MoE layers 13–16\ntask-conditioned\nrouter bias", "#DDF1E1")
    for left, right in ((1.4, 1.75), (3.0, 3.35), (4.5, 4.85), (6.3, 6.65), (7.9, 8.25)):
        arrow((left, 2.48), (right, 2.48))

    box(
        4.75,
        0.48,
        5.0,
        0.92,
        "Every sparse layer\n1 shared expert + top-2 of 8 routed experts\nDropless grouped BF16 GEMM",
        "#F7E8C6",
        7.5,
    )
    arrow((7.25, 1.4), (7.25, 2.02))
    box(4.85, 3.75, 1.4, 0.7, "Causal dialect head\n(router supervision)", "#F3DDE8")
    box(
        6.45,
        3.75,
        1.25,
        0.7,
        "Source head + GRL\n(domain invariance)",
        "#F3DDE8",
        7.0,
    )
    box(7.9, 3.75, 1.1, 0.7, "LM + MTP\nnormalization", "#F3DDE8")
    box(9.2, 3.75, 1.1, 0.7, "CLS head\ndialect ID", "#F3DDE8")
    arrow((5.55, 2.93), (5.55, 3.72))
    arrow((7.05, 2.93), (7.05, 3.72))
    arrow((8.45, 2.93), (8.45, 3.72))
    arrow((9.42, 2.93), (9.65, 3.72))
    axis.text(
        0.2,
        4.55,
        "Boichitro: staged, task-aware sparse Bangla dialect SLM",
        fontsize=13,
        fontweight="bold",
    )
    axis.text(
        0.2,
        0.2,
        "Hybrid optimization: Muon for hidden 2-D matrices; AdamW for embeddings, norms, routers, task parameters, and heads",
        fontsize=8,
        color="#333333",
    )
    save(fig, output, "fig_architecture")
    return ["fig_architecture"]


def protocol_flow_figure(output: Path, data_dir: Path) -> list[str]:
    specification = {
        "development_sequence": [
            "audited data and frozen tokenizer",
            "300M-token dense foundation",
            "matched 200M-token M0/M1/M2 continuations",
            "12M-token source-blind dialect adaptation",
            "normalization, causal ID, and bidirectional ID branches",
            "validation-only ablations and selection",
            "immutable protocol freeze",
            "single scripted locked evaluation",
        ],
        "main_neural_test_access_before_freeze": "forbidden",
        "prior_access_disclosure": "fixed classical baselines and tiny pipeline smoke only",
        "seeds": [1701, 2903, 4307],
    }
    (data_dir / "fig_protocol_flow.json").write_text(
        json.dumps(specification, indent=2) + "\n", encoding="utf-8"
    )
    fig, axis = plt.subplots(figsize=(11.2, 4.7))
    axis.set_xlim(0, 11.2)
    axis.set_ylim(0, 4.7)
    axis.axis("off")

    def box(x, y, width, height, text, color, fontsize=7.4, linewidth=0.9):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.07",
            facecolor=color,
            edgecolor="#333333",
            linewidth=linewidth,
            zorder=2,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            zorder=3,
        )
        return patch

    def arrow(start, end, color="#444444", dashed=False):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color=color,
                linewidth=1.0,
                linestyle="--" if dashed else "-",
                zorder=1,
            )
        )

    box(0.15, 3.15, 1.25, 0.8, "Audited 13-label data\n+ 300M Bangla corpus\n+ frozen tokenizer", "#DCEAF7", 6.8)
    box(1.75, 3.15, 1.25, 0.8, "Stage F\nDense foundation\n300M tokens", "#E8E8E8")
    box(3.35, 3.15, 1.45, 0.8, "Stage U\nM0 dense · M1 Switch\nM2 shared top-2 MoE\n200M tokens each", "#DDF1E1", 7.0)
    box(5.15, 3.15, 1.35, 0.8, "Stage A\nM0–M3 dialect adapt\n12M fixed tokens", "#DDF1E1")
    box(7.75, 3.15, 1.35, 0.8, "Validation only\n3 seeds + factorial +\nconfirmatory ablations", "#F7E8C6", 7.0)
    box(9.45, 3.15, 0.75, 0.8, "Protocol\nfreeze", "#F3DDE8", 7.2, 1.2)
    box(10.45, 3.15, 0.6, 0.8, "Locked\ntests", "#DDF1E1", 7.2, 1.2)
    for left, right in ((1.4, 1.75), (3.0, 3.35), (4.8, 5.15), (9.1, 9.45), (10.2, 10.45)):
        arrow((left, 3.55), (right, 3.55))

    box(4.45, 1.45, 1.65, 0.85, "Stage S normalization\n6M fixed tokens\nchrF++ + replay guard", "#DCEAF7")
    box(6.25, 1.45, 1.25, 0.85, "Causal ID\n13 classes\nGroupDRO", "#DCEAF7")
    box(6.25, 0.25, 1.65, 0.8, "M3B ID specialist\n5M MNTP + cross-source\nsupervised contrastive", "#E8DDF3", 7.0)
    arrow((5.8, 3.13), (5.3, 2.32))
    arrow((6.05, 3.13), (6.85, 2.32))
    arrow((6.85, 1.43), (7.05, 1.07), color="#76528B")
    arrow((6.0, 2.28), (7.73, 3.3))
    arrow((7.4, 2.28), (8.2, 3.13))
    arrow((7.9, 0.65), (8.6, 3.13), color="#76528B")
    axis.text(
        0.2,
        4.35,
        "Leakage-safe execution protocol: development choices end at the freeze boundary",
        fontsize=12,
        fontweight="bold",
    )
    axis.text(
        9.83,
        2.8,
        "No test-driven\nmodel choice",
        ha="center",
        va="top",
        fontsize=7,
        color="#8B1A1A",
    )
    save(fig, output, "fig_protocol_flow")
    return ["fig_protocol_flow"]


def collect_main(protocol: str, suffix: str, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    norm_rows = []
    id_rows = []
    for variant in MODEL_ORDER:
        for seed in seeds:
            root = PROJECT / "predictions" / protocol / f"{variant}__{suffix}" / str(seed)
            manifest = root / "evaluation_manifest.json"
            if not manifest.exists():
                continue
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            for track, metrics in payload["normalization"].items():
                norm_rows.append({"variant": variant, "seed": seed, "track": track, **metrics})
            for track, metrics in payload["identification"].items():
                id_rows.append({"variant": variant, "seed": seed, "track": track, **metrics})
    return pd.DataFrame(norm_rows), pd.DataFrame(id_rows)


def main_result_figures(
    output: Path, data_dir: Path, protocol: str, suffix: str, seeds: list[int]
) -> list[str]:
    normalization, identification = collect_main(protocol, suffix, seeds)
    if normalization.empty or identification.empty:
        return []
    normalization.to_csv(data_dir / "fig_main_normalization.csv", index=False)
    identification.to_csv(data_dir / "fig_main_identification.csv", index=False)
    created = []
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    sns.pointplot(
        data=normalization,
        x="track",
        y="macro_chrfpp",
        hue="variant",
        hue_order=[value for value in MODEL_ORDER if value in set(normalization["variant"])],
        palette=PALETTE,
        errorbar="sd",
        dodge=0.35,
        markers="o",
        ax=axes[0],
    )
    axes[0].set_ylabel("Macro chrF++")
    axes[0].set_xlabel("")
    axes[0].set_title("Source-blind normalization")
    axes[0].legend(title=None, frameon=False)
    sns.pointplot(
        data=identification,
        x="track",
        y="regional_macro_f1",
        hue="variant",
        hue_order=[value for value in MODEL_ORDER if value in set(identification["variant"])],
        palette=PALETTE,
        errorbar="sd",
        dodge=0.35,
        markers="o",
        ax=axes[1],
    )
    axes[1].set_ylabel("Regional macro-F1")
    axes[1].set_xlabel("")
    axes[1].set_title("Dialect identification")
    axes[1].get_legend().remove()
    save(fig, output, "fig_main_results")
    created.append("fig_main_results")

    rows = []
    for variant in sorted(normalization["variant"].unique()):
        for seed in seeds:
            path = (
                PROJECT
                / "predictions"
                / protocol
                / f"{variant}__{suffix}"
                / str(seed)
                / "normalization_source_ood_by_dialect.csv"
            )
            if path.exists():
                frame = pd.read_csv(path)
                frame["variant"] = variant
                frame["seed"] = seed
                rows.append(frame)
    if rows:
        dialect = pd.concat(rows, ignore_index=True)
        heat = dialect.groupby(["variant", "dialect"])["chrfpp"].mean().unstack()
        heat.to_csv(data_dir / "fig_dialect_heatmap.csv")
        fig, axis = plt.subplots(figsize=(8.0, 2.8))
        sns.heatmap(heat, annot=True, fmt=".1f", cmap="mako", ax=axis, cbar_kws={"label": "chrF++"})
        axis.set_xlabel("Dialect")
        axis.set_ylabel("Model")
        axis.set_title("Source-OOD normalization by dialect (seed mean)")
        save(fig, output, "fig_dialect_heatmap")
        created.append("fig_dialect_heatmap")
    return created


def source_blind_system_fusion_figure(
    output: Path, data_dir: Path, protocol: str, suffix: str, seeds: list[int]
) -> list[str]:
    """Contrast raw models, frozen fusion, and fair source-blind baselines."""

    rows = []
    task_contracts = {
        "normalization": (
            "normalization",
            "normalization_fused",
            "normalization_source_blind_baseline",
            "macro_chrfpp",
        ),
        "identification": (
            "identification",
            "identification_fused",
            "identification_classical_baseline",
            "regional_macro_f1",
        ),
    }
    for variant in MODEL_ORDER:
        for seed in seeds:
            path = (
                PROJECT
                / "predictions"
                / protocol
                / f"{variant}__{suffix}"
                / str(seed)
                / "evaluation_manifest.json"
            )
            if not path.exists():
                return []
            payload = json.loads(path.read_text(encoding="utf-8"))
            for task, (raw_key, fused_key, baseline_key, metric) in task_contracts.items():
                for view, key in (("Raw neural", raw_key), ("Frozen fusion", fused_key)):
                    values = payload.get(key, {}).get("source_ood")
                    if values is None:
                        return []
                    rows.append(
                        {
                            "task": task,
                            "variant": variant,
                            "seed": seed,
                            "view": view,
                            "value": float(values[metric]),
                        }
                    )
                if variant == "M3":
                    values = payload.get(baseline_key, {}).get("source_ood")
                    if values is None:
                        return []
                    rows.append(
                        {
                            "task": task,
                            "variant": "Fixed baseline",
                            "seed": seed,
                            "view": "Source-blind baseline",
                            "value": float(values[metric]),
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "fig_source_blind_system_fusion.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4))
    order = [*MODEL_ORDER, "Fixed baseline"]
    palette = {
        "Raw neural": "#4C78A8",
        "Frozen fusion": "#54A24B",
        "Source-blind baseline": "#B279A2",
    }
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = frame.loc[frame["task"].eq(task)]
        sns.pointplot(
            data=subset,
            x="variant",
            y="value",
            hue="view",
            order=order,
            palette=palette,
            errorbar="sd",
            dodge=0.25,
            markers="o",
            ax=axis,
        )
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=18)
        axis.set_title(task.capitalize() + " source-OOD")
        axis.set_ylabel(
            "Macro chrF++" if task == "normalization" else "Regional macro-F1"
        )
        if task == "identification" and axis.get_legend() is not None:
            axis.get_legend().remove()
    axes[0].legend(title=None, frameon=False, fontsize=7)
    fig.suptitle("Raw architecture evidence and frozen source-blind system fusion")
    save(fig, output, "fig_source_blind_system_fusion")
    return ["fig_source_blind_system_fusion"]


def routing_figure(
    output: Path, data_dir: Path, protocol: str, suffix: str, seeds: list[int]
) -> list[str]:
    rows = []
    representative_seed = seeds[0]
    for variant in ("M2", "M3"):
        for seed in (representative_seed,):
            root = PROJECT / "predictions" / protocol / f"{variant}__{suffix}" / str(seed)
            for task in ("normalization", "identification"):
                path = root / f"routing_{task}_source_ood.parquet"
                if not path.exists():
                    continue
                frame = pd.read_parquet(path)
                for row in frame.itertuples(index=False):
                    counts = np.asarray(row.expert_counts, dtype=np.float64)
                    for expert, count in enumerate(counts):
                        rows.append(
                            {
                                "variant": variant,
                                "seed": seed,
                                "task": task,
                                "layer": row.layer_index + 1,
                                "expert": expert,
                                "count": count,
                            }
                        )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["variant", "task", "layer", "expert"], as_index=False)["count"].sum()
    grouped["fraction"] = grouped["count"] / grouped.groupby(
        ["variant", "task", "layer"]
    )["count"].transform("sum")
    grouped.to_csv(data_dir / "fig_routing_load.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.4), sharex=True, sharey=True)
    for row_index, variant in enumerate(("M2", "M3")):
        for column_index, task in enumerate(("normalization", "identification")):
            subset = grouped.loc[(grouped["variant"] == variant) & (grouped["task"] == task)]
            pivot = subset.pivot(index="layer", columns="expert", values="fraction")
            sns.heatmap(pivot, cmap="rocket", vmin=0.0, vmax=max(0.25, grouped["fraction"].max()), ax=axes[row_index, column_index], cbar=row_index == 0 and column_index == 1)
            axes[row_index, column_index].set_title(f"{variant}: {task}")
            axes[row_index, column_index].set_xlabel("Expert")
            axes[row_index, column_index].set_ylabel("Layer")
    fig.suptitle(f"Representative locked routing traces (seed {representative_seed})", y=1.02)
    save(fig, output, "fig_routing_heatmaps")
    return ["fig_routing_heatmaps"]


def routing_specialization_figure(output: Path, data_dir: Path, protocol: str) -> list[str]:
    root = PROJECT / "reports/routing" / protocol
    metric_path = root / "expert_specialization_metrics.parquet"
    task_path = root / "task_routing_divergence.csv"
    if not metric_path.exists() or not task_path.exists():
        return []
    metrics = pd.read_parquet(metric_path)
    metrics["display_layer"] = metrics["layer"] + 1
    task = pd.read_csv(task_path)
    task["display_layer"] = task["layer"] + 1
    metrics.drop(columns=["dialect_labels"]).to_csv(
        data_dir / "fig_routing_specialization.csv", index=False
    )
    task.to_csv(data_dir / "fig_task_routing_divergence.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), sharex=True)
    sns.lineplot(
        data=metrics,
        x="display_layer",
        y="normalized_mutual_information",
        hue="variant",
        style="task",
        markers=True,
        errorbar="sd",
        palette=PALETTE,
        ax=axes[0],
    )
    axes[0].set_title("Dialect–expert NMI")
    axes[0].set_ylabel("Normalized MI")
    axes[0].set_xlabel("Transformer layer")
    axes[0].legend(title=None, fontsize=6, frameon=False)
    sns.lineplot(
        data=metrics,
        x="display_layer",
        y="mean_pairwise_dialect_js_divergence",
        hue="variant",
        style="task",
        markers=True,
        errorbar="sd",
        palette=PALETTE,
        ax=axes[1],
    )
    axes[1].set_title("Dialect routing separation")
    axes[1].set_ylabel("Mean pairwise JS divergence")
    axes[1].set_xlabel("Transformer layer")
    axes[1].get_legend().remove()
    sns.lineplot(
        data=task,
        x="display_layer",
        y="task_js_divergence",
        hue="variant",
        markers=True,
        errorbar="sd",
        palette=PALETTE,
        ax=axes[2],
    )
    axes[2].set_title("Normalization–ID separation")
    axes[2].set_ylabel("Task JS divergence")
    axes[2].set_xlabel("Transformer layer")
    axes[2].legend(title=None, frameon=False)
    save(fig, output, "fig_routing_specialization")
    return ["fig_routing_specialization"]


def robustness_figure(output: Path, data_dir: Path) -> list[str]:
    path = PROJECT / "reports/robustness/locked_robustness_v1/robustness_curves.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame.to_csv(data_dir / "fig_robustness_curves.csv", index=False)
    raw = (
        frame.loc[frame["system_view"].eq("raw_neural")].copy()
        if "system_view" in frame
        else frame
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = raw.loc[raw["task"].eq(task)]
        sns.lineplot(
            data=subset,
            x="severity",
            y="value",
            hue="variant",
            style="family",
            palette=PALETTE,
            markers=True,
            errorbar="sd",
            ax=axis,
        )
        axis.set_title(task.capitalize())
        axis.set_ylabel("Macro chrF++" if task == "normalization" else "Regional macro-F1")
        axis.set_xlabel("Perturbation severity")
    axes[1].get_legend().remove()
    axes[0].legend(title=None, fontsize=6, frameon=False)
    save(fig, output, "fig_robustness_curves")
    created = ["fig_robustness_curves"]
    if "system_view" in frame:
        system = frame.loc[frame["variant"].eq("M3")].copy()
        system.to_csv(data_dir / "fig_fusion_robustness.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
        view_palette = {
            "raw_neural": "#4C78A8",
            "source_blind_fused": "#54A24B",
            "source_blind_baseline": "#B279A2",
        }
        for axis, task in zip(axes, ("normalization", "identification")):
            subset = system.loc[system["task"].eq(task)]
            sns.lineplot(
                data=subset,
                x="severity",
                y="value",
                hue="system_view",
                style="family",
                palette=view_palette,
                markers=True,
                errorbar="sd",
                ax=axis,
            )
            axis.set_title(task.capitalize())
            axis.set_ylabel(
                "Macro chrF++"
                if task == "normalization"
                else "Regional macro-F1"
            )
            axis.set_xlabel("Perturbation severity")
        axes[1].get_legend().remove()
        axes[0].legend(title=None, fontsize=6, frameon=False)
        fig.suptitle("M3 raw, source-blind fusion, and baseline robustness")
        save(fig, output, "fig_fusion_robustness")
        created.append("fig_fusion_robustness")
    return created


def training_curves_figure(output: Path, data_dir: Path) -> list[str]:
    runs = {
        "F dense": "F_DENSE_300M",
        "M0 dense": "U_M0_DENSE_200M",
        "M1 Switch": "U_M1_SWITCH_200M",
        "M2/M3 common": "U_M2_STANDARD_MOE_200M",
    }
    frames = []
    validation_frames = []
    for label, run_id in runs.items():
        root = PROJECT / "runs" / run_id / "1701"
        report = root / "training_report.json"
        if not report.exists() or json.loads(report.read_text()).get("status") != "COMPLETE_FIXED_BUDGET":
            return []
        train = pd.read_json(root / "train_log.jsonl", lines=True)
        train["model"] = label
        train["phase"] = "foundation" if label == "F dense" else "continuation"
        train["tokens_millions"] = train["tokens_seen"] / 1e6
        train["smoothed_lm_loss"] = train["lm_loss"].rolling(5, min_periods=1).mean()
        frames.append(train)
        validation_path = root / "validation_log.jsonl"
        if validation_path.exists():
            validation = pd.read_json(validation_path, lines=True)
            validation["model"] = label
            validation["tokens_millions"] = validation["tokens_seen"] / 1e6
            validation_frames.append(validation)
    train = pd.concat(frames, ignore_index=True)
    validation = pd.concat(validation_frames, ignore_index=True)
    train.to_csv(data_dir / "fig_training_curves.csv", index=False)
    validation.to_csv(data_dir / "fig_validation_curves.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.2))
    foundation = train.loc[train["phase"].eq("foundation")]
    sns.lineplot(data=foundation, x="tokens_millions", y="smoothed_lm_loss", ax=axes[0])
    axes[0].set_title("Dense foundation")
    axes[0].set_xlabel("Stage tokens (millions)")
    axes[0].set_ylabel("Smoothed LM loss")
    continuation = train.loc[train["phase"].eq("continuation")]
    sns.lineplot(
        data=continuation,
        x="tokens_millions",
        y="smoothed_lm_loss",
        hue="model",
        ax=axes[1],
    )
    axes[1].set_title("Matched continuation")
    axes[1].set_xlabel("Stage tokens (millions)")
    axes[1].set_ylabel("Smoothed LM loss")
    axes[1].legend(title=None, frameon=False, fontsize=7)
    sns.lineplot(
        data=validation,
        x="tokens_millions",
        y="validation_bpc",
        hue="model",
        marker="o",
        ax=axes[2],
    )
    axes[2].set_title("Held-out general-domain BPC")
    axes[2].set_xlabel("Stage tokens (millions)")
    axes[2].set_ylabel("Bits per character")
    axes[2].legend(title=None, frameon=False, fontsize=7)
    save(fig, output, "fig_training_curves")
    return ["fig_training_curves"]


def upcycling_recovery_figure(output: Path, data_dir: Path) -> list[str]:
    runs = {
        "Banked upcycle": "P_M2_BANKED_20M",
        "Unbanked upcycle": "P_M2_UNBANKED_20M",
        "MoE from scratch": "P_M2_SCRATCH_20M",
        "Annealed cross-bank": "P_M2_ANNEALED_20M",
        "Permanent paired-bank": "P_M2_PAIRED_20M",
    }
    frames = []
    for label, run_id in runs.items():
        root = PROJECT / "runs" / run_id / "1701"
        report_path = root / "training_report.json"
        validation_path = root / "validation_log.jsonl"
        if not report_path.exists() or not validation_path.exists():
            return []
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "COMPLETE_FIXED_BUDGET":
            return []
        frame = pd.read_json(validation_path, lines=True)
        frame = frame.loc[frame["tokens_seen"].le(20_100_000)].copy()
        frame["condition"] = label
        frame["tokens_millions"] = frame["tokens_seen"] / 1e6
        frames.append(frame)
    recovery = pd.concat(frames, ignore_index=True)
    recovery.to_csv(data_dir / "fig_upcycling_recovery.csv", index=False)
    fig, axis = plt.subplots(figsize=(5.4, 3.2))
    sns.lineplot(
        data=recovery,
        x="tokens_millions",
        y="validation_bpc",
        hue="condition",
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Continuation tokens (millions)")
    axis.set_ylabel("Held-out BPC (lower is better)")
    axis.set_title("Development-only dense→MoE recovery pilot")
    axis.legend(title=None, frameon=False, fontsize=7)
    save(fig, output, "fig_upcycling_recovery")
    return ["fig_upcycling_recovery"]


def switch_router_recovery_figure(output: Path, data_dir: Path) -> list[str]:
    """Show the disclosed collapse and the validation-only router repair pilot."""

    runs = {
        "Zero-init top-1 (rejected)": PROJECT
        / "runs/aborted/U_M1_SWITCH_200M_zero_router_collapse_20260720/1701",
        "Straight-through, loss-free": PROJECT
        / "runs/P_M1_LOSS_FREE_ROUTER_10M/1701",
        "Straight-through + balance (selected)": PROJECT
        / "runs/P_M1_AUX_ROUTER_10M/1701",
    }
    validation_frames = []
    router_frames = []
    for condition, root in runs.items():
        validation_path = root / "validation_log.jsonl"
        train_path = root / "train_log.jsonl"
        if not validation_path.exists() or not train_path.exists():
            return []
        validation = pd.read_json(validation_path, lines=True)
        validation = validation.loc[validation["tokens_seen"].le(10_100_000)].copy()
        validation["condition"] = condition
        validation["tokens_millions"] = validation["tokens_seen"] / 1e6
        validation_frames.append(validation)
        router = pd.read_json(train_path, lines=True)
        router = router.loc[router["tokens_seen"].le(10_100_000)].copy()
        router["condition"] = condition
        router["tokens_millions"] = router["tokens_seen"] / 1e6
        router_frames.append(router)
    validation = pd.concat(validation_frames, ignore_index=True)
    router = pd.concat(router_frames, ignore_index=True)
    validation.to_csv(data_dir / "fig_switch_router_validation.csv", index=False)
    router.to_csv(data_dir / "fig_switch_router_load.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25))
    sns.lineplot(
        data=validation,
        x="tokens_millions",
        y="validation_bpc",
        hue="condition",
        marker="o",
        ax=axes[0],
    )
    axes[0].set_title("Held-out recovery")
    axes[0].set_xlabel("Continuation tokens (millions)")
    axes[0].set_ylabel("Validation BPC")
    axes[0].legend(title=None, frameon=False, fontsize=6)
    sns.lineplot(
        data=router,
        x="tokens_millions",
        y="router_load_cv",
        hue="condition",
        ax=axes[1],
    )
    axes[1].axhline(0.5, linestyle="--", linewidth=0.9, color="#555555")
    axes[1].set_title("Expert-load stability")
    axes[1].set_xlabel("Continuation tokens (millions)")
    axes[1].set_ylabel("Router load CV")
    if axes[1].get_legend() is not None:
        axes[1].get_legend().remove()
    fig.suptitle("Validation-only Switch-router failure analysis and repair")
    save(fig, output, "fig_switch_router_recovery")
    return ["fig_switch_router_recovery"]


def factorial_pilot_figure(output: Path, data_dir: Path) -> list[str]:
    factors = {
        "Dialect-evidence prior": "no_lexical_prior",
        "Dialect auxiliary": "no_dialect_head",
        "Source adversary": "no_source_adversary",
        "Task conditioning": "no_task_conditioning",
    }
    rows = []
    import itertools

    for present in itertools.product((False, True), repeat=len(factors)):
        ablations = [ablation for ablation, enabled in zip(factors.values(), present) if not enabled]
        suffix = "base" if not ablations else "__".join(sorted(ablations))
        path = (
            PROJECT
            / "runs/task/boichitro_factorial_pilot_v1"
            / f"M3__{suffix}/1701/stage_s/best_selection.json"
        )
        if not path.exists():
            return []
        metrics = json.loads(path.read_text(encoding="utf-8"))["validation"]
        row = {"suffix": suffix, "macro_chrfpp": float(metrics["macro_chrfpp"])}
        row.update({name: enabled for name, enabled in zip(factors, present)})
        rows.append(row)
    cells = pd.DataFrame(rows)
    effects = []
    for factor in factors:
        enabled = cells.loc[cells[factor], "macro_chrfpp"].mean()
        disabled = cells.loc[~cells[factor], "macro_chrfpp"].mean()
        effects.append(
            {
                "factor": factor,
                "enabled_mean": enabled,
                "disabled_mean": disabled,
                "main_effect_enabled_minus_disabled": enabled - disabled,
            }
        )
    effect_frame = pd.DataFrame(effects).sort_values("main_effect_enabled_minus_disabled")
    cells.to_csv(data_dir / "fig_factorial_pilot_cells.csv", index=False)
    effect_frame.to_csv(data_dir / "fig_factorial_pilot_effects.csv", index=False)
    fig, axis = plt.subplots(figsize=(5.4, 3.0))
    colors = ["#54A24B" if value >= 0 else "#E45756" for value in effect_frame["main_effect_enabled_minus_disabled"]]
    axis.barh(effect_frame["factor"], effect_frame["main_effect_enabled_minus_disabled"], color=colors)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Main effect on dev macro chrF++ (enabled − disabled)")
    axis.set_title("Registered 2⁴ factorial pilot (development only)")
    save(fig, output, "fig_factorial_pilot")
    return ["fig_factorial_pilot"]


def ablation_delta_figure(
    output: Path, data_dir: Path, protocol: str, seeds: list[int]
) -> list[str]:
    suffixes = [
        "no_lexical_prior",
        "no_dialect_head",
        "no_source_adversary",
        "no_task_conditioning",
        "randomized_lexical_prior",
        "no_groupdro",
        "no_synthetic",
        "no_general_replay",
        "adamw_only",
        "no_mtp",
    ]
    labels = {
        "no_lexical_prior": "− dialect-evidence prior",
        "no_dialect_head": "− dialect auxiliary",
        "no_source_adversary": "− source adversary",
        "no_task_conditioning": "− task conditioning",
        "randomized_lexical_prior": "randomized evidence mapping",
        "no_groupdro": "− GroupDRO",
        "no_synthetic": "− synthetic augmentation",
        "no_general_replay": "− general replay",
        "adamw_only": "AdamW only",
        "no_mtp": "− MTP",
    }
    rows = []
    for seed in seeds:
        base_path = PROJECT / "predictions" / protocol / "M3__base" / str(seed) / "evaluation_manifest.json"
        if not base_path.exists():
            return []
        base = json.loads(base_path.read_text(encoding="utf-8"))
        base_values = {
            "normalization": float(base["normalization"]["source_ood"]["macro_chrfpp"]),
            "identification": float(base["identification"]["source_ood"]["regional_macro_f1"]),
        }
        for suffix in suffixes:
            path = PROJECT / "predictions" / protocol / f"M3__{suffix}" / str(seed) / "evaluation_manifest.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = {
                "normalization": float(payload["normalization"]["source_ood"]["macro_chrfpp"]),
                "identification": float(payload["identification"]["source_ood"]["regional_macro_f1"]),
            }
            for task, value in values.items():
                rows.append(
                    {
                        "ablation": suffix,
                        "label": labels[suffix],
                        "seed": seed,
                        "task": task,
                        "value": value,
                        "delta_from_m3": value - base_values[task],
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame["ablation"]) != set(suffixes):
        return []
    frame.to_csv(data_dir / "fig_ablation_deltas.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True)
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = frame.loc[frame["task"].eq(task)]
        sns.pointplot(
            data=subset,
            x="delta_from_m3",
            y="label",
            errorbar="sd",
            join=False,
            color="#4C78A8",
            ax=axis,
        )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(task.capitalize() + " source-OOD")
        axis.set_xlabel("Ablation − full M3")
        axis.set_ylabel("")
    save(fig, output, "fig_ablation_deltas")
    return ["fig_ablation_deltas"]


def calibration_confusion_figure(
    output: Path, data_dir: Path, protocol: str, suffix: str, seeds: list[int]
) -> list[str]:
    reliability_rows = []
    matrices = []
    for seed in seeds:
        root = PROJECT / "predictions" / protocol / f"M3__{suffix}" / str(seed)
        prediction_path = root / "identification_source_ood.parquet"
        matrix_path = root / "identification_source_ood_confusion.npy"
        if not prediction_path.exists() or not matrix_path.exists():
            return []
        frame = pd.read_parquet(prediction_path)
        probabilities = np.asarray(frame["probabilities"].tolist(), dtype=np.float64)
        confidence = probabilities.max(axis=1)
        correct = frame["prediction_id"].to_numpy() == frame["label_id"].to_numpy()
        bins = np.minimum((confidence * 10).astype(int), 9)
        for bin_index in range(10):
            mask = bins == bin_index
            if mask.any():
                reliability_rows.append(
                    {
                        "seed": seed,
                        "bin": bin_index,
                        "count": int(mask.sum()),
                        "mean_confidence": float(confidence[mask].mean()),
                        "accuracy": float(correct[mask].mean()),
                    }
                )
        matrix = np.load(matrix_path).astype(np.float64)
        matrices.append(matrix / np.maximum(1.0, matrix.sum(axis=1, keepdims=True)))
    reliability = pd.DataFrame(reliability_rows)
    reliability.to_csv(data_dir / "fig_calibration_reliability.csv", index=False)
    confusion = np.mean(matrices, axis=0)
    pd.DataFrame(confusion, index=DIALECT_LABELS, columns=DIALECT_LABELS).to_csv(
        data_dir / "fig_source_ood_confusion.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.8))
    sns.lineplot(
        data=reliability,
        x="mean_confidence",
        y="accuracy",
        marker="o",
        errorbar="sd",
        ax=axes[0],
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("M3 source-OOD calibration")
    axes[0].set_xlabel("Mean calibrated confidence")
    axes[0].set_ylabel("Empirical accuracy")
    sns.heatmap(
        confusion,
        cmap="mako",
        vmin=0,
        vmax=1,
        xticklabels=DIALECT_LABELS,
        yticklabels=DIALECT_LABELS,
        cbar_kws={"label": "Row-normalized rate"},
        ax=axes[1],
    )
    axes[1].set_title("M3 source-OOD confusion")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)
    save(fig, output, "fig_calibration_confusion")
    return ["fig_calibration_confusion"]


def external_baseline_figure(
    output: Path, data_dir: Path, main_protocol: str, seeds: list[int]
) -> list[str]:
    rows = []
    for seed in seeds:
        main_path = PROJECT / "predictions" / main_protocol / "M3__base" / str(seed) / "evaluation_manifest.json"
        if not main_path.exists():
            return []
        main = json.loads(main_path.read_text(encoding="utf-8"))
        rows.extend(
            [
                {"task": "normalization", "model": "M3", "seed": seed, "value": main["normalization"]["source_ood"]["macro_chrfpp"]},
                {"task": "identification", "model": "M3", "seed": seed, "value": main["identification"]["source_ood"]["regional_macro_f1"]},
            ]
        )
        bidirectional_path = (
            PROJECT
            / "predictions"
            / main_protocol
            / "M3B_BIDIR__base"
            / str(seed)
            / "evaluation_manifest.json"
        )
        if not bidirectional_path.exists():
            return []
        bidirectional = json.loads(bidirectional_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "task": "identification",
                "model": "M3-Bi",
                "seed": seed,
                "value": bidirectional["identification"]["source_ood"]["regional_macro_f1"],
            }
        )
        for task, models, metric in (
            ("normalization", ("BANGLAT5_SMALL", "MT5_SMALL"), "macro_chrfpp"),
            ("identification", ("BANGLABERT_MIT", "XLMR_BASE"), "regional_macro_f1"),
        ):
            for model in models:
                path = PROJECT / "predictions/locked_external_test_v1" / task / model / str(seed) / "evaluation_manifest.json"
                if not path.exists():
                    return []
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        "task": task,
                        "model": model,
                        "seed": seed,
                        "value": payload["results"]["source_ood"][metric],
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(data_dir / "fig_external_baselines.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = frame.loc[frame["task"].eq(task)]
        sns.barplot(data=subset, x="model", y="value", errorbar="sd", ax=axis)
        axis.set_title(task.capitalize() + " source-OOD")
        axis.set_ylabel("Macro chrF++" if task == "normalization" else "Regional macro-F1")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=18)
    save(fig, output, "fig_external_baselines")
    return ["fig_external_baselines"]


def task_efficiency_figure(
    output: Path, data_dir: Path, protocol: str, seeds: list[int]
) -> list[str]:
    benchmark_path = PROJECT / "reports/model/task_inference_benchmark.csv"
    if not benchmark_path.exists():
        return []
    benchmark = pd.read_csv(benchmark_path)
    benchmark = benchmark.loc[
        benchmark.groupby(["variant", "task"])["batch_size"].transform("max")
        .eq(benchmark["batch_size"])
    ].copy()
    norm, identification = collect_main(protocol, "base", seeds)
    if norm.empty or identification.empty:
        return []
    metrics = pd.concat(
        (
            norm.loc[norm["track"].eq("source_ood"), ["variant", "seed", "macro_chrfpp"]]
            .groupby("variant", as_index=False)["macro_chrfpp"]
            .mean()
            .rename(columns={"macro_chrfpp": "quality"})
            .assign(task="normalization"),
            identification.loc[
                identification["track"].eq("source_ood"),
                ["variant", "seed", "regional_macro_f1"],
            ]
            .groupby("variant", as_index=False)["regional_macro_f1"]
            .mean()
            .rename(columns={"regional_macro_f1": "quality"})
            .assign(task="identification"),
        ),
        ignore_index=True,
    )
    frame = benchmark.merge(metrics, on=["variant", "task"], validate="one_to_one")
    frame.to_csv(data_dir / "fig_task_efficiency.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = frame.loc[frame["task"].eq(task)]
        for row in subset.itertuples(index=False):
            axis.scatter(
                row.examples_per_second,
                row.quality,
                s=65,
                color=PALETTE[row.variant],
                edgecolor="white",
                linewidth=0.7,
            )
            axis.annotate(
                row.variant,
                (row.examples_per_second, row.quality),
                xytext=(4, 3),
                textcoords="offset points",
            )
        axis.set_title(task.capitalize() + " source-OOD")
        axis.set_xlabel("End-to-end examples/s")
        axis.set_ylabel("Macro chrF++" if task == "normalization" else "Regional macro-F1")
    save(fig, output, "fig_task_efficiency")
    return ["fig_task_efficiency"]


def statistical_effects_figure(
    output: Path, data_dir: Path, protocol: str
) -> list[str]:
    sources = (
        (
            "main",
            PROJECT / "reports/statistics" / protocol / "confirmatory_statistics.csv",
        ),
        (
            "bidirectional",
            PROJECT
            / "reports/statistics"
            / protocol
            / "bidirectional_specialization/confirmatory_statistics.csv",
        ),
        (
            "system_fusion",
            PROJECT
            / "reports/statistics"
            / protocol
            / "source_blind_system_fusion/confirmatory_statistics.csv",
        ),
    )
    frames = []
    for family, path in sources:
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        frame["analysis_family"] = family
        frames.append(frame)
    effects = pd.concat(frames, ignore_index=True)
    labels = {
        "M3_vs_M2_normalization_source_ood": "M3 − M2",
        "M3_vs_M0_normalization_source_ood": "M3 − M0",
        "M3_vs_M2_identification_source_ood": "M3 − M2",
        "M3B_vs_M3_identification_source_ood": "M3B − M3",
        "M3B_fused_vs_M3_fused_identification_source_ood": "M3B fused − M3 fused",
        "M3B_fused_vs_raw_identification_source_ood": "M3B fused − raw",
        "M3_fused_vs_raw_normalization_source_ood": "M3 fused − raw",
        "M3_fused_vs_inferred_rewrite_source_ood": "M3 fused − rewrite",
        "M3_fused_vs_raw_identification_source_ood": "M3 fused − raw",
        "M3_fused_vs_char_svm_identification_source_ood": "M3 fused − SVM",
    }
    effects["label"] = effects["id"].map(labels).fillna(effects["id"])
    effects["holm_significant_005"] = effects["p_holm_two_sided"].lt(0.05)
    effects.to_csv(data_dir / "fig_confirmatory_effects.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.2))
    for axis, task in zip(axes, ("normalization", "identification")):
        subset = effects.loc[effects["task"].eq(task)].reset_index(drop=True)
        positions = np.arange(len(subset))
        centers = subset["bootstrap_mean_delta"].to_numpy()
        lower = centers - subset["confidence_lower"].to_numpy()
        upper = subset["confidence_upper"].to_numpy() - centers
        colors = [
            "#54A24B" if significant else "#4C78A8"
            for significant in subset["holm_significant_005"]
        ]
        for position, center, low, high, color in zip(
            positions, centers, lower, upper, colors
        ):
            axis.errorbar(
                center,
                position,
                xerr=np.asarray([[low], [high]]),
                fmt="o",
                color=color,
                capsize=3,
            )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_yticks(positions, subset["label"])
        axis.set_xlabel("Paired effect (treatment − control)")
        axis.set_title(task.capitalize() + " source-OOD")
        axis.invert_yaxis()
    fig.suptitle("Confirmatory effects with 95% hierarchical bootstrap intervals")
    save(fig, output, "fig_confirmatory_effects")
    return ["fig_confirmatory_effects"]


def human_evaluation_figure(output: Path, data_dir: Path) -> list[str]:
    root = PROJECT / "human_evaluation/blind_native_normalization_v1"
    path = root / "human_evaluation_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    required_dimensions = (
        "meaning_preservation",
        "standard_bangla_fluency",
        "overall_quality",
        "hallucination_or_unsupported_content",
    )
    if not set(required_dimensions).issubset(set(frame["dimension"])):
        return []
    frame.to_csv(data_dir / "fig_human_evaluation.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.6))
    titles = {
        "meaning_preservation": "Meaning preservation",
        "standard_bangla_fluency": "Standard Bangla fluency",
        "overall_quality": "Overall quality",
        "hallucination_or_unsupported_content": "Hallucination rate",
    }
    for axis, dimension in zip(axes.flat, required_dimensions):
        subset = frame.loc[frame["dimension"].eq(dimension)].reset_index(drop=True)
        positions = np.arange(len(subset))
        centers = subset["mean"].to_numpy()
        lower = np.maximum(0.0, centers - subset["confidence_lower"].to_numpy())
        upper = np.maximum(0.0, subset["confidence_upper"].to_numpy() - centers)
        axis.errorbar(
            centers,
            positions,
            xerr=np.vstack((lower, upper)),
            fmt="o",
            color="#4C78A8",
            capsize=3,
        )
        axis.set_yticks(positions, subset["system"])
        axis.invert_yaxis()
        axis.set_title(titles[dimension])
        axis.set_xlabel("Mean rating (95% CI)" if dimension != required_dimensions[-1] else "Flag rate (95% CI; lower is better)")
    fig.suptitle("Blinded native-speaker normalization evaluation")
    save(fig, output, "fig_human_evaluation")
    return ["fig_human_evaluation"]


def main() -> None:
    args = parse_args()
    style()
    output = PROJECT / "figures/paper"
    data_dir = PROJECT / "figures/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    created = preliminary_figures(output, data_dir)
    created += continuation_lr_stability_figure(output, data_dir)
    created += stage_s_retention_figure(output, data_dir)
    created += architecture_figure(output, data_dir)
    created += protocol_flow_figure(output, data_dir)
    created += main_result_figures(output, data_dir, args.protocol, args.suffix, args.seeds)
    created += source_blind_system_fusion_figure(
        output, data_dir, args.protocol, args.suffix, args.seeds
    )
    created += routing_figure(output, data_dir, args.protocol, args.suffix, args.seeds)
    created += routing_specialization_figure(output, data_dir, args.protocol)
    created += robustness_figure(output, data_dir)
    created += training_curves_figure(output, data_dir)
    created += upcycling_recovery_figure(output, data_dir)
    created += switch_router_recovery_figure(output, data_dir)
    created += factorial_pilot_figure(output, data_dir)
    created += ablation_delta_figure(output, data_dir, args.protocol, args.seeds)
    created += calibration_confusion_figure(
        output, data_dir, args.protocol, args.suffix, args.seeds
    )
    created += external_baseline_figure(output, data_dir, args.protocol, args.seeds)
    created += task_efficiency_figure(output, data_dir, args.protocol, args.seeds)
    created += statistical_effects_figure(output, data_dir, args.protocol)
    created += human_evaluation_figure(output, data_dir)
    manifest = {"created": created, "protocol": args.protocol, "seeds": args.seeds}
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
