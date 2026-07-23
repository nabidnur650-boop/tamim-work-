#!/usr/bin/env python3
"""Build a journal-neutral manuscript from frozen reports and locked outputs."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "manuscript/BOICHITRO_MOE_Q1_MANUSCRIPT.md"
SEEDS = (1701, 2903, 4307)
VARIANTS = ("M0", "M1", "M2", "M3")


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def fmt_mean(values: list[float], digits: int = 2) -> str:
    if not values:
        return "pending"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f}"


def locked_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = PROJECT / "predictions/locked_test_v1" / f"{variant}__base" / str(seed) / "evaluation_manifest.json"
            payload = load_json(path)
            if payload is None:
                continue
            for task, manifest_key, view, primary, secondary in (
                (
                    "normalization",
                    "normalization",
                    "raw neural",
                    "macro_chrfpp",
                    "worst_dialect_chrfpp",
                ),
                (
                    "normalization",
                    "normalization_fused",
                    "source-blind fused",
                    "macro_chrfpp",
                    "worst_dialect_chrfpp",
                ),
                (
                    "identification",
                    "identification",
                    "raw neural",
                    "regional_macro_f1",
                    "worst_present_dialect_f1",
                ),
                (
                    "identification",
                    "identification_fused",
                    "source-blind fused",
                    "regional_macro_f1",
                    "worst_present_dialect_f1",
                ),
            ):
                for track, metrics in payload.get(manifest_key, {}).items():
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "task": task,
                            "view": view,
                            "track": track,
                            "primary": metrics.get(primary),
                            "secondary": metrics.get(secondary),
                            "ece_15": metrics.get("ece_15"),
                        }
                    )
    return pd.DataFrame(rows)


def locked_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return (
            "> **Locked-result firewall:** no locked neural evaluation manifests were available when this draft was built. "
            "Numerical main-result claims are intentionally withheld until the registered pipeline completes."
        )
    lines = [
        "| Task | System view | Track | Model | Seeds | Primary metric, mean ± SD | Worst-group metric, mean ± SD | ECE-15, mean ± SD |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for (task, view, track, variant), subset in frame.groupby(
        ["task", "view", "track", "variant"], sort=False
    ):
        primary = [float(value) for value in subset.primary.dropna()]
        secondary = [float(value) for value in subset.secondary.dropna()]
        ece = [float(value) for value in subset.ece_15.dropna()]
        lines.append(
            f"| {task} | {view} | {track} | {variant} | {subset.seed.nunique()} | {fmt_mean(primary, 3 if task == 'identification' else 2)} | {fmt_mean(secondary, 3 if task == 'identification' else 2)} | {fmt_mean(ece, 3) if ece else '—'} |"
        )
    return "\n".join(lines)


def development_summary() -> tuple[str, int]:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            root = PROJECT / "runs/task/boichitro_q1_v1" / f"{variant}__base" / str(seed)
            norm = load_json(root / "stage_s/best_selection.json")
            ident = load_json(root / "stage_id/best_selection.json")
            if norm and ident:
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "macro_chrfpp": float(norm["validation"]["macro_chrfpp"]),
                        "replay_degradation": 100 * float(norm["validation"]["replay_relative_degradation"]),
                        "regional_macro_f1": float(ident["validation"]["regional_macro_f1"]),
                        "ece_15": float(ident["validation"]["ece_15"]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return "No complete main development runs were detected.", 0
    lines = [
        "| Model | Complete seeds | Validation macro chrF++ | Replay degradation (%) | Validation regional macro-F1 | ECE-15 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, subset in frame.groupby("variant", sort=False):
        lines.append(
            f"| {variant} | {subset.seed.nunique()} | {fmt_mean(subset.macro_chrfpp.tolist())} | {fmt_mean(subset.replay_degradation.tolist())} | {fmt_mean(subset.regional_macro_f1.tolist(), 3)} | {fmt_mean(subset.ece_15.tolist(), 3)} |"
        )
    fusion = load_json(PROJECT / "reports/model/development_fusion_uncertainty.json")
    norm_selection = load_json(
        PROJECT / "reports/model/normalization_fusion_selection_v2.json"
    )
    id_selection = load_json(PROJECT / "reports/model/id_fusion_selection.json")
    if fusion and norm_selection and id_selection:
        norm = fusion["normalization"]
        ident = fusion["identification"]
        selected_norm = norm_selection["selected"]
        selected_id = id_selection["selected_summary"]
        lines.extend(
            [
                "",
                "Fixed source-blind system views (development only):",
                "",
                "| System view | Development score | Paired gain | 95% hierarchical CI |",
                "|---|---:|---:|---:|",
                f"| Normalization selector V2 | {selected_norm['mean_macro_chrfpp']:.3f} macro chrF++ | {norm['mean_delta']:+.3f} | [{norm['confidence_lower_95']:.3f}, {norm['confidence_upper_95']:.3f}] |",
                f"| Neural/SVM identification blend | {selected_id['mean_regional_macro_f1']:.4f} regional macro-F1 | {ident['mean_delta']:+.4f} | [{ident['confidence_lower_95']:.4f}, {ident['confidence_upper_95']:.4f}] |",
                "",
                "References, gold dialects, source IDs, and evaluation-track labels are forbidden fusion inputs. Whole normalization semantic groups are held out together. Because the fusion settings were selected on these development rows, the intervals are selection-conditioned exploratory diagnostics, not confirmatory inference. Raw neural outputs remain primary for architectural inference.",
            ]
        )
    transfer = load_json(
        PROJECT / "reports/model/development_fusion_architecture_transfer.json"
    )
    if transfer and transfer.get("runs"):
        lines.extend(
            [
                "",
                "Fixed no-retuning transfer to later architectures (shared validation rows; not independent confirmation):",
                "",
                "| Model | Seed | Raw norm chrF++ | Fused norm chrF++ | Fused worst dialect | Raw ID F1 | Fused ID F1 |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in transfer["runs"]:
            lines.append(
                f"| {run['variant']} | {int(run['seed'])} | "
                f"{float(run['normalization_raw_macro_chrfpp']):.3f} | "
                f"{float(run['normalization_fused_macro_chrfpp']):.3f} | "
                f"{float(run['normalization_fused_worst_dialect_chrfpp']):.3f} | "
                f"{float(run['identification_raw_regional_macro_f1']):.4f} | "
                f"{float(run['identification_fused_regional_macro_f1']):.4f} |"
            )
    return "\n".join(lines), len(frame)


def benchmark_table() -> str:
    frame = pd.read_csv(PROJECT / "reports/model/gb10_model_benchmark.csv")
    lines = [
        "| System | Total parameters | Active parameters/token | Active fraction | Tokens/s | Peak memory (GiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.model_id} | {row.total_parameters/1e6:.1f}M | {row.active_parameters_per_token/1e6:.1f}M | {100*row.active_fraction:.1f}% | {row.tokens_per_second:,.0f} | {row.peak_memory_gib:.2f} |"
        )
    return "\n".join(lines)


def main() -> None:
    dataset = load_json(PROJECT / "reports/final_dataset_report.json") or {}
    counts = dataset.get("counts", {})
    tokenizer = load_json(PROJECT / "reports/tokenizer/TOKENIZER_FREEZE_REPORT.json") or {}
    pretraining = load_json(PROJECT / "reports/model/pretraining_corpus_report.json") or {}
    stage_s = load_json(PROJECT / "reports/model/stage_s_retention_pilot_selection.json") or {}
    figures = load_json(PROJECT / "figures/q1/figure_manifest.json") or {}
    native = load_json(PROJECT / "reports/native_review_report.json") or {}
    pipeline = load_json(PROJECT / "reports/pipeline/full_pipeline_state.json") or {}
    locked = locked_rows()
    development, development_runs = development_summary()
    raw_locked = locked.loc[locked.get("view", pd.Series(dtype=str)).eq("raw neural")]
    locked_complete = (
        not raw_locked.empty
        and raw_locked.loc[raw_locked.task.eq("normalization")].groupby("variant").seed.nunique().reindex(VARIANTS, fill_value=0).eq(3).all()
        and raw_locked.loc[raw_locked.task.eq("identification")].groupby("variant").seed.nunique().reindex(VARIANTS, fill_value=0).eq(3).all()
    )
    selected_stage_s = stage_s.get("selected_candidate_id", "pending")
    selected_stage_s_metrics = stage_s.get("selected_validation", {})
    figure_count = int(figures.get("figure_pairs", 0))
    native_fraction = float(native.get("completion_fraction", 0.0))
    pipeline_status = str(pipeline.get("status", "MISSING"))

    if locked_complete:
        source_ood = raw_locked.loc[raw_locked.track.eq("source_ood")]
        m3_norm = source_ood.loc[(source_ood.task == "normalization") & (source_ood.variant == "M3"), "primary"].dropna().astype(float).tolist()
        m2_norm = source_ood.loc[(source_ood.task == "normalization") & (source_ood.variant == "M2"), "primary"].dropna().astype(float).tolist()
        abstract_result = (
            f"On the locked source-OOD normalization track, M3 obtained {fmt_mean(m3_norm)} macro chrF++, "
            f"compared with {fmt_mean(m2_norm)} for the standard shared-expert M2 control. "
            "Confirmatory uncertainty estimates, rather than the mean alone, determine whether the architectural claim is supported."
        )
    else:
        abstract_result = (
            "Locked neural results are not stated in this draft because the registered computational pipeline has not yet produced all twelve main evaluation manifests."
        )
    publication_gate_sentence = (
        "The remaining non-computational publication gate is completion of the preregistered native-speaker review and blinded native evaluation."
        if locked_complete
        else "The registered computational studies, protocol freeze, locked evaluation, and preregistered native-speaker review and blinded native evaluation remain incomplete."
    )

    status_lines = [
        f"- Registered pipeline state: **{pipeline_status}**",
        f"- Complete main development runs detected: **{development_runs}/12**",
        f"- Complete locked M0–M3 × three-seed evidence: **{'yes' if locked_complete else 'no'}**",
        f"- Native dataset-review completion: **{100*native_fraction:.1f}%**",
        f"- Paired PNG/PDF figures: **{figure_count}**",
        "- Author identities, affiliations, contributions, funding, conflicts, and target-journal formatting: **required before submission**",
    ]

    text = f"""# Boichitro-MoE: Source-Invariant Sparse Small Language Modeling for Bangla Dialect Normalization

**Manuscript type:** Original research article
**Authors:** AUTHOR DETAILS REQUIRED BEFORE SUBMISSION
**Affiliations:** AFFILIATION DETAILS REQUIRED BEFORE SUBMISSION
**Corresponding author:** REQUIRED BEFORE SUBMISSION
**Draft generated:** {dt.datetime.now(dt.timezone.utc).isoformat()}

## Submission status (not part of the blinded manuscript)

{chr(10).join(status_lines)}

This is a journal-neutral scientific draft. “Q1” is not an acceptance claim: journal quartiles change over time and editorial decisions remain external. Numerical confirmatory claims are included only when the frozen locked-evaluation manifests exist.

## Abstract

Bangla dialect technology is frequently evaluated on merged corpora in which dialect identity and dataset source are entangled, creating a risk that systems learn collection artifacts instead of transferable dialect structure. We present Boichitro-MoE, a compact task-aware mixture-of-experts decoder designed for source-blind dialect-to-Standard-Bangla normalization and causal dialect identification. The study begins with a provenance-first reconstruction of local and public resources, connected-component leakage controls, protected source-held-out evaluation, and an explicit quarantine of legacy derived data. The frozen benchmark contains {counts.get('normalization_authentic', 'pending'):,} authentic normalization pairs, {counts.get('normalization_synthetic', 'pending'):,} traceable train-only perturbations, {counts.get('identification_all', 'pending'):,} conflict-cleaned identification rows, and {counts.get('romanized_ood', 'pending'):,} real romanized challenge items. A {pretraining.get('train_tokens', 0)/1e6:.1f}M-token dense Bangla foundation is continued into compute-matched dense, Switch top-1, and shared top-2 MoE controls. Boichitro adds causal dialect-evidence routing, task-conditioned late routing, source-adversarial supervision, and GroupDRO while retaining approximately matched active parameters. {abstract_result} The complete artifact contract saves per-example predictions, calibration, routing traces, hierarchical paired uncertainty estimates, and figure source data. {publication_gate_sentence}

**Keywords:** Bangla dialects; dialect normalization; mixture of experts; small language models; source shift; domain invariance; tokenizer fairness; reproducible NLP

## Research highlights

- A provenance-first Bangla dialect benchmark separates IID, source-OOD, external-transcript, and romanized challenge tracks.
- A custom {tokenizer.get('selected_id', 'pending')} tokenizer is frozen through intrinsic and three-seed proxy evaluation without test-set selection.
- Dense, Switch, standard-MoE, and task-aware MoE systems are compared under fixed token budgets and measured active-compute diagnostics.
- Replay-constrained normalization prevents task gains from being reported when general-language retention exceeds the preregistered guard.
- Every empirical figure has paired high-resolution PNG/vector PDF output and machine-readable source data.

## 1. Introduction

Bangla is a pluricentric, regionally diverse language whose written dialect resources differ sharply in provenance, annotation practice, orthographic convention, and source domain. These differences create both an opportunity and a methodological hazard. A model may appear to recognize a dialect while exploiting a dataset signature, or it may normalize familiar source templates without learning transformations that transfer to an independent source. This problem is especially acute when published datasets are merged without connected-component deduplication, ancestry tracking, or a protected source-held-out test.

Dialect normalization is a stronger test than classification alone because the system must preserve meaning while producing fluent Standard Bangla under an unknown-dialect input contract. At the same time, a compact deployable system must allocate limited capacity across shared lexical, syntactic, and regional phenomena. Sparse mixture-of-experts models provide conditional capacity, but ordinary routers are free to specialize by source artifacts. Our central question is therefore not simply whether an MoE is larger or more accurate. We ask whether a task-aware sparse small language model can encode dialect-relevant variation while suppressing source shortcuts at approximately matched token-active compute.

The study is designed around falsifiability. The primary architectural comparison is Boichitro-MoE (M3) versus a standard shared-expert top-2 MoE (M2). The primary endpoint is macro chrF++ on the locked source-independent normalization track. The claim is supported only if the registered hierarchical paired confidence interval for M3 − M2 lies entirely above zero; a positive point estimate is insufficient. IID performance, worst-dialect performance, replay retention, identification, calibration, robustness, routing specialization, efficiency, and blinded human ratings are supporting evidence rather than substitutes for the primary endpoint.

### 1.1 Contributions

1. We reconstruct a traceable Bangla dialect benchmark with immutable row identifiers, licenses, source versions, semantic groups, explicit exclusion reasons, and train/evaluation firewalls.
2. We introduce a staged sparse decoder whose router receives causal dialect evidence early and task-conditioned bias late, while a gradient-reversal source head penalizes source-predictive representations.
3. We provide compute-matched dense, Switch, and standard-MoE controls, validation-only architecture/optimizer pilots, three-seed main experiments, registered factorial and confirmatory ablations, and pinned external baselines.
4. We use a one-way protocol freeze before neural test access and retain per-example predictions for paired hierarchical inference.
5. We release an auditable publication bundle with {figure_count} paired figure exports at the time this draft was generated.

## 2. Related work

### 2.1 Bangla dialect resources and evaluation

Vashantor provides aligned regional and Standard-Bangla material; BanglaDial aggregates regional text for classification; ChatgaiyyaAlap contributes Chittagonian conversational pairs; BhasaBodh and related BanglaNLP resources add romanized or benchmark evidence. These resources are valuable but cannot be treated as independent merely because their filenames differ. The present work emphasizes ancestry recovery, exact/compact/near-duplicate controls, and source-held-out evaluation. BanglaBERT and multilingual encoders serve as external identification references, while sequence-to-sequence baselines are pinned before locked evaluation.

### 2.2 Sparse language models

Shared-expert and routed-expert designs increase total capacity while limiting active computation. DeepSeekMoE motivates shared experts and specialization; auxiliary-loss-free balancing avoids a training objective that can distort routing; upcycling transfers dense checkpoints into sparse models. OLMoE and related open studies highlight the need to report active parameters, total parameters, load balance, routing entropy, throughput, and stability rather than parameter count alone. Boichitro differs by making source invariance and task-conditioned routing explicit experimental targets in a low-resource dialect setting.

### 2.3 Tokenization and compact-model optimization

Tokenizer choice changes sequence length, dialect parity, memory, and the interpretation of token-normalized perplexity. We therefore compare WordPiece, Unigram, and byte-BPE candidates using tokens per character/byte, fertility, unknown rate, round-trip behavior, dialect dispersion, and a fixed-budget proxy language model. The final model uses Muon for eligible hidden matrices and AdamW for embeddings, norms, routers, and task heads, with an AdamW-only confirmatory control selected by a development-only LR pilot.

## 3. Data and benchmark construction

### 3.1 Frozen task inventory

The normalization task maps regional input to Standard Bangla under a source-blind `<dial_unknown>` prompt. Authentic and traceable perturbation rows total {counts.get('normalization_all', 'pending'):,}; only {counts.get('normalization_synthetic', 'pending'):,} perturbations are synthetic, and all are train-only. The identification task contains {counts.get('identification_all', 'pending'):,} rows across twelve regional labels plus Standard Bangla. A separate {counts.get('romanized_ood', 'pending'):,}-row track evaluates real romanized inputs.

### 3.2 Provenance and exclusion controls

Each admitted row records provider, DOI/version, license, source row, dialect taxonomy, normalized text, semantic group, split origin, evaluation track, quality tier, and synthetic ancestry where relevant. The complete legacy derived archive ({counts.get('legacy_derived_archive_quarantined_rows', 'pending'):,} rows) is quarantined from the main training pool. Exact and compact duplicates, cross-label text conflicts, protected evaluation relatives, and quality failures are recorded rather than silently dropped. SimHash and character n-gram controls supplement exact matching.

### 3.3 Evaluation tracks

The protocol separates local group-IID tests from source-held-out normalization, source-OOD identification, external transcripts, and romanized challenges. RAJ normalization is a zero-shot challenge and is not silently averaged into the trained-dialect endpoint. Test examples are unavailable to neural model selection, calibration fitting, schedule choice, tokenizer choice, and ablation pruning.

### 3.4 Human data review

A stratified 230-row review packet tests dialect authenticity, source/target validity, fluency, and label quality. At draft generation, {int(native.get('completed_rows', 0))}/230 rows were completed. Until this reaches 230/230, the corpus may support internal experiments but must not be described as fully linguistically validated or publicly redistribution-ready.

## 4. Tokenizer and general-language foundation

The selected tokenizer is `{tokenizer.get('selected_id', 'pending')}` with an actual vocabulary of 32,000. Selection combined intrinsic efficiency and dialect-parity gates with three proxy seeds trained for two million tokens each. Test data were not used for selection. Bits per character are reported for cross-tokenizer comparisons because token-level perplexity is not comparable across different vocabularies.

The general-language source is the revision-pinned Bengali subset of FineWeb-2. After Unicode, Bengali-ratio, length, and direct/compact benchmark decontamination, {pretraining.get('accepted_documents', 0):,} documents were accepted from {pretraining.get('raw_documents_seen', 0):,} examined. The fixed training budget contains {pretraining.get('train_tokens', 0):,} tokens, with a disjoint {pretraining.get('validation_tokens', 0):,}-token validation set. Packed block order and all source hashes are immutable.

## 5. Model

### 5.1 Shared decoder backbone

All main systems use a 16-layer causal decoder with width 512, grouped-query attention (8 query heads, 2 key/value heads), RoPE, QK normalization, RMSNorm, a frozen custom tokenizer, and multi-token prediction. M0 is dense. M1 uses Switch top-1 routing. M2 uses one shared expert plus top-2 of eight routed experts. M3 begins from the same M2 continuation checkpoint and adds the proposed task/dialect routing signals.

### 5.2 Boichitro routing

The first four blocks remain dense. Sparse layers 5–8 receive a causal dialect-evidence curriculum derived only from the input prefix visible at that position. Middle sparse layers use learned loss-free load bias. Layers 13–16 add a task-conditioned router bias. Every sparse layer retains one shared expert and activates two routed experts. A causal dialect head supplies auxiliary regional supervision, while a gradient-reversal source head discourages representations that identify the dataset source. GroupDRO reweights observed dialect/source/authenticity groups within the registered bounds.

### 5.3 Active-compute controls

{benchmark_table()}

The benchmark uses batch size 4 and sequence length 512 on the NVIDIA GB10. Total capacity differs, but M0–M3 activate approximately 83.8M parameters per token. We report throughput and memory because nominal active parameters do not capture routing and kernel overhead.

## 6. Training and specialization

The dense foundation is trained for 300M tokens. M0, M1, and M2 then receive matched 200M-token continuations from the same foundation and deterministic block order. High-LR mature-checkpoint restart behavior is retained as a negative result; a validation-only pilot selected a lower continuation LR before full runs.

Task adaptation uses three seeds (1701, 2903, 4307). Stage A consumes 12M fixed tokens from general replay, dialect CLM, normalization, and romanized material. Stage S consumes 6M fixed tokens and selects normalization checkpoints only if replay degradation remains at or below 5%. The selected schedule is `{selected_stage_s}`; its pilot checkpoint reached {selected_stage_s_metrics.get('macro_chrfpp', float('nan')):.2f} validation macro chrF++ with {100*selected_stage_s_metrics.get('replay_relative_degradation', float('nan')):.2f}% replay-NLL degradation. The causal ID branch trains for at most 20 epochs with early stopping and temperature calibration. A separate bidirectional MNTP/contrastive branch is used only for identification.

## 7. Experimental design

### 7.1 Baselines

Normalization controls include identity copying and a training-only word-rewrite model. The fair rewrite infers a supported normalization dialect from source text; the legacy gold-dialect rewrite is retained only as an explicitly labeled oracle upper bound. Identification controls include character TF–IDF SVM/SGD systems. Main neural controls are M0 dense, M1 Switch, and M2 standard shared-expert MoE. A fixed validation-selected, source-blind candidate selector and neural/SVM probability blend are reported as separate system views, never substituted for raw neural outputs in architectural inference. Pinned external systems are evaluated by the same locked scripts and split contracts.

### 7.2 Development-only experiments

All architecture, LR, replay-retention, upcycling, router-repair, optimizer, and 2^4 factorial decisions use training/validation data only. The following table is diagnostic and must not be cited as locked test evidence.

{development}

### 7.3 Locked evaluation and statistics

After all development choices, code/config/data/tokenizer/checkpoint/calibration hashes are frozen. One scripted pass produces normalization, identification, external, robustness, routing, and ablation predictions. Primary intervals use paired global-semantic-group bootstrap resampling: each group receives one multiplicity reused across seeds, architectures, and every dialect realization containing that group. Paired randomization likewise draws one treatment/control swap per global semantic group and reuses it across repeated outputs. Holm correction is applied within registered families. A claim of improvement requires the confidence interval—not merely the mean—to exclude zero in the favorable direction.

## 8. Results

### 8.1 Locked main results

{locked_table(locked)}

If the table is populated, all values are descriptive seed aggregates from frozen evaluation manifests. Inferential claims must be taken from `reports/statistics/locked_test_v1/confirmatory_statistics.csv` and must respect familywise correction.

### 8.2 Dialect-level behavior and calibration

Per-dialect normalization chrF++, identification precision/recall/F1, row-normalized confusion, ECE-15, Brier score, and reliability bins are saved for each run. Worst-present-dialect metrics accompany macro metrics to expose failures hidden by dominant CHI, SYL, or Standard-Bangla groups. Validation and locked confusion matrices are never conflated.

### 8.3 Ablations

The registered factorial study tests dialect evidence, the dialect auxiliary head, source adversary, and task conditioning without test access. Three-seed confirmatory ablations remove or randomize one component at a time, including GroupDRO, synthetic perturbations, replay, Muon, and multi-token prediction. Component necessity is concluded only from the locked three-seed effect family; factorial pilot effects are mechanism-screening evidence.

## 9. Routing, robustness, and efficiency

Routing analysis measures expert-load CV, entropy, dialect–expert normalized mutual information, pairwise dialect JS divergence, task JS divergence, and source predictability across layers. A router that separates datasets but not dialects would contradict the proposed mechanism even if aggregate accuracy improved. Perturbation robustness evaluates registered character, whitespace, punctuation, and code-mixing families by severity. Efficiency is reported as measured end-to-end throughput, memory, total parameters, active parameters, and active fraction.

## 10. Blinded native-speaker evaluation

Machine overlap metrics cannot determine whether a normalization preserves meaning, produces fluent Standard Bangla, or hallucinates unsupported content. The pipeline creates blinded randomized packets for qualified native reviewers covering meaning preservation, fluency, overall quality, and unsupported-content flags. This section must remain a protocol description until completed ratings and inter-rater diagnostics are present. No machine metric is substituted for missing human evidence.

## 11. Reproducibility and artifact availability

The executable pipeline is `tools/run_full_pipeline.py`. It records stage state and separate logs under `reports/pipeline/`. Immutable artifacts include source hashes, licenses, split manifests, exclusion decisions, tokenizer hashes, packed block order, configs, checkpoints, calibration, per-example predictions, routing traces, statistical source tables, environment capture, and publication source data. The Q1 figure suite contains {figure_count} PNG/PDF pairs and a hash manifest. Unit tests cover protocol firewalls, data leakage, tokenizer invariants, parameter ownership, training, evaluation, statistics, robustness, and human-packet handling.

## 12. Limitations

First, the frozen normalization inventory is uneven: only a subset of dialects has independent source-held-out parallel data, and RAJ is zero-shot. Second, published dialect labels sometimes collapse sociolinguistically heterogeneous varieties; the 13-label taxonomy is operational rather than an assertion that boundaries are discrete. Third, the 300M-token foundation is intentionally small and does not test scaling laws. Fourth, automated provenance controls cannot replace community-led linguistic validation. Fifth, the external transcript track covers only available sources and domains. Sixth, sparse routing diagnostics are correlational unless supported by the registered interventions and ablations. Seventh, performance on one written benchmark does not establish spoken-language robustness or suitability for consequential deployment.

## 13. Ethics, identity, and data governance

Dialect labels can stigmatize speakers and may encode geography, class, religion, age, or community membership. Identification outputs should not be used to infer identity, origin, or eligibility. Normalization can erase legitimate linguistic identity; the system is framed as optional transformation, not correction of “incorrect” speech. Raw-source licenses and redistribution conditions are preserved per source. Public redistribution and claims of full linguistic validation remain blocked until the stratified native review is complete. Any release should include the data statement, exclusions, intended uses, prohibited uses, model card, known dialect gaps, and a mechanism for community correction or takedown.

## 14. Conclusion

Boichitro-MoE tests a narrow, falsifiable proposition: task-aware and source-adversarial sparse routing may improve transfer to independent Bangla dialect sources at matched active compute. The contribution is equally the experimental control needed to decide that proposition honestly. Provenance reconstruction, protected OOD evaluation, replay-constrained specialization, one-way protocol freezing, paired inference, and native-speaker validation prevent a positive-looking internal score from becoming an unsupported linguistic claim. If the confirmatory M3 − M2 interval does not exclude zero, the audited benchmark, negative result, and routing diagnosis remain valid contributions.

## Declarations required before submission

- **Author contributions (CRediT):** REQUIRED.
- **Funding:** REQUIRED; state “none” if applicable.
- **Competing interests:** REQUIRED; state “none” if applicable.
- **Ethics/IRB determination for human ratings:** REQUIRED.
- **Informed consent and reviewer compensation:** REQUIRED.
- **Data availability statement:** finalize after license/native-review gate.
- **Code availability statement:** add repository and archival DOI.
- **Generative-AI disclosure:** REQUIRED according to the target journal's current policy.
- **Target-journal formatting and word/figure limits:** REQUIRED after journal selection.

## References (journal formatting to be finalized)

1. *Vashantor: A Large-scale Multilingual Benchmark Dataset for Automated Translation of Bangla Regional Dialects to Bangla Language.* [arXiv:2311.11142](https://arxiv.org/abs/2311.11142).
2. *BanglaDial: A merged and imbalanced text dataset for Bengali regional dialect analysis.* [PubMed Central record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12597015/).
3. *ChatgaiyyaAlap: a Chittagonian–Standard Bangla resource.* [PubMed Central record](https://pmc.ncbi.nlm.nih.gov/articles/PMC11925091/).
4. *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding Evaluation in Bangla.* [ACL Anthology](https://aclanthology.org/2022.findings-naacl.98/).
5. *BhasaBodh.* [ACL Anthology](https://aclanthology.org/2025.banglalp-1.9/).
6. *DeepSeekMoE: Towards Ultimate Expert Specialization.* [arXiv:2401.06066](https://arxiv.org/abs/2401.06066).
7. *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts.* [arXiv:2408.15664](https://arxiv.org/abs/2408.15664).
8. *Upcycling Large Language Models into Mixture of Experts.* [arXiv:2410.07524](https://arxiv.org/abs/2410.07524).
9. *OLMoE: Open Mixture-of-Experts Language Models.* [arXiv:2409.02060](https://arxiv.org/abs/2409.02060).
10. *Muon is Scalable for LLM Training.* [arXiv:2502.16982](https://arxiv.org/abs/2502.16982).
11. *Tokenizer Choice for LLM Training: Negligible or Crucial?* [ACL Anthology](https://aclanthology.org/2024.findings-naacl.247/).
12. *Training Compute-Optimal Large Language Models.* [arXiv:2203.15556](https://arxiv.org/abs/2203.15556).

## Figure and table provenance

All captions and file hashes are in `figures/q1/figure_manifest.json` and `figures/q1/FIGURE_CAPTIONS.md`. Table source data and LaTeX exports are under `tables/paper/`. Development-only and locked-test figures are explicitly labeled in the figure manifest.
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "BUILT",
                "output": str(OUTPUT),
                "pipeline_status": pipeline_status,
                "locked_main_complete": bool(locked_complete),
                "native_review_fraction": native_fraction,
                "figure_pairs": figure_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
