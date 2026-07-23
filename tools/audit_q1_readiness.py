#!/usr/bin/env python3
"""Audit computational completion and honest Q1-submission readiness."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-figures", type=int, default=30)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def glob_count(pattern: str) -> int:
    return sum(1 for path in PROJECT.glob(pattern) if path.is_file())


def status_is(path: str, accepted: set[str]) -> tuple[bool, str]:
    payload = load_json(PROJECT / path)
    if not isinstance(payload, dict):
        return False, "missing or invalid JSON"
    status = str(payload.get("status", "missing"))
    return status in accepted, status


def task_run_count(protocol: str, suffix: str = "base") -> tuple[int, int]:
    complete = 0
    discovered = 0
    for variant in ("M0", "M1", "M2", "M3"):
        for seed in (1701, 2903, 4307):
            root = PROJECT / "runs/task" / protocol / f"{variant}__{suffix}" / str(seed)
            if root.exists():
                discovered += 1
            manifest = load_json(root / "run_manifest.json")
            s_report = load_json(root / "stage_s/training_report.json")
            i_report = load_json(root / "stage_id/training_report.json")
            if (
                isinstance(manifest, dict)
                and manifest.get("status") == "COMPLETE_VALIDATION_ONLY"
                and isinstance(s_report, dict)
                and s_report.get("status") == "COMPLETE"
                and isinstance(i_report, dict)
                and i_report.get("status") == "COMPLETE"
                and (root / "stage_s/best_checkpoint.pt").exists()
                and (root / "stage_id/best_checkpoint.pt").exists()
                and (root / "stage_id/temperature_calibration.json").exists()
            ):
                complete += 1
    return complete, discovered


def ablation_run_count(protocol: str, suffixes: list[str]) -> tuple[int, int]:
    complete = 0
    expected = len(suffixes) * 3
    for suffix in suffixes:
        for seed in (1701, 2903, 4307):
            root = PROJECT / "runs/task" / protocol / f"M3__{suffix}" / str(seed)
            manifest = load_json(root / "run_manifest.json")
            if isinstance(manifest, dict) and manifest.get("status") == "COMPLETE_VALIDATION_ONLY":
                complete += 1
    return complete, expected


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        label: str,
        passed: bool,
        evidence: str,
        *,
        scope: str = "computational",
        blocking: bool = True,
    ) -> None:
        checks.append(
            {
                "id": check_id,
                "label": label,
                "passed": bool(passed),
                "evidence": evidence,
                "scope": scope,
                "blocking": blocking,
            }
        )

    data_gate = load_json(PROJECT / "reports/data_gate.json")
    add(
        "data_engineering_gate",
        "Frozen data engineering gate",
        isinstance(data_gate, dict) and data_gate.get("status") == "pass" and data_gate.get("training_authorized") is True,
        "reports/data_gate.json",
    )
    tokenizer_ok, tokenizer_status = status_is(
        "reports/tokenizer/TOKENIZER_FREEZE_REPORT.json", {"TOKENIZER_FROZEN"}
    )
    add("tokenizer_freeze", "Frozen tokenizer selected without test access", tokenizer_ok, f"reports/tokenizer/TOKENIZER_FREEZE_REPORT.json ({tokenizer_status})")

    foundation_runs = ["F_DENSE_300M", "U_M0_DENSE_200M", "U_M1_SWITCH_200M", "U_M2_STANDARD_MOE_200M"]
    foundation_complete = 0
    for run_id in foundation_runs:
        payload = load_json(PROJECT / "runs" / run_id / "1701/training_report.json")
        if isinstance(payload, dict) and payload.get("status") == "COMPLETE_FIXED_BUDGET":
            foundation_complete += 1
    add(
        "foundation_training",
        "Foundation and matched continuations",
        foundation_complete == len(foundation_runs),
        f"{foundation_complete}/{len(foundation_runs)} fixed-budget runs complete",
    )

    main_complete, main_discovered = task_run_count("boichitro_q1_v1")
    add(
        "main_task_runs",
        "Four models × three main seeds",
        main_complete == 12,
        f"{main_complete}/12 complete ({main_discovered} run directories discovered)",
    )

    fusion_reports = [
        PROJECT / "reports/model/source_blind_normalization_baseline_audit.json",
        PROJECT / "reports/model/id_fusion_selection.json",
        PROJECT / "reports/model/normalization_fusion_selection_v2.json",
        PROJECT / "reports/model/development_fusion_uncertainty.json",
    ]
    fusion_payloads = [load_json(path) for path in fusion_reports]
    fusion_complete = all(
        isinstance(payload, dict)
        and payload.get("status") == "COMPLETE_VALIDATION_ONLY"
        and payload.get("test_data_access") is False
        for payload in fusion_payloads
    ) and isinstance(fusion_payloads[2], dict) and fusion_payloads[2].get(
        "cross_validation_group_unit"
    ) == "semantic_group_id" and isinstance(
        fusion_payloads[3], dict
    ) and fusion_payloads[3].get(
        "confirmatory_inference"
    ) is False and fusion_payloads[1].get(
        "confirmatory_inference"
    ) is False and fusion_payloads[2].get(
        "confirmatory_inference"
    ) is False
    add(
        "source_blind_fusion",
        "Fair source-blind baselines and validation-selected system fusion",
        fusion_complete,
        ", ".join(str(path.relative_to(PROJECT)) for path in fusion_reports),
    )
    cross_task_firewall = load_json(
        PROJECT / "reports/model/cross_task_input_firewall.json"
    )
    cross_task_complete = (
        isinstance(cross_task_firewall, dict)
        and cross_task_firewall.get("status") == "PASS"
        and cross_task_firewall.get("protected_text_emitted") is False
        and cross_task_firewall.get("model_or_hyperparameter_selection_use") is False
        and bool(cross_task_firewall.get("tracks"))
        and all(
            int(row.get("exact_overlap_rows_with_id_train", 1)) == 0
            and int(row.get("compact_overlap_rows_with_id_train", 1)) == 0
            and int(row.get("near_overlap_rows_with_id_train", 1)) == 0
            for row in cross_task_firewall.get("tracks", [])
        )
    )
    add(
        "cross_task_input_firewall",
        "No normalization evaluation input appears in ID-classifier training",
        cross_task_complete,
        "reports/model/cross_task_input_firewall.json",
    )
    transfer = load_json(
        PROJECT / "reports/model/development_fusion_architecture_transfer.json"
    )
    transfer_complete = (
        isinstance(transfer, dict)
        and transfer.get("status") == "COMPLETE_VALIDATION_ONLY"
        and transfer.get("test_data_access") is False
        and int(transfer.get("completed_runs", 0)) == 6
        and int(transfer.get("expected_runs", 0)) == 6
    )
    add(
        "fusion_architecture_transfer",
        "Fixed fusion transferred without retuning to all M2/M3 development runs",
        transfer_complete,
        (
            "reports/model/development_fusion_architecture_transfer.json "
            f"({transfer.get('completed_runs', 0) if isinstance(transfer, dict) else 0}/6 complete)"
        ),
    )

    optimizer_ok, optimizer_status = status_is(
        "reports/model/optimizer_pilot_selection.json", {"COMPLETE_VALIDATION_ONLY"}
    )
    add("optimizer_pilot", "AdamW-only learning-rate pilot", optimizer_ok, f"reports/model/optimizer_pilot_selection.json ({optimizer_status})")

    bidirectional_complete = 0
    for seed in (1701, 2903, 4307):
        candidates = list((PROJECT / "runs/task").glob(f"*/M3B*/*{seed}*/run_manifest.json"))
        if any(isinstance(load_json(path), dict) and load_json(path).get("status") == "COMPLETE_VALIDATION_ONLY" for path in candidates):
            bidirectional_complete += 1
    add("bidirectional_id", "Three-seed bidirectional ID specialist", bidirectional_complete == 3, f"{bidirectional_complete}/3 run manifests complete")

    confirmatory = [
        "no_lexical_prior",
        "no_dialect_head",
        "no_source_adversary",
        "no_task_conditioning",
        "randomized_lexical_prior",
        "no_groupdro",
        "no_synthetic",
        "no_general_replay",
    ]
    optimization = ["adamw_only", "no_mtp"]
    confirmatory_complete, confirmatory_expected = ablation_run_count("boichitro_confirmatory_ablation_v1", confirmatory)
    optimization_complete, optimization_expected = ablation_run_count("boichitro_optimization_ablation_v1", optimization)
    add("confirmatory_ablations", "Registered three-seed confirmatory ablations", confirmatory_complete == confirmatory_expected, f"{confirmatory_complete}/{confirmatory_expected} complete")
    add("optimization_ablations", "Registered optimizer/MTP ablations", optimization_complete == optimization_expected, f"{optimization_complete}/{optimization_expected} complete")

    factorial_cells = glob_count("runs/task/boichitro_factorial_pilot_v1/M3__*/1701/stage_s/best_selection.json")
    add("factorial_pilot", "Registered 2^4 development factorial", factorial_cells == 16, f"{factorial_cells}/16 selected validation cells")

    freeze_candidates = list((PROJECT / "reports").glob("**/*freeze*.json")) + list((PROJECT / "artifacts").glob("**/*freeze*.json"))
    protocol_frozen = any(
        isinstance(load_json(path), dict)
        and str(load_json(path).get("status", "")).upper() in {"PASS", "FROZEN", "PROTOCOL_FROZEN"}
        for path in freeze_candidates
    )
    add("protocol_freeze", "Immutable protocol freeze before neural test access", protocol_frozen, ", ".join(str(path.relative_to(PROJECT)) for path in freeze_candidates) or "no freeze manifest found")

    locked_main = glob_count("predictions/locked_test_v1/M?__base/*/evaluation_manifest.json")
    add("locked_main", "Scripted locked main evaluation", locked_main == 12, f"{locked_main}/12 evaluation manifests")
    locked_ablation = glob_count("predictions/locked_test_v1/M3__*/*/evaluation_manifest.json") - glob_count("predictions/locked_test_v1/M3__base/*/evaluation_manifest.json")
    add("locked_ablations", "Locked registered ablation evaluation", locked_ablation >= 30, f"{locked_ablation}/30 or more evaluation manifests")

    external_manifests = glob_count("predictions/locked_external_test_v1/**/evaluation_manifest.json")
    add("external_baselines", "Pinned external baselines", external_manifests >= 12, f"{external_manifests} locked external manifests")

    robustness = (PROJECT / "reports/robustness/locked_robustness_v1/robustness_curves.csv").exists()
    routing = (PROJECT / "reports/routing/locked_test_v1/expert_specialization_metrics.parquet").exists()
    statistics = (PROJECT / "reports/statistics/locked_test_v1/confirmatory_statistics.csv").exists()
    add("robustness", "Registered perturbation robustness", robustness, "reports/robustness/locked_robustness_v1/robustness_curves.csv")
    add("routing", "Locked routing specialization analysis", routing, "reports/routing/locked_test_v1/expert_specialization_metrics.parquet")
    add("statistics", "Confirmatory uncertainty and multiplicity control", statistics, "reports/statistics/locked_test_v1/confirmatory_statistics.csv")

    figure_manifest = load_json(PROJECT / "figures/q1/figure_manifest.json")
    figure_count = int(figure_manifest.get("figure_pairs", 0)) if isinstance(figure_manifest, dict) else 0
    paired_files = min(glob_count("figures/q1/*.png"), glob_count("figures/q1/*.pdf"))
    add(
        "figure_suite",
        f"At least {args.minimum_figures} paired journal figures",
        figure_count >= args.minimum_figures and paired_files >= args.minimum_figures,
        f"manifest={figure_count}; paired files={paired_files}; figures/q1/figure_manifest.json",
    )
    figure_validation = load_json(PROJECT / "figures/q1/VALIDATION_REPORT.json")
    add(
        "figure_validation",
        "Figure hash, format, resolution, caption, and source-data validation",
        isinstance(figure_validation, dict) and figure_validation.get("status") == "PASS",
        (
            f"figures/q1/VALIDATION_REPORT.json "
            f"({figure_validation.get('checks_passed', 0)}/{figure_validation.get('checks_total', 0)})"
            if isinstance(figure_validation, dict)
            else "figures/q1/VALIDATION_REPORT.json missing"
        ),
    )

    test_report = load_json(PROJECT / "reports/Q1_TEST_REPORT.json")
    add(
        "regression_tests",
        "Full regression test suite",
        isinstance(test_report, dict)
        and test_report.get("status") == "PASS"
        and int(test_report.get("failed", 1)) == 0,
        (
            f"reports/Q1_TEST_REPORT.json ({test_report.get('passed', 0)} passed)"
            if isinstance(test_report, dict)
            else "reports/Q1_TEST_REPORT.json missing"
        ),
    )

    table_manifest = load_json(PROJECT / "tables/paper/table_manifest.json")
    table_count = len(table_manifest.get("created", [])) if isinstance(table_manifest, dict) else 0
    add("paper_tables", "Reproducible paper tables", table_count >= 8, f"{table_count} table entries in tables/paper/table_manifest.json")

    pipeline = load_json(PROJECT / "reports/pipeline/full_pipeline_state.json")
    pipeline_status = str(pipeline.get("status", "missing")) if isinstance(pipeline, dict) else "missing"
    add("pipeline_state", "Registered end-to-end pipeline", pipeline_status == "COMPLETE", f"reports/pipeline/full_pipeline_state.json ({pipeline_status})")

    native = load_json(PROJECT / "reports/native_review_report.json")
    native_complete = isinstance(native, dict) and native.get("status") == "COMPLETE" and float(native.get("completion_fraction", 0)) == 1.0
    add(
        "native_dataset_review",
        "Stratified native-speaker dataset review",
        native_complete,
        f"reports/native_review_report.json ({native.get('completed_rows', 0) if isinstance(native, dict) else 0}/230)",
        scope="human_submission",
    )

    human_summary = PROJECT / "human_evaluation/blind_native_normalization_v1/human_evaluation_summary.csv"
    add(
        "native_system_ratings",
        "Blinded native-speaker system-output ratings",
        human_summary.exists(),
        str(human_summary.relative_to(PROJECT)),
        scope="human_submission",
    )

    manuscript = PROJECT / "manuscript/BOICHITRO_MOE_Q1_MANUSCRIPT.md"
    add(
        "manuscript",
        "Journal-neutral manuscript draft",
        manuscript.exists(),
        str(manuscript.relative_to(PROJECT)),
        scope="submission_package",
    )
    author_metadata = PROJECT / "manuscript/AUTHOR_METADATA.yaml"
    add(
        "author_metadata",
        "Author, affiliation, CRediT, funding, and conflict metadata",
        author_metadata.exists() and "REQUIRED" not in author_metadata.read_text(encoding="utf-8"),
        str(author_metadata.relative_to(PROJECT)),
        scope="submission_package",
    )
    target_journal = PROJECT / "manuscript/TARGET_JOURNAL.md"
    add(
        "target_journal",
        "Target journal and current author-guideline adaptation",
        target_journal.exists() and "REQUIRED" not in target_journal.read_text(encoding="utf-8"),
        str(target_journal.relative_to(PROJECT)),
        scope="submission_package",
    )

    computational = [row for row in checks if row["scope"] == "computational" and row["blocking"]]
    human = [row for row in checks if row["scope"] == "human_submission" and row["blocking"]]
    package = [row for row in checks if row["scope"] == "submission_package" and row["blocking"]]
    computational_complete = all(row["passed"] for row in computational)
    human_complete = all(row["passed"] for row in human)
    package_complete = all(row["passed"] for row in package)
    submission_ready = computational_complete and human_complete and package_complete

    result = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "SUBMISSION_READY" if submission_ready else ("COMPUTATION_COMPLETE_HUMAN_ACTION_REQUIRED" if computational_complete else "INCOMPLETE"),
        "computational_pipeline_complete": computational_complete,
        "human_validation_complete": human_complete,
        "submission_package_complete": package_complete,
        "submission_ready": submission_ready,
        "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks),
        "checks": checks,
        "claim_guard": {
            "q1_acceptance_guaranteed": False,
            "publication_ready_corpus_claim_allowed": native_complete,
            "native_quality_claim_allowed": human_complete,
            "computational_claims_allowed": computational_complete,
        },
    }
    json_path = PROJECT / "reports/Q1_JOURNAL_READINESS_AUDIT.json"
    md_path = PROJECT / "reports/Q1_JOURNAL_READINESS_AUDIT.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Q1 journal readiness audit",
        "",
        f"Status: **{result['status']}**",
        "",
        f"Automated computational pipeline complete: **{'yes' if computational_complete else 'no'}**  ",
        f"Required human validation complete: **{'yes' if human_complete else 'no'}**  ",
        f"Submission package complete: **{'yes' if package_complete else 'no'}**  ",
        f"Overall submission-ready: **{'yes' if submission_ready else 'no'}**",
        "",
        "A Q1 acceptance outcome cannot be guaranteed by an artifact audit. The audit only establishes reproducibility, protocol completion, and whether required human evidence is present.",
        "",
        "## Checks",
        "",
        "| Result | Scope | Check | Evidence |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(f"| {'PASS' if row['passed'] else 'FAIL'} | {row['scope']} | {row['label']} | {row['evidence']} |")
    failed = [row for row in checks if not row["passed"] and row["blocking"]]
    lines.extend(["", "## Remaining blockers", ""])
    if failed:
        for row in failed:
            lines.append(f"- **{row['label']}** — {row['evidence']}")
    else:
        lines.append("- None detected by the registered audit.")
    lines.extend(
        [
            "",
            "## Claim discipline",
            "",
            "- Do not describe the corpus as fully linguistically validated until the 230-row native review is complete.",
            "- Do not substitute machine metrics for blinded native-speaker system ratings.",
            "- Report validation-only pilot and ablation evidence separately from locked-test confirmatory evidence.",
            "- Do not describe the work as accepted by, or guaranteed suitable for, a Q1 journal; quartile and editorial decisions are external.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": f"{result['checks_passed']}/{result['checks_total']}", "report": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
