"""Create a clean, report-ready results_final folder for Experiment 1."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
LATEST_DIR = THIS_DIR / "outputs" / "latest"
LATEST_SUMMARY_DIR = LATEST_DIR / "summary"
LATEST_PLOTS_DIR = LATEST_DIR / "plots"
LATEST_CM_DIR = LATEST_DIR / "confusion_matrices"
RESULTS_FINAL_DIR = THIS_DIR / "results_final"


def _safe_copy(src: Path, dst: Path, notes: list[str]) -> bool:
    if not src.exists():
        notes.append(f"[missing] {src}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    notes.append(f"[copied] {src} -> {dst}")
    return True


def _format_model_line(label: str, row: pd.Series) -> str:
    return (
        f"- {label}: {row['name']} "
        f"(Test F1={row['test_f1']:.4f}, Test Accuracy={row['test_accuracy']:.4f})"
    )


def _precision_recall_note(cnn_row: pd.Series, bilstm_row: pd.Series) -> str:
    cnn_p = float(cnn_row.get("test_precision", float("nan")))
    cnn_r = float(cnn_row.get("test_recall", float("nan")))
    bilstm_p = float(bilstm_row.get("test_precision", float("nan")))
    bilstm_r = float(bilstm_row.get("test_recall", float("nan")))

    if pd.isna(cnn_p) or pd.isna(cnn_r) or pd.isna(bilstm_p) or pd.isna(bilstm_r):
        return "- Precision/recall comparison: unavailable in summary table."

    precision_winner = "CNN" if cnn_p > bilstm_p else "BiLSTM"
    recall_winner = "CNN" if cnn_r > bilstm_r else "BiLSTM"

    return (
        "- Precision/recall: "
        f"CNN (P={cnn_p:.4f}, R={cnn_r:.4f}) vs "
        f"BiLSTM (P={bilstm_p:.4f}, R={bilstm_r:.4f}). "
        f"Higher precision: {precision_winner}; higher recall: {recall_winner}."
    )


def _get_best_rows(df: pd.DataFrame) -> Dict[str, pd.Series]:
    required_cols = {"name", "family", "test_f1", "test_accuracy"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Summary file missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Summary file contains no experiments.")

    best_overall = df.sort_values("test_f1", ascending=False).iloc[0]

    cnn_df = df[df["family"] == "cnn"]
    bilstm_df = df[df["family"] == "bilstm"]
    if cnn_df.empty or bilstm_df.empty:
        raise ValueError("Could not find both CNN and BiLSTM experiments in summary.")

    best_cnn = cnn_df.sort_values("test_f1", ascending=False).iloc[0]
    best_bilstm = bilstm_df.sort_values("test_f1", ascending=False).iloc[0]

    return {"overall": best_overall, "cnn": best_cnn, "bilstm": best_bilstm}


def _copy_model_artifacts(row: pd.Series, prefix: str, notes: list[str]) -> None:
    model_name = row["name"]
    curve_src = LATEST_PLOTS_DIR / f"{model_name}_training_curves.png"
    cm_src = LATEST_CM_DIR / f"{model_name}_test_cm.png"

    _safe_copy(curve_src, RESULTS_FINAL_DIR / f"{prefix}_best_training_curve.png", notes)
    _safe_copy(cm_src, RESULTS_FINAL_DIR / f"{prefix}_best_confusion_matrix_test.png", notes)


def _write_final_summary(best_rows: Dict[str, pd.Series], notes: list[str]) -> None:
    overall = best_rows["overall"]
    cnn = best_rows["cnn"]
    bilstm = best_rows["bilstm"]

    better_family = "CNN" if float(cnn["test_f1"]) > float(bilstm["test_f1"]) else "BiLSTM"

    lines = [
        "Experiment 1 Final Summary",
        "=" * 28,
        "",
        "Best models",
        _format_model_line("Overall best", overall),
        _format_model_line("Best CNN", cnn),
        _format_model_line("Best BiLSTM", bilstm),
        "",
        "Brief comparison",
        f"- Better family by Test F1: {better_family}.",
        _precision_recall_note(cnn, bilstm),
        "",
        "Notes",
        "- This summary is generated from outputs/latest artifacts only.",
        "- Missing artifacts are listed in collection_notes.txt.",
    ]

    out_path = RESULTS_FINAL_DIR / "final_summary.txt"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    notes.append(f"[written] {out_path}")


def main() -> None:
    notes: list[str] = []

    if RESULTS_FINAL_DIR.exists():
        shutil.rmtree(RESULTS_FINAL_DIR)
    RESULTS_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    all_summary_csv = LATEST_SUMMARY_DIR / "all_experiments_summary.csv"
    leaderboard_csv = LATEST_SUMMARY_DIR / "leaderboard_by_test_f1.csv"

    _safe_copy(all_summary_csv, RESULTS_FINAL_DIR / "all_experiments_summary.csv", notes)
    _safe_copy(leaderboard_csv, RESULTS_FINAL_DIR / "leaderboard_by_test_f1.csv", notes)

    if not all_summary_csv.exists():
        note_path = RESULTS_FINAL_DIR / "collection_notes.txt"
        note_path.write_text(
            "Cannot proceed: missing all_experiments_summary.csv under outputs/latest/summary.\n",
            encoding="utf-8",
        )
        print(f"Created {RESULTS_FINAL_DIR}, but required summary missing: {all_summary_csv}")
        return

    df = pd.read_csv(all_summary_csv)
    best_rows = _get_best_rows(df)

    _copy_model_artifacts(best_rows["cnn"], "cnn", notes)
    _copy_model_artifacts(best_rows["bilstm"], "bilstm", notes)
    _copy_model_artifacts(best_rows["overall"], "overall", notes)

    comparison_src = LATEST_PLOTS_DIR / "best_models_comparison.png"
    _safe_copy(comparison_src, RESULTS_FINAL_DIR / "overall_best_model.png", notes)

    _write_final_summary(best_rows, notes)

    notes_path = RESULTS_FINAL_DIR / "collection_notes.txt"
    notes_path.write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"Created results_final at: {RESULTS_FINAL_DIR}")


if __name__ == "__main__":
    main()
