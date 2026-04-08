"""Reporting layer for global summaries and markdown analysis."""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from plotting import save_best_models_comparison_plot
from utils import write_text


def _df_to_markdown_safe(df: pd.DataFrame) -> str:
    """Render markdown table without requiring optional tabulate dependency."""
    if df.empty:
        return "_No rows_"
    headers = [str(c) for c in df.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in df.itertuples(index=False, name=None):
        vals = [str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "name": r["name"],
                "family": r["family"],
                "validation_accuracy": r["validation_accuracy"],
                "validation_precision": r["validation_precision"],
                "validation_recall": r["validation_recall"],
                "validation_f1": r["validation_f1"],
                "test_accuracy": r["test_accuracy"],
                "test_precision": r["test_precision"],
                "test_recall": r["test_recall"],
                "test_f1": r["test_f1"],
                "epochs_ran": r["epochs_ran"],
                "dropout": r["config"]["dropout"],
                "l2_value": r["config"]["l2_value"],
                "batch_norm": r["config"]["batch_norm"],
                "filters": r["config"]["filters"],
                "kernel_size": r["config"]["kernel_size"],
                "lstm_units": r["config"]["lstm_units"],
                "stacked": r["config"]["stacked"],
                "dense_units": r["config"]["dense_units"],
            }
        )
    return pd.DataFrame(rows)


def _observation_lines(df: pd.DataFrame) -> List[str]:
    lines = []
    if df.empty:
        return ["No experiments were run."]

    best_f1 = df.sort_values("test_f1", ascending=False).iloc[0]
    best_acc = df.sort_values("test_accuracy", ascending=False).iloc[0]
    lines.append(
        f"- Overall best by Test F1: `{best_f1['name']}` ({best_f1['test_f1']:.4f})."
    )
    lines.append(
        f"- Overall best by Test Accuracy: `{best_acc['name']}` ({best_acc['test_accuracy']:.4f})."
    )

    fam = df.groupby("family")[["test_f1", "test_accuracy"]].mean().reset_index()
    if len(fam) >= 2:
        fam_rows = [
            f"`{row['family']}` mean Test F1={row['test_f1']:.4f}, Test Acc={row['test_accuracy']:.4f}"
            for _, row in fam.iterrows()
        ]
        lines.append("- Family-level means: " + "; ".join(fam_rows) + ".")

    bn_delta = (
        df.groupby("batch_norm")[["test_f1", "test_accuracy"]].mean().reset_index()
        if df["batch_norm"].nunique() > 1
        else None
    )
    if bn_delta is not None:
        lines.append(
            "- Batch normalization comparison (mean): "
            + ", ".join(
                [
                    f"batch_norm={row['batch_norm']}: F1={row['test_f1']:.4f}, Acc={row['test_accuracy']:.4f}"
                    for _, row in bn_delta.iterrows()
                ]
            )
            + "."
        )

    dr = df.groupby("dropout")[["test_f1", "test_accuracy"]].mean().reset_index()
    if len(dr) > 1:
        lines.append(
            "- Dropout comparison (mean): "
            + ", ".join(
                [
                    f"dropout={row['dropout']}: F1={row['test_f1']:.4f}, Acc={row['test_accuracy']:.4f}"
                    for _, row in dr.iterrows()
                ]
            )
            + "."
        )

    l2 = df.groupby("l2_value")[["test_f1", "test_accuracy"]].mean().reset_index()
    if len(l2) > 1:
        lines.append(
            "- L2 comparison (mean): "
            + ", ".join(
                [
                    f"l2={row['l2_value']}: F1={row['test_f1']:.4f}, Acc={row['test_accuracy']:.4f}"
                    for _, row in l2.iterrows()
                ]
            )
            + "."
        )

    gap = (df["validation_f1"] - df["test_f1"]).mean()
    lines.append(
        f"- Mean validation-to-test F1 gap is {gap:.4f}; positive values suggest some overfitting."
    )
    return lines


def write_summary_artifacts(results: List[Dict], meta: Dict, summary_dir: str, plots_dir: str) -> None:
    os.makedirs(summary_dir, exist_ok=True)
    df = _results_to_dataframe(results)
    df.to_csv(os.path.join(summary_dir, "all_experiments_summary.csv"), index=False)

    by_f1 = df.sort_values("test_f1", ascending=False)
    by_acc = df.sort_values("test_accuracy", ascending=False)
    by_f1.to_csv(os.path.join(summary_dir, "leaderboard_by_test_f1.csv"), index=False)
    by_acc.to_csv(os.path.join(summary_dir, "leaderboard_by_test_accuracy.csv"), index=False)

    save_best_models_comparison_plot(
        df, os.path.join(plots_dir, "best_models_comparison.png")
    )

    best_cnn = by_f1[by_f1["family"] == "cnn"].head(1)
    best_bilstm = by_f1[by_f1["family"] == "bilstm"].head(1)
    best_f1 = by_f1.head(1)
    best_acc = by_acc.head(1)

    report_lines = [
        "# Experiment 1 Report",
        "",
        "## Dataset and Vectorization",
        f"- Train samples: {meta['split_sizes']['train']}",
        f"- Validation samples: {meta['split_sizes']['validation']}",
        f"- Test samples: {meta['split_sizes']['test']}",
        f"- Vocabulary size: {meta['vocab_size']}",
        f"- Sequence length: {meta['seq_len']}",
        "",
        "## Experiment Configurations",
        "",
        _df_to_markdown_safe(
            df[
            [
                "name",
                "family",
                "dropout",
                "l2_value",
                "batch_norm",
                "filters",
                "kernel_size",
                "lstm_units",
                "stacked",
                "dense_units",
            ]
            ]
        ),
        "",
        "## Main Metrics",
        "",
        _df_to_markdown_safe(
            df[
                [
                    "name",
                    "family",
                    "validation_accuracy",
                    "validation_f1",
                    "test_accuracy",
                    "test_f1",
                ]
            ].sort_values("test_f1", ascending=False)
        ),
        "",
        "## Best Models",
        "",
        "### Best CNN",
        _df_to_markdown_safe(best_cnn) if not best_cnn.empty else "No CNN run.",
        "",
        "### Best BiLSTM",
        _df_to_markdown_safe(best_bilstm) if not best_bilstm.empty else "No BiLSTM run.",
        "",
        "### Overall Best by Test F1",
        _df_to_markdown_safe(best_f1),
        "",
        "### Overall Best by Test Accuracy",
        _df_to_markdown_safe(best_acc),
        "",
        "## Observations",
        *_observation_lines(df),
        "",
        "## Notes",
        "- Claims are based only on measured validation/test metrics from this run.",
        "- Compare this report against future archived runs to track stability across random seeds and settings.",
    ]

    write_text(os.path.join(summary_dir, "experiment1_report.md"), "\n".join(report_lines))
