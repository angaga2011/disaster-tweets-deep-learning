"""Dataset exploration artifacts for Experiment 1."""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

from utils import write_text


def _token_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def generate_exploration_artifacts(train_df: pd.DataFrame, out_dir: str) -> Dict:
    os.makedirs(out_dir, exist_ok=True)

    text_col = "processed_text" if "processed_text" in train_df.columns else "text"
    target_col = "target"

    token_len = _token_count(train_df[text_col])
    char_len = train_df[text_col].fillna("").astype(str).str.len()

    class_counts = train_df[target_col].value_counts().sort_index()
    class_props = train_df[target_col].value_counts(normalize=True).sort_index()

    # 1) class distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(i) for i in class_counts.index], class_counts.values)
    ax.set_title("Class Distribution (target)")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Count")
    for i, v in enumerate(class_counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) token length histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(token_len, bins=40)
    ax.set_title("Tweet Length Distribution (tokens)")
    ax.set_xlabel("Tokens per tweet")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "token_length_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) token length by class boxplot
    fig, ax = plt.subplots(figsize=(7, 4))
    grouped = [
        token_len[train_df[target_col] == 0].values,
        token_len[train_df[target_col] == 1].values,
    ]
    ax.boxplot(grouped, labels=["0", "1"], showfliers=False)
    ax.set_title("Token Length by Class")
    ax.set_xlabel("Class label")
    ax.set_ylabel("Tokens per tweet")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "token_length_by_class_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4) missingness chart for key columns
    key_cols = [c for c in ["keyword", "location", "text", "processed_text", "target"] if c in train_df.columns]
    missing_pct = train_df[key_cols].isna().mean().mul(100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(missing_pct.index, missing_pct.values)
    ax.set_title("Missing Values (%) by Column")
    ax.set_ylabel("Percent missing")
    ax.set_ylim(0, max(5, float(missing_pct.max()) + 5))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "missing_values_percent.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5) sample rows text file
    sample_cols = [c for c in ["id", "keyword", "location", "text", "processed_text", "target"] if c in train_df.columns]
    sample_df = train_df[sample_cols].sample(min(12, len(train_df)), random_state=42)
    write_text(
        os.path.join(out_dir, "sample_rows.txt"),
        sample_df.to_string(index=False),
    )

    # 6) textual exploration summary
    q = token_len.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    imbalance_ratio = (class_counts.max() / class_counts.min()) if class_counts.min() > 0 else float("inf")
    summary_lines = [
        "Dataset exploration summary",
        f"Rows: {len(train_df)}",
        f"Text column used: {text_col}",
        f"Class counts: {class_counts.to_dict()}",
        f"Class proportions: {class_props.round(4).to_dict()}",
        f"Imbalance ratio (majority/minority): {imbalance_ratio:.3f}",
        f"Token length mean: {token_len.mean():.2f}",
        f"Token length median (p50): {q.get(0.5, 0):.2f}",
        f"Token length p90: {q.get(0.9, 0):.2f}",
        f"Token length p95: {q.get(0.95, 0):.2f}",
        f"Token length p99: {q.get(0.99, 0):.2f}",
        f"Character length mean: {char_len.mean():.2f}",
        "Missingness (%): " + ", ".join([f"{k}={v:.2f}" for k, v in missing_pct.to_dict().items()]),
    ]
    write_text(os.path.join(out_dir, "exploration_summary.txt"), "\n".join(summary_lines))

    return {
        "class_counts": class_counts.to_dict(),
        "class_proportions": class_props.to_dict(),
        "imbalance_ratio": float(imbalance_ratio),
        "token_length_mean": float(token_len.mean()),
        "token_length_median": float(q.get(0.5, 0)),
        "token_length_p95": float(q.get(0.95, 0)),
    }
