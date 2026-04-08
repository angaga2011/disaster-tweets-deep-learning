"""Plotting utilities for training curves and confusion matrices."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def save_training_curves(history, model_name: str, out_path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(history.history.get("loss", []), label="train")
    ax[0].plot(history.history.get("val_loss", []), label="val")
    ax[0].set_title(f"{model_name} - Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.history.get("accuracy", []), label="train")
    ax[1].plot(history.history.get("val_accuracy", []), label="val")
    ax[1].set_title(f"{model_name} - Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(cm: List[List[int]], title: str, out_path: str) -> None:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_best_models_comparison_plot(results_df: pd.DataFrame, out_path: str) -> None:
    if results_df.empty:
        return
    idx = results_df.groupby("family")["test_f1"].idxmax()
    best_df = results_df.loc[idx].copy()
    overall_best = results_df.loc[[results_df["test_f1"].idxmax()]]
    combined = pd.concat([best_df, overall_best], ignore_index=True)
    combined["label"] = combined["name"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(len(combined))
    ax.bar(x, combined["test_f1"], label="Test F1")
    ax.plot(x, combined["test_accuracy"], marker="o", linewidth=2, label="Test Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(combined["label"], rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Best Models Comparison")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
