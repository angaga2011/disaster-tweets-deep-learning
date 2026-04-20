"""
Experiment 3: State-of-the-Art Model — BERTweet for Disaster Tweet Classification
Compares:
  (A) Fine-tuned BERTweet (pre-trained on 850M tweets)
  (B) BERTweet architecture trained from scratch (random init)

Dataset: NLP with Disaster Tweets (Kaggle)

Usage:
  python experiment3_sota.py              # uses saved checkpoints if available
  python experiment3_sota.py --retrain    # forces full re-training
"""

import os
import re
import json
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaConfig,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == "cpu":
    print("  ⚠  No GPU detected — running on CPU.")
    print("     Estimated time: ~4–8 hrs per model. Consider Google Colab (free GPU).")
    print("     Tip: reduce BATCH_SIZE to 8 if you run out of memory.\n")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = "Dataset"
RESULTS_DIR    = "results/experiment3"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR,    exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── Per-model checkpoint sub-folders ─────────────────────────────────────────
CKPT_A = os.path.join(CHECKPOINT_DIR, "model_a")   # pre-trained
CKPT_B = os.path.join(CHECKPOINT_DIR, "model_b")   # from scratch

# ── Hyperparameters ──────────────────────────────────────────────────────────
MAX_LEN    = 128
BATCH_SIZE = 16   # Lower to 8 if you get OOM errors on CPU
EPOCHS     = 4
LR         = 2e-5

MODEL_NAME = "vinai/bertweet-base"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def normalize_tweet(text: str) -> str:
    """Basic tweet normalization (BERTweet recommends @USER / HTTPURL tokens)."""
    text = re.sub(r"http\S+|www\S+", "HTTPURL", text)
    text = re.sub(r"@\w+", "@USER", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data():
    print("Loading and preprocessing data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_df["text"] = train_df["text"].fillna("").apply(normalize_tweet)
    test_df["text"]  = test_df["text"].fillna("").apply(normalize_tweet)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df["text"].tolist(),
        train_df["target"].tolist(),
        test_size=0.2,
        random_state=SEED,
        stratify=train_df["target"]
    )

    print(f"  Train: {len(train_texts)} | Val: {len(val_texts)} | Test: {len(test_df)}")
    return train_texts, val_texts, train_labels, val_labels, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.labels[idx]
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def checkpoint_exists(ckpt_dir: str) -> bool:
    """Returns True if a saved model + metadata exist in ckpt_dir."""
    return (
        os.path.isdir(ckpt_dir) and
        os.path.isfile(os.path.join(ckpt_dir, "pytorch_model.bin")) and
        os.path.isfile(os.path.join(ckpt_dir, "history.json"))      and
        os.path.isfile(os.path.join(ckpt_dir, "best_preds.npy"))    and
        os.path.isfile(os.path.join(ckpt_dir, "best_labels.npy"))
    )


def save_checkpoint(model, history: dict, best_preds: list, best_labels: list,
                    ckpt_dir: str):
    """Save model weights, training history, and best predictions."""
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    with open(os.path.join(ckpt_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.save(os.path.join(ckpt_dir, "best_preds.npy"),  np.array(best_preds))
    np.save(os.path.join(ckpt_dir, "best_labels.npy"), np.array(best_labels))
    print(f"  ✓ Checkpoint saved → {ckpt_dir}")


def load_checkpoint(ckpt_dir: str):
    """Load history, predictions, and labels from checkpoint."""
    with open(os.path.join(ckpt_dir, "history.json")) as f:
        history = json.load(f)
    best_preds  = np.load(os.path.join(ckpt_dir, "best_preds.npy")).tolist()
    best_labels = np.load(os.path.join(ckpt_dir, "best_labels.npy")).tolist()
    return history, best_preds, best_labels


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING & EVALUATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, epoch, total_epochs):
    model.train()
    total_loss, preds_all, labels_all = 0, [], []

    pbar = tqdm(loader, desc=f"  Epoch {epoch}/{total_epochs} [train]",
                unit="batch", dynamic_ncols=True)

    for batch in pbar:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds_all.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    train_f1 = f1_score(labels_all, preds_all, average="weighted")
    return avg_loss, train_f1


def eval_epoch(model, loader, split="val"):
    model.eval()
    total_loss, preds_all, labels_all = 0, [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"           [{split}] ", unit="batch",
                    dynamic_ncols=True, leave=False)
        for batch in pbar:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs    = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds_all.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(labels_all, preds_all)
    f1       = f1_score(labels_all, preds_all, average="weighted")
    prec     = precision_score(labels_all, preds_all, average="weighted", zero_division=0)
    rec      = recall_score(labels_all, preds_all, average="weighted", zero_division=0)
    return avg_loss, acc, f1, prec, rec, preds_all, labels_all


def format_elapsed(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def run_experiment(model, tokenizer, train_texts, val_texts, train_labels, val_labels,
                   experiment_name, ckpt_dir, retrain: bool):
    """
    Trains (or loads from checkpoint) one model.
    Returns: history dict, best_preds, best_labels
    """

    # ── Load from checkpoint if available and --retrain not set ──────────────
    if not retrain and checkpoint_exists(ckpt_dir):
        print(f"\n{'='*60}")
        print(f"  {experiment_name}")
        print(f"  Checkpoint found — skipping training.")
        print(f"  Loading from: {ckpt_dir}")
        print(f"{'='*60}")
        history, best_preds, best_labels = load_checkpoint(ckpt_dir)
        return history, best_preds, best_labels

    # ── Tokenize datasets (batch tokenization — much faster than per-sample) ──
    print(f"\n{'='*60}")
    print(f"  {experiment_name}")
    print(f"{'='*60}")
    print(f"  Tokenizing {len(train_texts)} training samples...", flush=True)
    t0 = time.time()
    train_dataset = TweetDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    print(f"  Tokenizing {len(val_texts)} validation samples...", flush=True)
    val_dataset   = TweetDataset(val_texts,   val_labels,   tokenizer, MAX_LEN)
    print(f"  Tokenization done in {format_elapsed(time.time() - t0)}.", flush=True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * EPOCHS
    print(f"  Batches/epoch: {steps_per_epoch} | Total steps: {total_steps}")
    print(f"  Starting training...\n", flush=True)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    history = {"train_loss": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_preds, best_labels = -1.0, [], []
    best_model_tmp = os.path.join(ckpt_dir, "_best_tmp")
    experiment_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, scheduler,
                                           epoch, EPOCHS)
        val_loss, val_acc, val_f1, val_prec, val_rec, preds, true_labels = eval_epoch(
            model, val_loader)

        elapsed    = format_elapsed(time.time() - epoch_start)
        total_done = format_elapsed(time.time() - experiment_start)

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        marker = " ★ best" if val_f1 > best_f1 else ""
        print(f"\n  Epoch {epoch}/{EPOCHS} ({elapsed} | total {total_done})")
        print(f"    train_loss={train_loss:.4f}  train_f1={train_f1:.4f}")
        print(f"    val_loss  ={val_loss:.4f}  val_acc ={val_acc:.4f}  "
              f"val_f1={val_f1:.4f}{marker}", flush=True)

        if val_f1 > best_f1:
            best_f1     = val_f1
            best_preds  = list(preds)
            best_labels = list(true_labels)
            os.makedirs(best_model_tmp, exist_ok=True)
            model.save_pretrained(best_model_tmp)
            print(f"    → Best model saved (val_f1={best_f1:.4f})", flush=True)

    # Promote best weights: copy files from _best_tmp into ckpt_dir
    import shutil
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.isdir(best_model_tmp):
        for fname in os.listdir(best_model_tmp):
            shutil.copy2(os.path.join(best_model_tmp, fname),
                         os.path.join(ckpt_dir, fname))
        shutil.rmtree(best_model_tmp)

    save_checkpoint(model, history, best_preds, best_labels, ckpt_dir)
    print(f"\n  Training complete in {format_elapsed(time.time() - experiment_start)}.")
    return history, best_preds, best_labels


# ─────────────────────────────────────────────────────────────────────────────
# 5. PLOTTING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(history_a, history_b, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history_a["train_loss"]) + 1)

    axes[0].plot(epochs, history_a["train_loss"], "b-o",  label="Pre-trained train")
    axes[0].plot(epochs, history_a["val_loss"],   "b--o", label="Pre-trained val")
    axes[0].plot(epochs, history_b["train_loss"], "r-s",  label="Scratch train")
    axes[0].plot(epochs, history_b["val_loss"],   "r--s", label="Scratch val")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, history_a["val_f1"], "b-o", label="Pre-trained val F1")
    axes[1].plot(epochs, history_b["val_f1"], "r-s", label="Scratch val F1")
    axes[1].set_title("Validation F1 Score")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Weighted F1")
    axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(true_labels, preds, title, save_path):
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Disaster", "Disaster"],
                yticklabels=["Non-Disaster", "Disaster"])
    plt.title(title)
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def plot_metrics_comparison(metrics_a, metrics_b, save_path):
    metric_names = ["Accuracy", "F1 Score", "Precision", "Recall"]
    x     = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_a = ax.bar(x - width/2, [metrics_a[m] for m in metric_names], width,
                    label="Pre-trained BERTweet", color="#4C72B0")
    bars_b = ax.bar(x + width/2, [metrics_b[m] for m in metric_names], width,
                    label="BERTweet from Scratch", color="#DD8452")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x); ax.set_xticklabels(metric_names)
    ax.set_ylabel("Score"); ax.set_title("Model Performance Comparison")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.7)

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force re-training even if checkpoints exist"
    )
    args = parser.parse_args()

    if args.retrain:
        print("--retrain flag set: ignoring existing checkpoints, training from scratch.")
    else:
        print("Checkpoint mode: will skip training if saved weights are found.")

    # --- Load data ---
    train_texts, val_texts, train_labels, val_labels, test_df = load_data()

    # --- Shared tokenizer (use_fast=True: 10-20x faster than use_fast=False) ---
    print(f"\nLoading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    print("  ✓ Tokenizer loaded.\n")

    # ── Experiment A: Fine-tuned BERTweet ────────────────────────────────────
    print("Loading pre-trained BERTweet model...")
    model_a = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)
    num_params = sum(p.numel() for p in model_a.parameters()) / 1e6
    print(f"  ✓ Model A loaded ({num_params:.1f}M parameters).")

    history_a, preds_a, labels_a = run_experiment(
        model_a, tokenizer,
        train_texts, val_texts, train_labels, val_labels,
        "Experiment A: Fine-tuned BERTweet (Pre-trained)",
        ckpt_dir=CKPT_A,
        retrain=args.retrain
    )

    # Free GPU memory before loading model B
    del model_a
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ── Experiment B: BERTweet from Scratch ──────────────────────────────────
    print("\nInitializing BERTweet from scratch (random weights)...")
    config  = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model_b = RobertaForSequenceClassification(config).to(DEVICE)
    num_params = sum(p.numel() for p in model_b.parameters()) / 1e6
    print(f"  ✓ Model B initialized ({num_params:.1f}M parameters, random weights).")

    history_b, preds_b, labels_b = run_experiment(
        model_b, tokenizer,
        train_texts, val_texts, train_labels, val_labels,
        "Experiment B: BERTweet from Scratch",
        ckpt_dir=CKPT_B,
        retrain=args.retrain
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    def compute_metrics(true, pred):
        return {
            "Accuracy":  accuracy_score(true, pred),
            "F1 Score":  f1_score(true, pred, average="weighted"),
            "Precision": precision_score(true, pred, average="weighted", zero_division=0),
            "Recall":    recall_score(true, pred, average="weighted", zero_division=0),
        }

    metrics_a = compute_metrics(labels_a, preds_a)
    metrics_b = compute_metrics(labels_b, preds_b)

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<12} {'Pre-trained':>14} {'From Scratch':>14}")
    print("-"*42)
    for m in ["Accuracy", "F1 Score", "Precision", "Recall"]:
        print(f"{m:<12} {metrics_a[m]:>14.4f} {metrics_b[m]:>14.4f}")

    print("\nPre-trained BERTweet — Classification Report:")
    print(classification_report(labels_a, preds_a, target_names=["Non-Disaster", "Disaster"]))

    print("BERTweet from Scratch — Classification Report:")
    print(classification_report(labels_b, preds_b, target_names=["Non-Disaster", "Disaster"]))

    # ── Save metrics CSV ──────────────────────────────────────────────────────
    results_df = pd.DataFrame({
        "Metric":       list(metrics_a.keys()),
        "Pre-trained":  list(metrics_a.values()),
        "From Scratch": list(metrics_b.values()),
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
    print(f"\nMetrics saved to {RESULTS_DIR}/metrics_comparison.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(
        history_a, history_b,
        os.path.join(RESULTS_DIR, "training_curves.png")
    )
    plot_confusion_matrix(
        labels_a, preds_a,
        "Confusion Matrix — Pre-trained BERTweet",
        os.path.join(RESULTS_DIR, "cm_pretrained.png")
    )
    plot_confusion_matrix(
        labels_b, preds_b,
        "Confusion Matrix — BERTweet from Scratch",
        os.path.join(RESULTS_DIR, "cm_scratch.png")
    )
    plot_metrics_comparison(
        metrics_a, metrics_b,
        os.path.join(RESULTS_DIR, "metrics_comparison.png")
    )

    print(f"\nAll results saved to: {RESULTS_DIR}/")


if __name__ == "__main__":
    import sys
    # Fix for Jupyter/Colab: argparse conflicts with kernel launcher arguments
    is_notebook = hasattr(sys, 'ps1') or 'ipykernel' in sys.modules
    if is_notebook:
        # In Colab/Jupyter: set retrain manually here (True = force retrain)
        class _Args:
            retrain = False   # ← change to True to force re-training
        args = _Args()
        if args.retrain:
            print("Retrain mode: ignoring existing checkpoints.")
        else:
            print("Checkpoint mode: will skip training if saved weights are found.")

        train_texts, val_texts, train_labels, val_labels, test_df = load_data()

        print(f"\nLoading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
        print("  ✓ Tokenizer loaded.\n")

        print("Loading pre-trained BERTweet model...")
        model_a = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2
        ).to(DEVICE)
        num_params = sum(p.numel() for p in model_a.parameters()) / 1e6
        print(f"  ✓ Model A loaded ({num_params:.1f}M parameters).")

        history_a, preds_a, labels_a = run_experiment(
            model_a, tokenizer,
            train_texts, val_texts, train_labels, val_labels,
            "Experiment A: Fine-tuned BERTweet (Pre-trained)",
            ckpt_dir=CKPT_A,
            retrain=args.retrain
        )

        del model_a
        torch.cuda.empty_cache()

        print("\nInitializing BERTweet from scratch (random weights)...")
        config  = RobertaConfig.from_pretrained(MODEL_NAME, num_labels=2)
        model_b = RobertaForSequenceClassification(config).to(DEVICE)
        num_params = sum(p.numel() for p in model_b.parameters()) / 1e6
        print(f"  ✓ Model B initialized ({num_params:.1f}M parameters, random weights).")

        history_b, preds_b, labels_b = run_experiment(
            model_b, tokenizer,
            train_texts, val_texts, train_labels, val_labels,
            "Experiment B: BERTweet from Scratch",
            ckpt_dir=CKPT_B,
            retrain=args.retrain
        )

        def compute_metrics(true, pred):
            return {
                "Accuracy":  accuracy_score(true, pred),
                "F1 Score":  f1_score(true, pred, average="weighted"),
                "Precision": precision_score(true, pred, average="weighted", zero_division=0),
                "Recall":    recall_score(true, pred, average="weighted", zero_division=0),
            }

        metrics_a = compute_metrics(labels_a, preds_a)
        metrics_b = compute_metrics(labels_b, preds_b)

        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"{'Metric':<12} {'Pre-trained':>14} {'From Scratch':>14}")
        print("-"*42)
        for m in ["Accuracy", "F1 Score", "Precision", "Recall"]:
            print(f"{m:<12} {metrics_a[m]:>14.4f} {metrics_b[m]:>14.4f}")

        print("\nPre-trained BERTweet — Classification Report:")
        print(classification_report(labels_a, preds_a, target_names=["Non-Disaster", "Disaster"]))
        print("BERTweet from Scratch — Classification Report:")
        print(classification_report(labels_b, preds_b, target_names=["Non-Disaster", "Disaster"]))

        results_df = pd.DataFrame({
            "Metric":       list(metrics_a.keys()),
            "Pre-trained":  list(metrics_a.values()),
            "From Scratch": list(metrics_b.values()),
        })
        results_df.to_csv(os.path.join(RESULTS_DIR, "metrics_comparison.csv"), index=False)
        print(f"\nMetrics saved to {RESULTS_DIR}/metrics_comparison.csv")

        plot_training_curves(history_a, history_b,
            os.path.join(RESULTS_DIR, "training_curves.png"))
        plot_confusion_matrix(labels_a, preds_a,
            "Confusion Matrix — Pre-trained BERTweet",
            os.path.join(RESULTS_DIR, "cm_pretrained.png"))
        plot_confusion_matrix(labels_b, preds_b,
            "Confusion Matrix — BERTweet from Scratch",
            os.path.join(RESULTS_DIR, "cm_scratch.png"))
        plot_metrics_comparison(metrics_a, metrics_b,
            os.path.join(RESULTS_DIR, "metrics_comparison.png"))

        print(f"\nAll results saved to: {RESULTS_DIR}/")
    else:
        main()