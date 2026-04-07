
"""
experiment1_best_two_models_screen.py
Experiment 1 - Supervised Learning

Final simplified version that keeps only:
1. Best CNN by Test F1  -> CNN_nf64_k3_cb1
2. Best BiLSTM by Test F1 -> BiLSTM_stack64

Behavior:
- shows training graphs on screen
- shows confusion matrices on screen
- prints reports and summary in terminal
- does NOT save files
"""

import os
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers, models, regularizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight

from preprocess import load_dataset

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 76
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
DEFAULT_TRAIN = os.path.normpath(
    os.path.join(BASE_DIR, "..", "data", "train.csv")
)


# -----------------------------
# Text vectorizer wrapper
# -----------------------------
class TextVectorizerWrapper:
    """Wraps Keras TextVectorization so vocab/config is reusable across models."""

    def __init__(self, max_tokens: int = 20000, seq_len: int = 60):
        self.max_tokens = max_tokens
        self.seq_len = seq_len
        self.vectorizer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=seq_len,
        )

    def adapt(self, texts):
        self.vectorizer.adapt(np.array(texts))

    def __call__(self, texts):
        return self.vectorizer(np.array(texts))

    @property
    def vocab_size(self) -> int:
        return len(self.vectorizer.get_vocabulary())


# -----------------------------
# Data helpers
# -----------------------------
def find_text_and_target_columns(df) -> Tuple[str, str]:
    """Detect likely text and target columns."""
    possible_text_cols = ["text", "clean_text", "processed_text", "tweet", "tweet_text"]
    possible_target_cols = ["target", "label", "labels", "y"]

    text_col = next((c for c in possible_text_cols if c in df.columns), None)
    target_col = next((c for c in possible_target_cols if c in df.columns), None)

    if text_col is None or target_col is None:
        raise ValueError(
            f"Could not detect text/target columns automatically.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Expected text columns: {possible_text_cols}\n"
            f"Expected target columns: {possible_target_cols}"
        )

    return text_col, target_col


def make_train_val_test_split(df, test_size=0.15, val_size=0.15):
    """Create stratified train / validation / test split."""
    text_col, target_col = find_text_and_target_columns(df)

    X = df[text_col].fillna("").astype(str).values
    y = df[target_col].astype(int).values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=SEED,
        stratify=y,
    )

    val_relative = val_size / (1.0 - test_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=SEED,
        stratify=y_trainval,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_class_weights(y_train) -> Dict[int, float]:
    """Compute class weights for imbalance handling."""
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# -----------------------------
# Model builders
# -----------------------------
def build_best_cnn(
    vocab_size: int,
    seq_len: int,
    embed_dim: int = 128,
    num_filters: int = 64,
    kernel_size: int = 3,
    dense_units: int = 64,
    dropout: float = 0.4,
    l2_value: float = 1e-4,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Best CNN by Test F1:
    CNN_nf64_k3_cb1
    """
    inp = layers.Input(shape=(seq_len,), dtype="int64")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=False)(inp)

    x = layers.Conv1D(
        filters=num_filters,
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_value),
    )(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_value),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="Best_Disaster_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_best_bilstm(
    vocab_size: int,
    seq_len: int,
    embed_dim: int = 128,
    rnn_units: int = 64,
    dense_units: int = 64,
    dropout: float = 0.5,
    l2_value: float = 1e-4,
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Best BiLSTM by Test F1:
    BiLSTM_stack64
    """
    inp = layers.Input(shape=(seq_len,), dtype="int64")
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(inp)
    x = layers.SpatialDropout1D(0.2)(x)

    x = layers.Bidirectional(
        layers.LSTM(
            rnn_units,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.0
        )
    )(x)

    x = layers.Bidirectional(
        layers.LSTM(
            rnn_units // 2,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.0
        )
    )(x)

    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_value),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)

    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out, name="Best_Disaster_BiLSTM")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# -----------------------------
# Plotting
# -----------------------------
def plot_history(history, name: str):
    """Show training curves on screen."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(history.history["loss"], label="train")
    ax[0].plot(history.history["val_loss"], label="val")
    ax[0].set_title(f"{name} - Loss")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()

    ax[1].plot(history.history["accuracy"], label="train")
    ax[1].plot(history.history["val_accuracy"], label="val")
    ax[1].set_title(f"{name} - Accuracy")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()

    fig.tight_layout()
    plt.show()
    plt.close(fig)


def show_conf_matrix(y_true, y_pred, name: str, split_name: str):
    """Show confusion matrix on screen."""
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} - {split_name} Confusion Matrix")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    return cm.tolist()


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_split(model, X_split, y_split, name: str, split_name: str):
    """Evaluate model on one split and show confusion matrix."""
    probs = model.predict(X_split, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_split, preds)
    f1 = f1_score(y_split, preds)
    report = classification_report(y_split, preds, digits=4)
    cm = confusion_matrix(y_split, preds)

    print(f"\n--- {name} | {split_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    show_conf_matrix(y_split, preds, name, split_name)

    return {
        f"{split_name.lower()}_accuracy": float(acc),
        f"{split_name.lower()}_f1": float(f1),
        f"{split_name.lower()}_cm": cm.tolist(),
    }


def train_and_evaluate(
    model: tf.keras.Model,
    name: str,
    X_train_v,
    y_train,
    X_val_v,
    y_val,
    X_test_v,
    y_test,
    epochs: int,
    batch_size: int,
    class_weight: Dict[int, float],
):
    """Train, show plots, and print evaluation in terminal."""
    print(f"\n{'=' * 20} Training {name} {'=' * 20}")
    model.summary()

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    rlrop = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-5,
        verbose=1,
    )

    history = model.fit(
        X_train_v,
        y_train,
        validation_data=(X_val_v, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, rlrop],
        class_weight=class_weight,
        verbose=2,
    )

    plot_history(history, name)

    val_metrics = evaluate_split(model, X_val_v, y_val, name, "validation")
    test_metrics = evaluate_split(model, X_test_v, y_test, name, "test")

    all_metrics = {"model": name}
    all_metrics.update(val_metrics)
    all_metrics.update(test_metrics)

    return all_metrics


# -----------------------------
# Main run
# -----------------------------
def run(train_csv: str, epochs: int, batch_size: int):
    """Run Experiment 1 with only the best CNN and best BiLSTM."""
    train_df, _ = load_dataset(train_csv)

    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(train_df)

    print("Dataset split sizes:")
    print(f"Train: {len(X_train)}")
    print(f"Validation: {len(X_val)}")
    print(f"Test: {len(X_test)}")

    vec = TextVectorizerWrapper(max_tokens=20000, seq_len=60)
    vec.adapt(X_train)

    X_train_v = vec(X_train)
    X_val_v = vec(X_val)
    X_test_v = vec(X_test)

    print(f"Vocabulary size: {vec.vocab_size}")
    print(f"Sequence length: {vec.seq_len}")

    class_weight = get_class_weights(y_train)
    print("Class weights:", class_weight)

    results: List[Dict] = []

    # Best CNN
    best_cnn = build_best_cnn(
        vocab_size=vec.vocab_size,
        seq_len=vec.seq_len,
        embed_dim=128,
        num_filters=64,
        kernel_size=3,
        dense_units=64,
        dropout=0.4,
        l2_value=1e-4,
        learning_rate=1e-3,
    )

    cnn_metrics = train_and_evaluate(
        model=best_cnn,
        name="CNN_nf64_k3_cb1_best",
        X_train_v=X_train_v,
        y_train=y_train,
        X_val_v=X_val_v,
        y_val=y_val,
        X_test_v=X_test_v,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
    )
    results.append(cnn_metrics)

    # Best BiLSTM
    best_bilstm = build_best_bilstm(
        vocab_size=vec.vocab_size,
        seq_len=vec.seq_len,
        embed_dim=128,
        rnn_units=64,
        dense_units=64,
        dropout=0.5,
        l2_value=1e-4,
        learning_rate=1e-3,
    )

    bilstm_metrics = train_and_evaluate(
        model=best_bilstm,
        name="BiLSTM_stack64_best",
        X_train_v=X_train_v,
        y_train=y_train,
        X_val_v=X_val_v,
        y_val=y_val,
        X_test_v=X_test_v,
        y_test=y_test,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
    )
    results.append(bilstm_metrics)

    # Sort by Test F1
    results_sorted = sorted(results, key=lambda x: x["test_f1"], reverse=True)

    print("\n" + "=" * 70)
    print("Experiment 1 Summary (Best CNN + Best BiLSTM)")
    print("=" * 70)
    for r in results_sorted:
        print(
            f"{r['model']:<25} | "
            f"Val Acc: {r['validation_accuracy']:.4f} | "
            f"Val F1: {r['validation_f1']:.4f} | "
            f"Test Acc: {r['test_accuracy']:.4f} | "
            f"Test F1: {r['test_f1']:.4f}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=DEFAULT_TRAIN)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    run(args.train, args.epochs, args.batch_size)