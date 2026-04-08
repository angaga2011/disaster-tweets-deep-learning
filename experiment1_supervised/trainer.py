"""Training orchestration for per-config model runs."""

from __future__ import annotations

import os
from typing import Dict

import tensorflow as tf
from tensorflow.keras import callbacks

from config import ExperimentConfig
from evaluator import evaluate_predictions, predict_binary
from models import build_model
from plotting import save_confusion_matrix, save_training_curves
from utils import write_json, write_text


def _build_callbacks() -> list:
    return [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=1,
            min_lr=1e-5,
            verbose=1,
        ),
    ]


def run_single_experiment(
    cfg: ExperimentConfig,
    vectorized_data: Dict,
    meta: Dict,
    epochs: int,
    batch_size: int,
    out_dirs: Dict[str, str],
) -> Dict:
    x_train = vectorized_data["x_train"]
    x_val = vectorized_data["x_val"]
    x_test = vectorized_data["x_test"]
    y_train = vectorized_data["y_train"]
    y_val = vectorized_data["y_val"]
    y_test = vectorized_data["y_test"]
    class_weight = vectorized_data["class_weight"]
    vocab_size = vectorized_data["vocab_size"]
    seq_len = vectorized_data["seq_len"]

    model = build_model(cfg, vocab_size=vocab_size, seq_len=seq_len)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=_build_callbacks(),
        verbose=2,
    )

    _, val_preds = predict_binary(model, x_val)
    _, test_preds = predict_binary(model, x_test)
    val_eval = evaluate_predictions(y_val, val_preds)
    test_eval = evaluate_predictions(y_test, test_preds)

    save_training_curves(
        history,
        cfg.name,
        os.path.join(out_dirs["plots"], f"{cfg.name}_training_curves.png"),
    )
    save_confusion_matrix(
        val_eval["confusion_matrix"],
        f"{cfg.name} - Validation Confusion Matrix",
        os.path.join(out_dirs["confusion_matrices"], f"{cfg.name}_validation_cm.png"),
    )
    save_confusion_matrix(
        test_eval["confusion_matrix"],
        f"{cfg.name} - Test Confusion Matrix",
        os.path.join(out_dirs["confusion_matrices"], f"{cfg.name}_test_cm.png"),
    )

    result = {
        "name": cfg.name,
        "family": cfg.family,
        "config": cfg.to_dict(),
        "validation_accuracy": val_eval["accuracy"],
        "validation_precision": val_eval["precision"],
        "validation_recall": val_eval["recall"],
        "validation_f1": val_eval["f1"],
        "test_accuracy": test_eval["accuracy"],
        "test_precision": test_eval["precision"],
        "test_recall": test_eval["recall"],
        "test_f1": test_eval["f1"],
        "validation_confusion_matrix": val_eval["confusion_matrix"],
        "test_confusion_matrix": test_eval["confusion_matrix"],
        "validation_classification_report": val_eval["classification_report"],
        "test_classification_report": test_eval["classification_report"],
        "history": history.history,
        "epochs_ran": len(history.history.get("loss", [])),
        "meta": meta,
    }

    write_json(os.path.join(out_dirs["metrics"], f"{cfg.name}_metrics.json"), result)
    run_txt = "\n".join(
        [
            f"Experiment: {cfg.name}",
            f"Family: {cfg.family}",
            f"Validation Accuracy: {result['validation_accuracy']:.4f}",
            f"Validation Precision: {result['validation_precision']:.4f}",
            f"Validation Recall: {result['validation_recall']:.4f}",
            f"Validation F1: {result['validation_f1']:.4f}",
            f"Test Accuracy: {result['test_accuracy']:.4f}",
            f"Test Precision: {result['test_precision']:.4f}",
            f"Test Recall: {result['test_recall']:.4f}",
            f"Test F1: {result['test_f1']:.4f}",
            "",
            "Validation classification report:",
            result["validation_classification_report"],
            "Test classification report:",
            result["test_classification_report"],
        ]
    )
    write_text(os.path.join(out_dirs["summaries"], f"{cfg.name}_summary.txt"), run_txt)
    tf.keras.backend.clear_session()
    return result
