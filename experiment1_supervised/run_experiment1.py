"""Main runner for Experiment 1 supervised learning pipeline."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

# Robust pathing: allow importing root-level preprocess.py when run from any cwd.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from preprocess import load_dataset  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEQ_LEN,
    DEFAULT_TEST_SIZE,
    DEFAULT_VAL_SIZE,
    SEED,
    build_experiment_grid,
    resolve_default_train_path,
)
from reporting import write_summary_artifacts  # noqa: E402
from trainer import run_single_experiment  # noqa: E402
from utils import make_output_dirs, set_global_seed, sync_run_to_latest, write_text  # noqa: E402


class TeeStream:
    """Write stream output to terminal and file simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def find_text_and_target_columns(df) -> Tuple[str, str]:
    text_candidates = ["processed_text", "text", "clean_text", "tweet", "tweet_text"]
    target_candidates = ["target", "label", "labels", "y"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if text_col is None or target_col is None:
        raise ValueError(
            f"Could not detect text/target columns. Found columns: {list(df.columns)}"
        )
    return text_col, target_col


def make_train_val_test_split(df, test_size=DEFAULT_TEST_SIZE, val_size=DEFAULT_VAL_SIZE):
    text_col, target_col = find_text_and_target_columns(df)
    x_all = df[text_col].fillna("").astype(str).values
    y_all = df[target_col].astype(int).values

    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x_all, y_all, test_size=test_size, random_state=SEED, stratify=y_all
    )
    val_relative = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        test_size=val_relative,
        random_state=SEED,
        stratify=y_trainval,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


class TextVectorizerWrapper:
    """Keras TextVectorization wrapper for reusable config."""

    def __init__(self, max_tokens=DEFAULT_MAX_TOKENS, seq_len=DEFAULT_SEQ_LEN):
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
    def vocab_size(self):
        return len(self.vectorizer.get_vocabulary())


def get_class_weights(y_train) -> Dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def build_cli() -> argparse.Namespace:
    default_train = resolve_default_train_path(REPO_ROOT)
    default_output = os.path.join(THIS_DIR, "outputs")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=default_train)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_experiments", type=int, default=0)
    parser.add_argument("--output_dir", default=default_output)
    parser.add_argument("--quick_mode", action="store_true")
    return parser.parse_args()


def select_experiments(max_experiments: int, quick_mode: bool):
    grid = build_experiment_grid()
    if quick_mode:
        grid = [cfg for cfg in grid if cfg.family == "cnn"][:2] + [
            cfg for cfg in grid if cfg.family == "bilstm"
        ][:2]
    if max_experiments and max_experiments > 0:
        grid = grid[:max_experiments]
    return grid


def main():
    args = build_cli()
    set_global_seed(SEED)

    paths = make_output_dirs(args.output_dir)
    run_log = os.path.join(paths["run_logs"], "run.log")

    with open(run_log, "w", encoding="utf-8") as logf:
        tee_out = TeeStream(sys.__stdout__, logf)
        tee_err = TeeStream(sys.__stderr__, logf)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print("Running Experiment 1 supervised pipeline")
            print(f"Train CSV: {args.train}")
            print(f"Seed: {SEED}")
            print(f"Output archive run dir: {paths['run_root']}")

            train_df, _ = load_dataset(args.train)
            x_train, x_val, x_test, y_train, y_val, y_test = make_train_val_test_split(
                train_df
            )
            vec = TextVectorizerWrapper()
            vec.adapt(x_train)

            vectorized = {
                "x_train": vec(x_train),
                "x_val": vec(x_val),
                "x_test": vec(x_test),
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test,
                "class_weight": get_class_weights(y_train),
                "vocab_size": vec.vocab_size,
                "seq_len": vec.seq_len,
            }

            split_sizes = {
                "train": len(x_train),
                "validation": len(x_val),
                "test": len(x_test),
            }
            print(f"Split sizes: {split_sizes}")
            print(f"Vocabulary size: {vec.vocab_size}")
            print(f"Sequence length: {vec.seq_len}")
            print(f"Class weights: {vectorized['class_weight']}")

            experiments = select_experiments(args.max_experiments, args.quick_mode)
            print(f"Number of experiments: {len(experiments)}")

            results: List[Dict] = []
            for idx, cfg in enumerate(experiments, start=1):
                print("\n" + "=" * 88)
                print(f"[{idx}/{len(experiments)}] Training {cfg.name} ({cfg.family})")
                print(f"Config: {cfg.to_dict()}")
                run_result = run_single_experiment(
                    cfg=cfg,
                    vectorized_data=vectorized,
                    meta={"split_sizes": split_sizes, "vocab_size": vec.vocab_size, "seq_len": vec.seq_len},
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    out_dirs={
                        "plots": paths["run_plots"],
                        "confusion_matrices": paths["run_confusion_matrices"],
                        "metrics": paths["run_metrics"],
                        "summaries": paths["run_summaries"],
                    },
                )
                results.append(run_result)
                print(
                    "Done:"
                    f" val_f1={run_result['validation_f1']:.4f},"
                    f" test_f1={run_result['test_f1']:.4f},"
                    f" test_acc={run_result['test_accuracy']:.4f}"
                )

            write_summary_artifacts(
                results=results,
                meta={"split_sizes": split_sizes, "vocab_size": vec.vocab_size, "seq_len": vec.seq_len},
                summary_dir=paths["run_summary"],
                plots_dir=paths["run_plots"],
            )
            print("\nSummary artifacts generated.")

    sync_run_to_latest(paths)
    write_text(
        os.path.join(paths["latest_logs"], "run_location.txt"),
        f"Archive run path: {paths['run_root']}\n",
    )


if __name__ == "__main__":
    main()
