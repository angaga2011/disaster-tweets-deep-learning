"""Configuration for Experiment 1 supervised pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List


SEED = 42
DEFAULT_MAX_TOKENS = 20000
DEFAULT_SEQ_LEN = 60
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15
DEFAULT_LR = 1e-3


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    family: str  # cnn | bilstm
    embed_dim: int
    dense_units: int
    dropout: float
    l2_value: float
    learning_rate: float
    batch_norm: bool
    # CNN fields
    filters: int = 0
    kernel_size: int = 0
    # BiLSTM fields
    lstm_units: int = 0
    stacked: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


def resolve_default_train_path(repo_root: str) -> str:
    """Pick the first available train.csv from known locations."""
    candidates = [
        os.path.join(repo_root, "data", "train.csv"),
        os.path.join(repo_root, "nlp-getting-started", "train.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.normpath(path)
    return os.path.normpath(candidates[0])


def build_experiment_grid() -> List[ExperimentConfig]:
    """Curated set of 10 experiments across CNN and BiLSTM families."""
    return [
        # CNN variants
        ExperimentConfig(
            name="cnn_f32_k3_d32_do03_l2e4_bn",
            family="cnn",
            embed_dim=128,
            filters=32,
            kernel_size=3,
            dense_units=32,
            dropout=0.3,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        ExperimentConfig(
            name="cnn_f32_k5_d64_do05_l2e4_bn",
            family="cnn",
            embed_dim=128,
            filters=32,
            kernel_size=5,
            dense_units=64,
            dropout=0.5,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        ExperimentConfig(
            name="cnn_f64_k3_d64_do03_l2e4_bn",
            family="cnn",
            embed_dim=128,
            filters=64,
            kernel_size=3,
            dense_units=64,
            dropout=0.3,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        ExperimentConfig(
            name="cnn_f64_k5_d32_do05_l2e4_no_bn",
            family="cnn",
            embed_dim=128,
            filters=64,
            kernel_size=5,
            dense_units=32,
            dropout=0.5,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=False,
        ),
        ExperimentConfig(
            name="cnn_f64_k3_d64_do05_l2e3_bn",
            family="cnn",
            embed_dim=128,
            filters=64,
            kernel_size=3,
            dense_units=64,
            dropout=0.5,
            l2_value=1e-3,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        # BiLSTM variants
        ExperimentConfig(
            name="bilstm_u32_single_d32_do03_l2e4_bn",
            family="bilstm",
            embed_dim=128,
            lstm_units=32,
            stacked=False,
            dense_units=32,
            dropout=0.3,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        ExperimentConfig(
            name="bilstm_u32_stack_d64_do05_l2e4_bn",
            family="bilstm",
            embed_dim=128,
            lstm_units=32,
            stacked=True,
            dense_units=64,
            dropout=0.5,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        ExperimentConfig(
            name="bilstm_u64_single_d64_do03_l2e4_no_bn",
            family="bilstm",
            embed_dim=128,
            lstm_units=64,
            stacked=False,
            dense_units=64,
            dropout=0.3,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=False,
        ),
        ExperimentConfig(
            name="bilstm_u64_stack_d64_do05_l2e4_bn",
            family="bilstm",
            embed_dim=128,
            lstm_units=64,
            stacked=True,
            dense_units=64,
            dropout=0.5,
            l2_value=1e-4,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
        ExperimentConfig(
            name="bilstm_u64_stack_d32_do03_l2e3_bn",
            family="bilstm",
            embed_dim=128,
            lstm_units=64,
            stacked=True,
            dense_units=32,
            dropout=0.3,
            l2_value=1e-3,
            learning_rate=DEFAULT_LR,
            batch_norm=True,
        ),
    ]


def timestamp_for_run() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
