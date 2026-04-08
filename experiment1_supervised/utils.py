"""Shared utility helpers for Experiment 1."""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from config import timestamp_for_run


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def repo_root_from_here() -> str:
    return str(Path(__file__).resolve().parent.parent)


def make_output_dirs(base_output_dir: str) -> Dict[str, str]:
    """
    Create archive run directory and refresh outputs/latest.

    Preserves historical runs in archive while latest points to current run.
    """
    base = Path(base_output_dir)
    latest = base / "latest"
    archive = base / "archive"
    run_stamp = timestamp_for_run()
    run_dir = archive / run_stamp

    run_subdirs = {
        "root": run_dir,
        "plots": run_dir / "plots",
        "confusion_matrices": run_dir / "confusion_matrices",
        "metrics": run_dir / "metrics",
        "summaries": run_dir / "summaries",
        "summary": run_dir / "summary",
        "logs": run_dir / "logs",
    }

    for p in [archive, *run_subdirs.values()]:
        p.mkdir(parents=True, exist_ok=True)

    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_dir, latest)

    latest_subdirs = {k: latest / v.name for k, v in run_subdirs.items()}
    latest_subdirs["root"] = latest

    return {
        "base": str(base),
        "archive_root": str(archive),
        "latest_root": str(latest),
        "run_stamp": run_stamp,
        "run_root": str(run_subdirs["root"]),
        "run_plots": str(run_subdirs["plots"]),
        "run_confusion_matrices": str(run_subdirs["confusion_matrices"]),
        "run_metrics": str(run_subdirs["metrics"]),
        "run_summaries": str(run_subdirs["summaries"]),
        "run_summary": str(run_subdirs["summary"]),
        "run_logs": str(run_subdirs["logs"]),
        "latest_plots": str(latest_subdirs["plots"]),
        "latest_confusion_matrices": str(latest_subdirs["confusion_matrices"]),
        "latest_metrics": str(latest_subdirs["metrics"]),
        "latest_summaries": str(latest_subdirs["summaries"]),
        "latest_summary": str(latest_subdirs["summary"]),
        "latest_logs": str(latest_subdirs["logs"]),
    }


def sync_run_to_latest(paths: Dict[str, str]) -> None:
    """Refresh outputs/latest from current archive run directory."""
    latest = Path(paths["latest_root"])
    run_root = Path(paths["run_root"])
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_root, latest)


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=to_serializable)


def write_text(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
