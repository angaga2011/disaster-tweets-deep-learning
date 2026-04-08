"""
Dataset loading and lightweight text preprocessing utilities.

Expected Kaggle files:
- data/train.csv
- data/test.csv (optional for training script)
"""

import os
import re
from typing import Optional, Tuple

import pandas as pd


def _clean_tweet_text(text: str) -> str:
    """Basic normalization for tweet text."""
    text = str(text or "")
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ensure_train_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize train dataframe columns."""
    required = {"id", "text", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Train CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["processed_text"] = out["text"].apply(_clean_tweet_text)
    out["target"] = out["target"].astype(int)
    return out


def _ensure_test_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize test dataframe columns."""
    required = {"id", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Test CSV is missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["processed_text"] = out["text"].apply(_clean_tweet_text)
    return out


def load_dataset(train_csv_path: str, test_csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Kaggle Disaster Tweets CSVs and return normalized dataframes.

    Returns:
        (train_df, test_df)
    """
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(
            f"Train CSV not found at: {train_csv_path}\n"
            "Download Kaggle dataset and place train.csv under data/."
        )

    train_df = pd.read_csv(train_csv_path)
    train_df = _ensure_train_columns(train_df)

    if test_csv_path is None:
        test_csv_path = os.path.join(os.path.dirname(train_csv_path), "test.csv")

    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        test_df = _ensure_test_columns(test_df)
    else:
        # Keep contract consistent even when test.csv is absent.
        test_df = pd.DataFrame(columns=["id", "text", "processed_text"])

    return train_df, test_df
