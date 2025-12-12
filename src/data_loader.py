from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from .config import DATA_FILE


def load_raw_dataframe(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw DataFrame from the specified file path (CSV or Excel)."""
    path = file_path or DATA_FILE
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read Excel or CSV based on file extension
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()
    return df


def extract_comment_and_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Extract 'Text' and 'Transferred Chat' columns for model training.

    'Text' is renamed to 'comment' (input feature).
    'Transferred Chat' is renamed to 'rating' (target label).
    """

    # Define columns
    text_col = "Text"
    target_col = "Transferred Chat"

    # Check for required columns
    if text_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Required columns missing. Current columns: {df.columns.tolist()}")

    # Extract data and rename columns
    subset = df[[text_col, target_col]].copy()
    subset.columns = ["comment", "rating"]

    # Handle missing values and ensure comment is string
    subset = subset.dropna(subset=["comment"])
    subset["comment"] = subset["comment"].astype(str)

    # Map target values: True -> 'transfer', False -> 'done'
    # 'transfer' indicates the chat was escalated/transferred.
    subset["rating"] = subset["rating"].apply(
        lambda x: "transfer" if str(x).lower() == "true" else "done"
    )

    print(f"Data loaded successfully! (Target distribution):\n{subset['rating'].value_counts()}")

    return subset