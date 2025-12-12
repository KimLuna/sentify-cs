from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from .config import DATA_FILE


def load_raw_dataframe(file_path: Optional[Path] = None) -> pd.DataFrame:
    """Load the raw chat dataset from disk.

    The loader automatically handles CSV and Excel files.

    Parameters
    ----------
    file_path: Optional[Path]
        Optionally override the default dataset location. If omitted, `DATA_FILE` from `config.py` is used.

    Returns
    -------
    pandas.DataFrame
        The raw DataFrame loaded from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    path = file_path or DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. Please place your data file in the data directory"
        )

    # Determine loader based on extension
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df


def extract_comment_and_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame containing only the comment and rating columns.

    This helper simplifies downstream processing by selecting the
    relevant columns and renaming them to canonical names.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame returned by `load_raw_dataframe`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with two columns: `comment` and `rating`. Rows
        with missing or empty comments are dropped.
    """
    # Some datasets may use different header names; handle common variations
    comment_col_candidates = [
        "Customer Comment",
        "Text",
        "CustomerComment",
    ]
    rating_col_candidates = [
        "Customer Rating",
        "Rating",
        "CustomerRating",
    ]

    # Ensure comment and rating columns exist
    comment_col = next((c for c in comment_col_candidates if c in df.columns), None)
    rating_col = next((c for c in rating_col_candidates if c in df.columns), None)

    if comment_col is None or rating_col is None:
        raise ValueError(
            "Could not find the required comment and rating columns in the dataset."
        )

    # Select the relevant columns and rename them
    subset = df[[comment_col, rating_col]].copy()
    subset.columns = ["comment", "rating"]

    # Drop rows without comments
    subset = subset.dropna(subset=["comment"])

    # Ensure 'comment' is of string type
    subset["comment"] = subset["comment"].astype(str)

    # Optional: Further clean rating values (e.g., handle incorrect or missing ratings)
    return subset
