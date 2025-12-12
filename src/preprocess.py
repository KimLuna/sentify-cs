from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD


# Compile a regex pattern to remove punctuation and digits
_punct_num_pattern = re.compile(r"[^a-zA-Z\s]")


def clean_comment(text: str, extra_stopwords: Optional[Iterable[str]] = None) -> str:
    """Clean a single customer comment.

    Steps performed:

    1. Convert to lower case.
    2. Remove punctuation, numbers and other non‑letter characters.
    3. Collapse multiple whitespace characters into a single space.
    4. Remove stop words.

    Parameters
    ----------
    text: str
        The raw comment text.
    extra_stopwords: Iterable[str] | None
        Additional stop words to remove. Can be useful for domain-specific words such as 'please', 'agent', etc.

    Returns
    -------
    str
        The cleaned comment.
    """
    if not isinstance(text, str):
        text = str(text)
    # Lower case
    text = text.lower()
    # Remove punctuation and numbers
    text = _punct_num_pattern.sub(" ", text)
    # Remove extra whitespace
    words = text.split()
    stopwords = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stopwords.update([w.lower() for w in extra_stopwords])
    # Filter stop words
    cleaned_words: List[str] = [w for w in words if w not in stopwords]
    return " ".join(cleaned_words)


def map_rating_to_label(rating: Optional[float | int | str | None]) -> Optional[str]:
    """Map a numeric rating to a sentiment label.

    Ratings greater than or equal to `POSITIVE_THRESHOLD` are mapped
    to "positive". Ratings less than or equal to `NEGATIVE_THRESHOLD`
    are mapped to "negative". Ratings between the two thresholds 
    (exclusive) return `None`, indicating a neutral label. 
    Neutral entries can be discarded or handled separately.

    Parameters
    ----------
    rating: Optional[float | int | str | None]
        The raw rating value from the dataset.

    Returns
    -------
    str | None
        `positive` if rating >= POSITIVE_THRESHOLD, `negative` if rating <= NEGATIVE_THRESHOLD, 
        otherwise `None` for neutral.
    """
    if rating is None:
        return None

    if isinstance(rating, str):
        rating = rating.strip()
        if rating == "":
            return None

    try:
        rating_val = float(rating)
    except (TypeError, ValueError):
        return None

    if rating_val >= POSITIVE_THRESHOLD:
        return "positive"
    if rating_val <= NEGATIVE_THRESHOLD:
        return "negative"
    return None


def preprocess_dataframe(df: pd.DataFrame, extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Clean comments and map ratings to labels for the entire DataFrame.

    Parameters
    ----------
    df: pandas.DataFrame
        A DataFrame with columns `comment` and `rating`.
    extra_stopwords: Iterable[str] | None
        Additional stop words to remove during cleaning.

    Returns
    -------
    pandas.DataFrame
        DataFrame with three columns: `comment`, `rating` and `label`. 
        Rows with neutral labels are dropped.
    """
    # Use 'apply' for more efficient processing
    df['label'] = df['rating'].apply(map_rating_to_label)
    df = df.dropna(subset=['label'])  # Remove rows with neutral labels

    # Clean the comments
    df['comment'] = df['comment'].apply(clean_comment, extra_stopwords=extra_stopwords)
    
    # Drop empty comments
    df = df[df['comment'].str.strip() != '']

    return df[['comment', 'label']]
