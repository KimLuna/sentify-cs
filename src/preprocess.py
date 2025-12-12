from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    # Import thresholds when run as a package
    from .config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD
except ImportError:
    # Import thresholds when run as a script (e.g., Streamlit demo)
    from config import POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD


# Compile a regex pattern to remove punctuation and digits
_punct_num_pattern = re.compile(r"[^a-zA-Z\s]")


def clean_comment(text: str, extra_stopwords: Optional[Iterable[str]] = None) -> str:
    """
    Clean a single customer comment by lowercasing, removing punctuation/numbers, 
    and filtering English and optional custom stop words.
    """
    if not isinstance(text, str):
        text = str(text)
        
    # Lower case
    text = text.lower()
    # Remove punctuation and numbers
    text = _punct_num_pattern.sub(" ", text)
    # Tokenize and define stop words
    words = text.split()
    stopwords = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stopwords.update([w.lower() for w in extra_stopwords])
        
    # Filter stop words and rejoin
    cleaned_words: List[str] = [w for w in words if w not in stopwords]
    return " ".join(cleaned_words)


def map_rating_to_label(rating: Optional[float | int | str | None]) -> Optional[str]:
    """
    Map a numeric rating to a label: "positive", "negative", or None (neutral/excluded).
    Uses POSITIVE_THRESHOLD and NEGATIVE_THRESHOLD from config.
    """
    if rating is None or rating == "":
        return None

    try:
        rating_val = float(rating)
    except (TypeError, ValueError):
        return None

    if rating_val >= POSITIVE_THRESHOLD:
        return "positive"
    if rating_val <= NEGATIVE_THRESHOLD:
        return "negative"
        
    return None # Neutral


def preprocess_dataframe(df: pd.DataFrame, extra_stopwords: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """
    Apply comment cleaning and rating-to-label mapping to the entire DataFrame.
    Returns a DataFrame with `comment` and `label`, dropping neutral/empty rows.
    """
    # Map ratings to labels and drop neutral rows
    df['label'] = df['rating'].apply(map_rating_to_label)
    df = df.dropna(subset=['label'])

    # Clean the comments
    df['comment'] = df['comment'].apply(clean_comment, extra_stopwords=extra_stopwords)
    
    # Drop rows where comment is now empty
    df = df[df['comment'].str.strip() != '']

    return df[['comment', 'label']]