from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import joblib

from . import preprocess
from .config import MODEL_DIR, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sentiment inference on new text")
    parser.add_argument(
        "text",
        type=str,
        help="Input sentence(s) to classify. Use a semicolon ';' to separate multiple sentences.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_DIR / "sentiment_model.joblib"),
        help="Path to the trained model",
    )
    parser.add_argument(
        "--vectoriser_path",
        type=str,
        default=str(MODEL_DIR / "tfidf_vectoriser.joblib"),
        help="Path to the TF-IDF vectoriser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Clean input sentences (remove leading/trailing whitespaces)
    sentences = [s.strip() for s in args.text.split(";") if s.strip()]
    
    if not sentences:
        print("No input provided. Please pass one or more sentences.")
        return

    model_path = Path(args.model_path)
    vectoriser_path = Path(args.vectoriser_path)

    # Check if model and vectoriser exist
    if not model_path.exists() or not vectoriser_path.exists():
        raise FileNotFoundError(
            "Model or vectoriser files not found. Train the model first with `python -m src.train`."
        )

    # Load the trained model and vectoriser
    clf = joblib.load(model_path)
    vectoriser = joblib.load(vectoriser_path)

    # Clean and preprocess input sentences
    cleaned = [preprocess.clean_comment(s) for s in sentences]
    
    # Vectorize sentences
    X = vectoriser.transform(cleaned)

    # Predict sentiment labels
    preds = clf.predict(X)
    
    # Output predictions
    for s, p in zip(sentences, preds):
        sentiment = "positive" if p == "positive" else "negative"
        print(f"Text: \"{s}\" -> Predicted sentiment: {sentiment}")

if __name__ == "__main__":
    main()
