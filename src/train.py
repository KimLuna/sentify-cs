from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from . import data_loader, preprocess, model
from .config import (
    TEST_SIZE,
    RANDOM_STATE,
    MAX_FEATURES,
    MODEL_DIR,
    REPORTS_DIR,
    ensure_directories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment classifier")
    parser.add_argument(
        "--max_features",
        type=int,
        default=MAX_FEATURES,
        help="Maximum number of TF-IDF features to use",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data to reserve for testing",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(MODEL_DIR),
        help="Directory to store the trained model and vectoriser",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()
    model_output_dir = Path(args.output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess data
    raw_df = data_loader.load_raw_dataframe()
    df = data_loader.extract_comment_and_rating(raw_df)
    processed = preprocess.preprocess_dataframe(df)
    if processed.empty:
        raise ValueError("No data available after preprocessing. Check your rating thresholds.")
    texts = processed["comment"].tolist()
    labels = processed["label"].tolist()

    # Split data into train and test sets
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Vectorise text using TF-IDF
    vectoriser = TfidfVectorizer(max_features=args.max_features)
    X_train = vectoriser.fit_transform(X_train_texts)
    X_test = vectoriser.transform(X_test_texts)

    # Train classifier
    clf = model.train_classifier(X_train, np.array(y_train))

    # Evaluate on test set
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    report = classification_report(y_test, y_pred, digits=4)
    print(report)

    # Save model and vectoriser
    model_path = model_output_dir / "sentiment_model.joblib"
    vect_path = model_output_dir / "tfidf_vectoriser.joblib"
    joblib.dump(clf, model_path)
    joblib.dump(vectoriser, vect_path)
    print(f"Model saved to {model_path}")
    print(f"Vectoriser saved to {vect_path}")

    # Save classification report to file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_file = REPORTS_DIR / "classification_report.txt"
    with metrics_file.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
    print(f"Classification report saved to {metrics_file}")

if __name__ == "__main__":
    main()
