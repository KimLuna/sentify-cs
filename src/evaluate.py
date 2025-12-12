from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from . import data_loader, preprocess
from .config import (
    MODEL_DIR,
    REPORTS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    MAX_FEATURES,
    ensure_directories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sentiment classifier")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_DIR / "sentiment_model.joblib"),
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--vectoriser_path",
        type=str,
        default=str(MODEL_DIR / "tfidf_vectoriser.joblib"),
        help="Path to the saved TF-IDF vectoriser file",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data to reserve for testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    # Load model and vectoriser
    model_path = Path(args.model_path)
    vectoriser_path = Path(args.vectoriser_path)
    if not model_path.exists() or not vectoriser_path.exists():
        raise FileNotFoundError(
            f"Model or vectoriser files not found. Train the model first with `python -m src.train`."
        )
    clf = joblib.load(model_path)
    vectoriser = joblib.load(vectoriser_path)

    # Load and preprocess data
    raw_df = data_loader.load_raw_dataframe()
    df = data_loader.extract_comment_and_rating(raw_df)
    processed = preprocess.preprocess_dataframe(df)
    if processed.empty:
        raise ValueError("No data available after preprocessing. Check your rating thresholds.")
    texts = processed["comment"].tolist()
    labels = processed["label"].tolist()

    # Split into train/test for evaluation (using same seed)
    _, X_test_texts, _, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Vectorise test set
    X_test = vectoriser.transform(X_test_texts)
    # Predict
    y_pred = clf.predict(X_test)
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

    # Save metrics to text file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "evaluation_report.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
    # Plot confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Evaluation report saved to {metrics_path}")
    print(f"Confusion matrix saved to {fig_path}")

    # Also print summary to console
    print(f"Test accuracy: {acc:.3f}")
    print(report)

if __name__ == "__main__":
    main()
