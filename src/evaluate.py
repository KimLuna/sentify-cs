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
    ensure_directories,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation settings."""
    parser = argparse.ArgumentParser(description="Evaluate the classifier.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_DIR / "sentiment_model.joblib"),
        help="Path to the trained model file.",
    )
    parser.add_argument(
        "--vectoriser_path",
        type=str,
        default=str(MODEL_DIR / "tfidf_vectoriser.joblib"),
        help="Path to the saved TF-IDF vectoriser file.",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=TEST_SIZE,
        help="Fraction of data reserved for testing.",
    )
    return parser.parse_args()


def main() -> None:
    """Main function to load model, evaluate, and save reports."""
    args = parse_args()
    ensure_directories()

    # Load trained artifacts
    model_path = Path(args.model_path)
    vectoriser_path = Path(args.vectoriser_path)
    if not model_path.exists() or not vectoriser_path.exists():
        raise FileNotFoundError(
            "Model or vectoriser files not found. Train the model first."
        )
    clf = joblib.load(model_path)
    vectoriser = joblib.load(vectoriser_path)

    # Load and preprocess data
    raw_df = data_loader.load_raw_dataframe()
    df = data_loader.extract_comment_and_rating(raw_df)
    processed = preprocess.preprocess_dataframe(df)
    if processed.empty:
        raise ValueError("No data available after preprocessing. Check configuration.")
    texts = processed["comment"].tolist()
    labels = processed["label"].tolist()

    # Split data for consistent evaluation
    _, X_test_texts, _, y_test = train_test_split(
        texts,
        labels,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    # Vectorize and predict
    X_test = vectoriser.transform(X_test_texts)
    y_pred = clf.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=["negative", "positive"])

    # Save metrics report
    metrics_path = REPORTS_DIR / "evaluation_report.txt"
    with metrics_path.open("w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(report)
    
    # Plot and save confusion matrix 
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["negative", "positive"], yticklabels=["negative", "positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    fig_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Print results
    print(f"Test accuracy: {acc:.3f}")
    print(report)
    print(f"Evaluation report saved to {metrics_path}")
    print(f"Confusion matrix saved to {fig_path}")

if __name__ == "__main__":
    main()