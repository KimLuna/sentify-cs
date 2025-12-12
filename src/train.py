from __future__ import annotations

import pandas as pd
from pathlib import Path
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from . import data_loader, preprocess
from .config import MODEL_DIR

# Define paths for saving model artifacts and report
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECTORISER_PATH = MODEL_DIR / "tfidf_vectoriser.joblib"
REPORT_PATH = MODEL_DIR.parent / "classification_report.txt"

def main() -> None:
    """Main function to load data, train a classifier, and save artifacts."""
    print("Starting training process (Target: Transferred Chat)")
    print("Model: Random Forest")

    # 1. Load data
    raw_df = data_loader.load_raw_dataframe()
    df = data_loader.extract_comment_and_rating(raw_df)

    # 2. Data validation
    if df.empty:
        raise ValueError("Data is empty after loading.")

    print(f"Total training data samples: {len(df)}")

    # 3. Text Preprocessing
    print("Preprocessing text...")
    # Clean comments and filter out rows where text became empty after cleaning
    df['clean_text'] = df['comment'].apply(preprocess.clean_comment)
    df = df[df['clean_text'].str.strip().astype(bool)]

    # 4. Separate features (X) and target (y)
    X_text = df['clean_text']
    y = df['rating']  # Target labels: 'transfer' / 'done'

    # 5. Split data into training and testing sets
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Feature Vectorization (TF-IDF)
    # Increased max_features to 7000
    vectoriser = TfidfVectorizer(max_features=7000)
    X_train = vectoriser.fit_transform(X_train_text)
    X_test = vectoriser.transform(X_test_text)

    # 7. Model Training (Random Forest)
    print("Training model... (This may take a moment)")
    # n_jobs=-1 uses all available CPU cores for faster training
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 8. Performance Evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("Detailed Report:")
    print(report)

    # 9. Save Model, Vectorizer, and Report
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(vectoriser, VECTORISER_PATH)
    
    with open(REPORT_PATH, "w") as f:
        f.write(report)

    print(f"Model and vectorizer saved successfully.")

if __name__ == "__main__":
    main()