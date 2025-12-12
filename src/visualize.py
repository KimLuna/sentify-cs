import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from . import data_loader, preprocess
from .config import MODEL_DIR, REPORTS_DIR

# Define paths for model artifacts and output figure
MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
VECTORISER_PATH = MODEL_DIR / "tfidf_vectoriser.joblib"
FIGURE_PATH = REPORTS_DIR / "confusion_matrix.png"

def main():
    """Generates and saves the Confusion Matrix visualization."""
    print("Generating Confusion Matrix...")

    # 1. Load data (Same process as training)
    raw_df = data_loader.load_raw_dataframe()
    df = data_loader.extract_comment_and_rating(raw_df)

    # 2. Text Preprocessing
    df['clean_text'] = df['comment'].apply(preprocess.clean_comment)
    df = df[df['clean_text'].str.strip().astype(bool)]

    # 3. Data Split (Using the same random_state=42 to reproduce the test set)
    X_text = df['clean_text']
    y = df['rating']  # 'done' or 'transfer'

    _, X_test_text, _, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Load Model & Vectorizer
    if not MODEL_PATH.exists() or not VECTORISER_PATH.exists():
        print(f"Model files not found. Please run 'python -m src.train' first.")
        return

    clf = joblib.load(MODEL_PATH)
    vectoriser = joblib.load(VECTORISER_PATH)

    # 5. Perform Prediction
    X_test = vectoriser.transform(X_test_text)
    y_pred = clf.predict(X_test)

    # 6. Generate Confusion Matrix
    labels = ['done', 'transfer']
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # 7. Plotting the Confusion Matrix 
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred: Done', 'Pred: Transfer'],
                yticklabels=['Actual: Done', 'Actual: Transfer'])
    
    plt.title('Confusion Matrix: Chat Routing (Done vs Transfer)', fontsize=15)
    plt.ylabel('Actual Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    
    # 8. Save Figure
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to: {FIGURE_PATH}")
    print("-" * 30)
    print("Interpretation:")
    print(f" - [0,0] (Top-Left): Correctly predicted 'Done' (True Negatives): {cm[0][0]}")
    print(f" - [1,1] (Bottom-Right): Correctly predicted 'Transfer' (True Positives): {cm[1][1]}")
    print(f" - [1,0] (Bottom-Left): Actual 'Transfer' but predicted 'Done' (False Negatives - Missed Transfer)")
    print("-" * 30)

if __name__ == "__main__":
    main()