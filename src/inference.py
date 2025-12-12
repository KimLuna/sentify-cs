from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import re
from . import preprocess
from .config import MODEL_DIR


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments, including the input text and model paths."""
    parser = argparse.ArgumentParser(description="Predict chat transfer status.")
    parser.add_argument("text", type=str, help="Input sentence(s), separated by semicolons.")
    parser.add_argument("--model_path", type=str, default=str(MODEL_DIR / "sentiment_model.joblib"))
    parser.add_argument("--vectoriser_path", type=str, default=str(MODEL_DIR / "tfidf_vectoriser.joblib"))
    return parser.parse_args()


def check_keyword_rules(text: str) -> float:
    """
    Apply business rules: force a high transfer probability if critical keywords are present.
    Returns the rule-based probability (e.g., 0.85) or 0.0 if no rule is matched.
    """
    text_lower = text.lower()
    
    # Critical keywords that trigger a mandatory 'transfer' recommendation
    critical_keywords = [
        "supervisor", "manager", "connect me", "transfer", 
        "complaint", "speak to", "immediate", "urgent", "broken"
    ]
    
    for kw in critical_keywords:
        if kw in text_lower:
            return 0.85  # Forced high probability
            
    return 0.0


def main() -> None:
    """Main function for loading the model and making predictions."""
    args = parse_args()
    # Split input text by semicolon to handle multiple sentences
    sentences = [s.strip() for s in args.text.split(";") if s.strip()]
    
    if not sentences:
        print("No input sentence provided.")
        return

    # Load artifacts
    clf = joblib.load(args.model_path)
    vectoriser = joblib.load(args.vectoriser_path)

    # Preprocess text and vectorize
    cleaned = [preprocess.clean_comment(s) for s in sentences]
    X = vectoriser.transform(cleaned)
    
    # Get AI model's pure prediction probabilities (index 1 is 'transfer' class)
    probs = clf.predict_proba(X)

    # Process and output results for each sentence
    THRESHOLD = 0.2  # Decision threshold for 'transfer'

    for s, p in zip(sentences, probs):
        # 1. AI's pure prediction
        ai_prob_transfer = p[1]
        
        # 2. Check Rule-based override
        rule_prob = check_keyword_rules(s)
        
        # 3. Final probability determination (Rule overrides AI if matched)
        if rule_prob > 0:
            final_prob = rule_prob
        else:
            final_prob = ai_prob_transfer

        # 4. Output Result
        print(f"💬 Input: \"{s}\"")
        
        if final_prob > THRESHOLD:
            prediction = "Transfer"
            status = "🚨 [Prediction: Transfer Required]"
        else:
            prediction = "Resolved Here"
            status = "✅ [Prediction: Resolved Here]"
            
        print(status)
        print(f"   -> Transfer Probability: {final_prob*100:.1f}%")
        print("-" * 30)


if __name__ == "__main__":
    main()