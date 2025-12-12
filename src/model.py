from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE


def build_logistic_regression(C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
    """Construct an untrained Logistic Regression classifier."""
    return LogisticRegression(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            n_jobs=-1,
            penalty="l2",
    )


def build_svm(C: float = 1.0) -> SVC:
    """Construct an untrained Support Vector Machine (SVM) classifier."""
    return SVC(C=C, kernel='linear', random_state=RANDOM_STATE)


def build_random_forest(n_estimators: int = 100, max_depth: int = None) -> RandomForestClassifier:
    """Construct an untrained Random Forest classifier."""
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE)


def train_classifier(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'logistic', C: float = 1.0, 
    n_estimators: int = 100, max_depth: int = None, max_iter: int = 1000
) -> LogisticRegression | SVC | RandomForestClassifier:
    """
    Train a classifier ('logistic', 'svm', or 'random_forest') on the provided TF-IDF features.
    """
    if model_type == 'logistic':
        clf = build_logistic_regression(C=C, max_iter=max_iter)
    elif model_type == 'svm':
        clf = build_svm(C=C)
    elif model_type == 'random_forest':
        clf = build_random_forest(n_estimators=n_estimators, max_depth=max_depth)
    else:
        raise ValueError("Invalid model_type. Choose from ['logistic', 'svm', 'random_forest']")

    clf.fit(X_train, y_train)
    return clf