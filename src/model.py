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
    """Construct a logistic regression classifier.

    Parameters
    ----------
    C: float, default=1.0
        Inverse of regularisation strength.  Smaller values specify
        stronger regularisation.
    max_iter: int, default=1000
        Maximum number of iterations for the solver to converge.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        The untrained logistic regression classifier.
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        n_jobs=-1,
        penalty="l2",
    )


def build_svm(C: float = 1.0) -> SVC:
    """Construct a Support Vector Machine (SVM) classifier.

    Parameters
    ----------
    C: float, default=1.0
        Regularization parameter. The strength of the regularization is inversely proportional to C.

    Returns
    -------
    sklearn.svm.SVC
        The untrained SVM classifier.
    """
    return SVC(C=C, kernel='linear', random_state=RANDOM_STATE)


def build_random_forest(n_estimators: int = 100, max_depth: int = None) -> RandomForestClassifier:
    """Construct a Random Forest classifier.

    Parameters
    ----------
    n_estimators: int, default=100
        The number of trees in the forest.
    max_depth: int, default=None
        The maximum depth of the trees. If None, then nodes are expanded until all leaves are pure.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        The untrained Random Forest classifier.
    """
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE)


def train_classifier(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str = 'logistic', C: float = 1.0, 
    n_estimators: int = 100, max_depth: int = None, max_iter: int = 1000
) -> LogisticRegression | SVC | RandomForestClassifier:
    """Train a classifier (Logistic Regression, SVM, or Random Forest) on the provided data.

    Parameters
    ----------
    X_train: numpy.ndarray
        Matrix of TF‑IDF features for the training set.
    y_train: numpy.ndarray
        Array of sentiment labels corresponding to `X_train`.
    model_type: str, default='logistic'
        The type of model to train. Options are 'logistic', 'svm', 'random_forest'.
    C: float, default=1.0
        Regularisation strength for logistic regression or SVM.
    n_estimators: int, default=100
        The number of trees in the forest (only for Random Forest).
    max_depth: int, default=None
        The maximum depth of the trees (only for Random Forest).
    max_iter: int, default=1000
        Maximum number of iterations for logistic regression.

    Returns
    -------
    Trained classifier
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
