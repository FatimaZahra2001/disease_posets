"""Classifiers, feature ranking, thresholds, and raw baselines."""

from __future__ import annotations

import numpy as np

from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def score(model, X):
    """Return continuous classifier scores."""

    if hasattr(model, "decision_function"):
        return model.decision_function(X)

    return model.predict_proba(X)[:, 1]


def classifier(
    name,
    C,
    seed,
):
    """Construct one of the prespecified linear classifiers."""

    if name == "logreg_l1":
        return LogisticRegression(
            C=C,
            penalty="l1",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )

    if name == "logreg_l2":
        return LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=5000,
            random_state=seed,
        )

    if name == "linear_svm":
        return SVC(
            C=C,
            kernel="linear",
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )

    raise ValueError(
        f"Unknown classifier: {name}"
    )


def pipe(
    name,
    C,
    seed,
):
    """Build the leakage-safe classifier pipeline."""

    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scale",
            StandardScaler(),
        ),
        (
            "clf",
            classifier(
                name,
                C,
                seed,
            ),
        ),
    ])


def threshold(
    y_true,
    scores,
):
    """Choose the threshold maximizing balanced accuracy."""

    unique_scores = np.unique(
        scores
    )

    if len(unique_scores) > 1:
        midpoints = (
            unique_scores[:-1]
            + unique_scores[1:]
        ) / 2

        candidates_ = np.unique(
            np.r_[
                unique_scores,
                midpoints,
            ]
        )

    else:
        candidates_ = (
            unique_scores
        )

    values = [
        balanced_accuracy_score(
            y_true,
            scores >= value,
        )
        for value in candidates_
    ]

    return float(
        candidates_[
            np.argmax(values)
        ]
    )


def metrics(
    y_true,
    scores,
    decision_threshold,
):
    """Calculate held-out classification metrics."""

    prediction = (
        scores >= decision_threshold
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[0, 1],
        ).ravel()
    )

    values = {
        "auc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),
        "accuracy": float(
            np.mean(
                prediction
                == y_true
            )
        ),
        "sensitivity": float(
            tp / max(tp + fn, 1)
        ),
        "specificity": float(
            tn / max(tn + fp, 1)
        ),
    }

    return values, prediction


def rank(
    X,
    y,
):
    """Rank representation coordinates using training-only ANOVA."""

    scores, _ = f_classif(
        X,
        y,
    )

    scores = np.nan_to_num(
        scores,
        nan=0,
        posinf=0,
        neginf=0,
    )

    return np.argsort(
        scores,
        kind="stable",
    )[::-1]


def candidates(
    dimension,
    budgets,
):
    """Return valid feature budgets for a representation."""

    if dimension == 1:
        return [1]

    valid = [
        k
        for k in budgets
        if k <= dimension
    ]

    return (
        valid
        if valid
        else [dimension]
    )


def attach_raw_baselines(
    train_bank,
    test_bank,
    raw_train,
    raw_test,
):
    """Attach raw FreeSurfer and PCA baselines to representation banks."""

    imputer = SimpleImputer(
        strategy="median"
    ).fit(raw_train)

    train = imputer.transform(
        raw_train
    )

    test = imputer.transform(
        raw_test
    )

    train_bank[0][
        "raw_freesurfer"
    ] = train

    test_bank[0][
        "raw_freesurfer"
    ] = test

    raw_names = [
        f"raw_{i}"
        for i in range(
            train.shape[1]
        )
    ]

    train_bank[1][
        "raw_freesurfer"
    ] = raw_names

    test_bank[1][
        "raw_freesurfer"
    ] = raw_names

    scaler = StandardScaler().fit(
        train
    )

    train_scaled = scaler.transform(
        train
    )

    test_scaled = scaler.transform(
        test
    )

    n_components = min(
        20,
        train.shape[0] - 1,
        train.shape[1],
    )

    pca = PCA(
        n_components=n_components,
        random_state=42,
    ).fit(train_scaled)

    train_bank[0][
        "pca_freesurfer"
    ] = pca.transform(
        train_scaled
    )

    test_bank[0][
        "pca_freesurfer"
    ] = pca.transform(
        test_scaled
    )

    pca_names = [
        f"pca_{i + 1}"
        for i in range(
            n_components
        )
    ]

    train_bank[1][
        "pca_freesurfer"
    ] = pca_names

    test_bank[1][
        "pca_freesurfer"
    ] = pca_names


__all__ = [
    "attach_raw_baselines",
    "candidates",
    "classifier",
    "metrics",
    "pipe",
    "rank",
    "score",
    "threshold",
]