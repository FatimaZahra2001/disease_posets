"""Exact cohort loading and training-fold evidence orchestration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from .disease_evidence import probability_embedding
from .node_mapping import corrected_node_table


def map_labels(y):
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(
            pd.Series(y)
            .dropna()
            .unique()
            .tolist()
        )

        if set(vals).issubset({0, 1}):
            return y.astype(int)

        if set(vals).issubset({0, 0.5, 1}):
            return y.map({
                0: 0,
                0.5: 1,
                1: 1,
            }).astype(int)

        if set(vals).issubset({1, 2}):
            return y.map({
                1: 0,
                2: 1,
            }).astype(int)

        if set(vals).issubset({1, 2, 3}):
            return y.map({
                1: 0,
                2: 1,
                3: np.nan,
            })

    mapping = {
        "HC": 0,
        "CN": 0,
        "CONTROL": 0,
        "NORMAL": 0,
        "COGNITIVELY NORMAL": 0,
        "MCI": 1,
        "EMCI": 1,
        "LMCI": 1,
        "SMC": 1,
    }

    return (
        y.astype(str)
        .str.upper()
        .str.strip()
        .map(mapping)
    )


def infer_feature_columns(
    df,
    id_col="subject_id",
    label_col="label_str",
    exclude_cols=None,
):
    excluded = set(
        exclude_cols or []
    )

    excluded.update([
        id_col,
        label_col,
    ])

    return [
        column
        for column in df.select_dtypes(
            include=[np.number]
        ).columns
        if column not in excluded
    ]


def clean_feature_columns(
    feature_cols,
    exclude_keywords=None,
):
    bad = [
        "has_",
        "label",
        "source",
        "subject",
        "ptid",
        "rid",
        "cdglobal",
        "old_label",
        "n_extracted",
        "diagnosis",
        "dx",
    ]

    if exclude_keywords:
        bad.extend(
            exclude_keywords
        )

    cleaned = []
    removed = []

    for column in feature_cols:
        if any(
            keyword.lower() in column.lower()
            for keyword in bad
        ):
            removed.append(column)
        else:
            cleaned.append(column)

    return cleaned, removed


def load_data(args):
    d = (
        pd.read_csv(args.csv)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    d["_y"] = map_labels(
        d[args.label_col]
    )

    d = (
        d.dropna(subset=["_y"])
        .sort_values(args.subject_col)
        .reset_index(drop=True)
    )

    d["_y"] = d._y.astype(int)

    if d[args.subject_col].duplicated().any():
        raise AssertionError(
            "duplicate subjects"
        )

    f, r = clean_feature_columns(
        infer_feature_columns(
            d,
            args.subject_col,
            args.label_col,
        )
    )

    X, n, t = corrected_node_table(
        d,
        f,
    )

    from .tensor import ORIGINAL_ORDER

    if n != ORIGINAL_ORDER:
        raise AssertionError(
            "27-node manifest changed"
        )

    return (
        d,
        d._y.to_numpy(),
        d[args.subject_col]
        .astype(str)
        .to_numpy(),
        X,
        n,
        t,
        f,
    )


def fitted_evidence(
    Xtr,
    Xte,
    ytr,
    directions,
    embedding,
    args,
):
    imp = SimpleImputer(
        strategy="median"
    ).fit(Xtr)

    A = imp.transform(Xtr)
    B = imp.transform(Xte)

    (
        Ptr,
        Pte,
        Etr,
        Ete,
        stats,
    ) = probability_embedding(
        A,
        B,
        ytr,
        directions,
        embedding,
        args.epsilon,
        "abs_z",
        args.min_sigma,
    )

    if (
        Ptr.shape[1] != 27
        or np.any(Ptr <= 0)
        or np.any(Pte <= 0)
    ):
        raise AssertionError(
            "positive fixed evidence failed"
        )

    return (
        Ptr,
        Pte,
        Etr,
        Ete,
        stats,
    )


load_dataset = load_data