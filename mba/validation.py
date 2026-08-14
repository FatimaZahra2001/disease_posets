"""Validation gate for the official-MBA pipeline."""

from __future__ import annotations

import numpy as np

from .data import (
    fitted_evidence,
    load_data,
)
from .representations import (
    build_bank,
)
from .tensor import (
    BIOLOGICAL_ORDER,
)


def validation_gate(args):
    """Run fast numerical and leakage checks before an experiment run."""

    (
        _data,
        y,
        subject_ids,
        X,
        nodes,
        node_table,
        _features,
    ) = load_data(args)

    train_idx = np.arange(
        0,
        70,
    )

    test_idx = np.arange(
        70,
        80,
    )

    if not set(
        subject_ids[train_idx]
    ).isdisjoint(
        subject_ids[test_idx]
    ):
        raise AssertionError(
            "Train/test split isolation failed."
        )

    directions = (
        node_table
        .set_index("node")
        .loc[
            nodes,
            "direction",
        ]
        .tolist()
    )

    (
        P_train,
        _P_test,
        evidence_train,
        _evidence_test,
        _,
    ) = fitted_evidence(
        X[train_idx],
        X[test_idx],
        y[train_idx],
        directions,
        "abs_z",
        args,
    )

    first = build_bank(
        P_train[:3],
        evidence_train[:3],
        BIOLOGICAL_ORDER,
        args,
        "gate_a",
        subject_ids[
            train_idx
        ][:3],
        y[train_idx][:3],
    )

    second = build_bank(
        P_train[:3],
        evidence_train[:3],
        BIOLOGICAL_ORDER,
        args,
        "gate_b",
        subject_ids[
            train_idx
        ][:3],
        y[train_idx][:3],
    )

    for representations in [
        first[0],
        second[0],
    ]:
        if not any(
            np.var(value) > 0
            for value
            in representations.values()
        ):
            raise AssertionError(
                "No finite representation variance."
            )

    for key in [
        "Q1",
        "Q2",
        "Q3",
    ]:
        if not np.allclose(
            first[0][key],
            second[0][key],
        ):
            raise AssertionError(
                f"Determinism failed for {key}."
            )

    if not first[
        2
    ].optimizer_success.all():
        raise AssertionError(
            "At least one official MBA projection failed."
        )

    if not first[
        3
    ][
        [
            "passed_delta2",
            "passed_delta3",
        ]
    ].all().all():
        raise AssertionError(
            "Residual identity validation failed."
        )

    print(
        "VALIDATION GATE PASSED: "
        "positivity, normalization, convergence, "
        "constraints, Q3 identity, residual identities, "
        "split isolation, determinism, and variance."
    )


__all__ = [
    "validation_gate",
]
