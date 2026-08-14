"""Validation helpers for official MBA probability tensors."""

from __future__ import annotations

import numpy as np


def validate_probability_tensor(
    probability,
    context,
    subject_id,
    *,
    atol=1e-10,
):
    """Validate one 3x3x3 probability tensor."""

    array = np.asarray(
        probability,
        dtype=float,
    )

    finite = bool(
        np.all(np.isfinite(array))
    )

    minimum = (
        float(np.min(array))
        if array.size
        else float("nan")
    )

    maximum = (
        float(np.max(array))
        if array.size
        else float("nan")
    )

    total = float(
        np.sum(array)
    )

    problems = []

    if array.shape != (3, 3, 3):
        problems.append(
            f"shape={array.shape}, expected=(3, 3, 3)"
        )

    if not finite:
        problems.append(
            "contains NaN or infinity"
        )

    if finite and np.any(array < 0):
        problems.append(
            "contains negative values"
        )

    if (
        not np.isfinite(total)
        or not np.isclose(
            total,
            1.0,
            rtol=0.0,
            atol=atol,
        )
    ):
        problems.append(
            f"sum={total!r}, expected approximately 1.0"
        )

    if problems:
        raise ValueError(
            "Invalid probability tensor: "
            f"context={context} "
            f"subject={subject_id} "
            f"min={minimum!r} "
            f"max={maximum!r} "
            f"sum={total!r} "
            f"finite={finite}; "
            + "; ".join(problems)
        )

    return {
        "min": minimum,
        "max": maximum,
        "sum": total,
        "finite": finite,
    }


__all__ = [
    "validate_probability_tensor",
]