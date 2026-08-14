"""Official MBA projections and derived representation banks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .official_backend import (
    body_indices,
    import_true_pymba,
    kl_pq,
    official_project,
)
from .projection import (
    validate_probability_tensor,
)
from .tensor import (
    ORIGINAL_ORDER,
    tensors_from_evidence,
)


def build_bank(
    node_probabilities,
    evidence,
    order,
    args,
    context,
    subject_ids,
    labels,
):
    """Construct official MBA representations for one split.

    The official PyMBA implementation is used for all body projections.
    """

    # ------------------------------------------------------------------
    # Construct the subject-specific 3x3x3 tensors
    # ------------------------------------------------------------------

    P = np.asarray(
        tensors_from_evidence(
            node_probabilities,
            ORIGINAL_ORDER,
            order,
        ),
        dtype=np.float64,
    )

    if args.probability_floor:
        P = np.maximum(
            P,
            float(args.probability_floor),
        )

    totals = P.sum(
        axis=(1, 2, 3),
        keepdims=True,
    )

    if (
        not np.all(np.isfinite(totals))
        or np.any(totals <= 0)
    ):
        raise ValueError(
            "Invalid probability totals before "
            f"normalisation: context={context}"
        )

    P = P / totals

    input_stats = [
        validate_probability_tensor(
            probability,
            context,
            subject_ids[i],
        )
        for i, probability in enumerate(P)
    ]

    # ------------------------------------------------------------------
    # Official PyMBA natural coordinates
    # ------------------------------------------------------------------

    backend = import_true_pymba()
    transform = backend["transform"]

    coordinate_indices = np.array([
        index
        for index in np.ndindex(
            3,
            3,
            3,
        )
        if any(index)
    ])

    theta = {
        "P": np.array([
            transform.theta_from_prob(
                probability
            )[tuple(coordinate_indices.T)]
            for probability in P
        ])
    }

    Q = {}
    diagnostics = []

    # ------------------------------------------------------------------
    # Official body-1/body-2/body-3 projections
    # ------------------------------------------------------------------

    for body in (1, 2, 3):

        projected = []

        for i, probability in enumerate(P):

            stats = input_stats[i]
            last_error = None

            for attempt in range(
                1,
                args.projection_retries + 2,
            ):
                if attempt == 1:
                    projection_init = "uniform"
                    projection_seed = (
                        args.seed + i
                    )
                else:
                    projection_init = "random"
                    projection_seed = (
                        args.seed
                        + i
                        + 104729 * (attempt - 1)
                    )

                try:
                    validate_probability_tensor(
                        probability,
                        context,
                        subject_ids[i],
                    )

                    (
                        projected_probability,
                        _theta,
                        _eta,
                        _history,
                        diagnostic,
                    ) = official_project(
                        probability,
                        body,
                        max_iter=args.mba_max_iter,
                        tol=args.mba_tol,
                        init=projection_init,
                        method="lbfgs",
                        seed=projection_seed,
                    )

                    validate_probability_tensor(
                        projected_probability,
                        (
                            f"{context} returned_Q "
                            f"attempt={attempt}"
                        ),
                        subject_ids[i],
                    )

                    if (
                        args.fail_on_nonconvergence
                        and not diagnostic[
                            "optimizer_success"
                        ]
                    ):
                        raise RuntimeError(
                            "optimizer_success is false"
                        )

                except Exception as exc:
                    last_error = exc
                    continue

                projected.append(
                    projected_probability
                )

                diagnostics.append({
                    "subject_id":
                        subject_ids[i],
                    "label":
                        labels[i],
                    "context":
                        context,
                    "projection_initialization":
                        projection_init,
                    "projection_seed":
                        projection_seed,
                    "projection_attempt":
                        attempt,
                    "input_probability_floor":
                        args.probability_floor,
                    "input_min_after_floor":
                        stats["min"],
                    "input_max_after_floor":
                        stats["max"],
                    "input_probability_sum":
                        stats["sum"],
                    **diagnostic,
                })

                break

            else:
                raise RuntimeError(
                    "Projection failed after "
                    f"{args.projection_retries + 1} "
                    "initialisations: "
                    f"context={context} "
                    f"subject={subject_ids[i]} "
                    f"body={body} "
                    f"input_P_min={stats['min']!r} "
                    f"input_P_max={stats['max']!r} "
                    f"input_P_sum={stats['sum']!r} "
                    f"input_P_finite={stats['finite']} "
                    f"last_error="
                    f"{type(last_error).__name__}: "
                    f"{last_error}"
                ) from last_error

        Q[body] = np.array(
            projected
        )

        theta[body] = np.array([
            transform.theta_from_prob(
                probability
            )[tuple(coordinate_indices.T)]
            for probability in Q[body]
        ])

    # ------------------------------------------------------------------
    # Incremental natural-coordinate structure
    # ------------------------------------------------------------------

    delta_theta_2 = (
        theta[2]
        - theta[1]
    )

    delta_theta_3 = (
        theta["P"]
        - theta[2]
    )

    identity_checks = []
    projection_losses = []

    for i, probability in enumerate(P):

        error_delta_2 = (
            theta[2][i]
            - (
                theta[1][i]
                + delta_theta_2[i]
            )
        )

        error_delta_3 = (
            theta["P"][i]
            - (
                theta[2][i]
                + delta_theta_3[i]
            )
        )

        identity_checks.append({
            "subject_id":
                subject_ids[i],
            "max_abs_error_delta2":
                np.abs(error_delta_2).max(),
            "mean_abs_error_delta2":
                np.abs(error_delta_2).mean(),
            "max_abs_error_delta3":
                np.abs(error_delta_3).max(),
            "mean_abs_error_delta3":
                np.abs(error_delta_3).mean(),
            "passed_delta2":
                bool(
                    np.abs(
                        error_delta_2
                    ).max() < 1e-10
                ),
            "passed_delta3":
                bool(
                    np.abs(
                        error_delta_3
                    ).max() < 1e-10
                ),
            "tolerance": 1e-10,
        })

        kl_p_q1 = kl_pq(
            probability,
            Q[1][i],
        )

        kl_p_q2 = kl_pq(
            probability,
            Q[2][i],
        )

        kl_q2_q1 = kl_pq(
            Q[2][i],
            Q[1][i],
        )

        projection_losses.append({
            "subject_id":
                subject_ids[i],
            "label":
                labels[i],
            "KL_P_Q1":
                kl_p_q1,
            "KL_P_Q2":
                kl_p_q2,
            "KL_Q2_Q1":
                kl_q2_q1,
            "gain_body2":
                kl_p_q1 - kl_p_q2,
            "entropy_P":
                float(
                    -np.sum(
                        probability
                        * np.log(probability)
                    )
                ),
            "entropy_Q1":
                float(
                    -np.sum(
                        Q[1][i]
                        * np.log(Q[1][i])
                    )
                ),
            "entropy_Q2":
                float(
                    -np.sum(
                        Q[2][i]
                        * np.log(Q[2][i])
                    )
                ),
        })

    # ------------------------------------------------------------------
    # Active coordinate subsets
    # ------------------------------------------------------------------

    full_tuples = list(
        map(
            tuple,
            coordinate_indices,
        )
    )

    positions = {
        coordinate: i
        for i, coordinate
        in enumerate(full_tuples)
    }

    active_1 = [
        positions[tuple(index)]
        for index in body_indices(1)
    ]

    active_2 = [
        positions[tuple(index)]
        for index in body_indices(2)
    ]

    loss_matrix = np.array([
        [
            row["KL_P_Q1"],
            row["KL_P_Q2"],
            row["KL_Q2_Q1"],
            row["gain_body2"],
        ]
        for row in projection_losses
    ])

    # ------------------------------------------------------------------
    # Representation bank
    # ------------------------------------------------------------------

    representations = {
        "P":
            P.reshape(len(P), -1),

        "Q1":
            Q[1].reshape(len(P), -1),

        "Q2":
            Q[2].reshape(len(P), -1),

        "Q3":
            Q[3].reshape(len(P), -1),

        "theta_P":
            theta["P"],

        "theta_Q1":
            theta[1][:, active_1],

        "theta_Q2":
            theta[2][:, active_2],

        "theta_Q3":
            theta[3],

        "delta_theta_2":
            delta_theta_2,

        "delta_theta_3":
            delta_theta_3,

        "residual_Q2":
            (
                P - Q[2]
            ).reshape(len(P), -1),

        "residual_log_Q2":
            (
                np.log(P)
                - np.log(Q[2])
            ).reshape(len(P), -1),

        "KL":
            loss_matrix,

        "theta_Q1_delta2":
            np.c_[
                theta[1][:, active_1],
                delta_theta_2,
            ],

        "theta_Q2_delta3":
            np.c_[
                theta[2][:, active_2],
                delta_theta_3,
            ],

        "theta_Q1_delta2_delta3":
            np.c_[
                theta[1][:, active_1],
                delta_theta_2,
                delta_theta_3,
            ],

        "KL_delta3":
            np.c_[
                loss_matrix[:, [0, 1, 3]],
                delta_theta_3,
            ],

        "node_burden":
            evidence,

        "node_composition":
            node_probabilities,

        "entropy_P":
            np.array([
                [row["entropy_P"]]
                for row
                in projection_losses
            ]),
    }

    # ------------------------------------------------------------------
    # Human-readable coordinate names
    # ------------------------------------------------------------------

    tensor_names = [
        f"cell_{a}_{b}_{c}"
        for a, b, c
        in np.ndindex(
            3,
            3,
            3,
        )
    ]

    theta_names = [
        f"theta_{a}_{b}_{c}"
        for a, b, c
        in coordinate_indices
    ]

    names = {}

    for key, values in representations.items():

        if values.shape[1] == 26:
            names[key] = [
                f"{key}_{name}"
                for name in theta_names
            ]

        elif values.shape[1] == 27:
            names[key] = [
                f"{key}_{name}"
                for name in tensor_names
            ]

        else:
            names[key] = [
                f"{key}_{i}"
                for i
                in range(
                    values.shape[1]
                )
            ]

    names["theta_Q1"] = [
        (
            "theta_Q1_"
            + "_".join(
                map(
                    str,
                    body_indices(1)[i],
                )
            )
        )
        for i in range(
            len(active_1)
        )
    ]

    names["theta_Q2"] = [
        (
            "theta_Q2_"
            + "_".join(
                map(
                    str,
                    body_indices(2)[i],
                )
            )
        )
        for i in range(
            len(active_2)
        )
    ]

    return (
        representations,
        names,
        pd.DataFrame(diagnostics),
        pd.DataFrame(identity_checks),
        pd.DataFrame(projection_losses),
    )


__all__ = [
    "build_bank",
]
