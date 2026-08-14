"""Disease-aware evidence embeddings for the MBA MRI experiment.

All reference statistics are estimated from the training partition only.
The functions in this module preserve the numerical behaviour of the
original disease-aware implementation.
"""

from __future__ import annotations

import numpy as np


def direction_sign(direction):
    """Return the disease-direction sign used by directional embeddings."""

    if direction == "increase":
        return 1.0

    if direction == "decrease":
        return -1.0

    return 0.0


def stable_sigmoid(x):
    """Numerically stable sigmoid."""

    x = np.asarray(x, dtype=float)

    out = np.empty_like(x, dtype=float)

    positive = x >= 0

    out[positive] = 1.0 / (
        1.0 + np.exp(-x[positive])
    )

    exp_x = np.exp(x[~positive])

    out[~positive] = exp_x / (
        1.0 + exp_x
    )

    return out


def gaussian_logpdf(x, mu, sigma):
    """Gaussian log-density."""

    return (
        -0.5
        * np.log(
            2.0
            * np.pi
            * sigma
            * sigma
        )
        - 0.5
        * ((x - mu) / sigma) ** 2
    )


def fit_hc_normative_stats(
    X_train,
    y_train,
    epsilon,
):
    """Estimate normative mean and SD from training HC subjects only."""

    hc = X_train[
        np.asarray(y_train) == 0
    ]

    if hc.shape[0] == 0:
        raise RuntimeError(
            "Cannot fit HC normative statistics: "
            "this training fold has no HC subjects."
        )

    mean = np.nanmean(
        hc,
        axis=0,
    )

    std = np.nanstd(
        hc,
        axis=0,
        ddof=1 if hc.shape[0] > 1 else 0,
    )

    std = np.where(
        np.isfinite(std)
        & (std > epsilon),
        std,
        1.0,
    )

    return mean, std


def burden_from_z(
    z,
    directions,
    embedding,
    epsilon,
    unknown_policy="abs_z",
):
    """Convert normative z-scores into non-negative disease burden."""

    directions = list(directions)

    out = np.zeros_like(
        z,
        dtype=float,
    )

    for j, direction in enumerate(directions):

        if embedding == "abs_z":
            out[:, j] = np.abs(
                z[:, j]
            )

        elif embedding == "signed_z":

            if direction == "decrease":
                out[:, j] = -z[:, j]

            elif direction == "increase":
                out[:, j] = z[:, j]

            else:
                out[:, j] = (
                    np.abs(z[:, j])
                    if unknown_policy == "abs_z"
                    else 0.0
                )

            out[:, j] = np.maximum(
                out[:, j],
                0.0,
            )

        elif embedding in {
            "atrophy_energy",
            "directional_z_positive",
            "rank_normalized_directional",
        }:

            if direction == "decrease":
                out[:, j] = np.maximum(
                    0.0,
                    -z[:, j],
                )

            elif direction == "increase":
                out[:, j] = np.maximum(
                    0.0,
                    z[:, j],
                )

            else:
                out[:, j] = (
                    np.abs(z[:, j])
                    if unknown_policy == "abs_z"
                    else 0.0
                )

        elif embedding == "directional_softplus_z":

            sign = direction_sign(
                direction
            )

            if sign:
                out[:, j] = np.logaddexp(
                    0.0,
                    sign * z[:, j],
                )

            else:
                out[:, j] = (
                    np.logaddexp(
                        0.0,
                        np.abs(z[:, j]),
                    )
                    if unknown_policy == "abs_z"
                    else 0.0
                )

        elif embedding == "directional_sigmoid_z":

            sign = direction_sign(
                direction
            )

            if sign:
                out[:, j] = stable_sigmoid(
                    sign * z[:, j]
                )

            else:
                out[:, j] = (
                    stable_sigmoid(
                        np.abs(z[:, j])
                    )
                    if unknown_policy == "abs_z"
                    else 0.0
                )

        else:
            raise ValueError(
                "Unsupported z-score embedding: "
                f"{embedding}"
            )

    return out + epsilon


def disease_likelihood_burden(
    z_train,
    z_test,
    y_train,
    directions,
    epsilon,
    unknown_policy="abs_z",
    min_sigma=1e-3,
    min_class_samples=3,
):
    """Construct disease-likelihood burden from training HC/MCI densities."""

    directions = list(directions)
    y_train = np.asarray(y_train)

    A_train = np.zeros_like(
        z_train,
        dtype=float,
    )

    A_test = np.zeros_like(
        z_test,
        dtype=float,
    )

    rows = []
    fallback_count = 0

    for j, direction in enumerate(directions):

        sign = direction_sign(
            direction
        )

        if sign:
            s_train = (
                sign * z_train[:, j]
            )
            s_test = (
                sign * z_test[:, j]
            )

        elif unknown_policy == "abs_z":
            s_train = np.abs(
                z_train[:, j]
            )
            s_test = np.abs(
                z_test[:, j]
            )

        else:
            A_train[:, j] = 0.0
            A_test[:, j] = 0.0

            rows.append({
                "feature_index": j,
                "fallback": "unknown_zero",
            })

            continue

        hc = s_train[
            (y_train == 0)
            & np.isfinite(s_train)
        ]

        mci = s_train[
            (y_train == 1)
            & np.isfinite(s_train)
        ]

        if (
            len(hc) < min_class_samples
            or len(mci) < min_class_samples
        ):
            A_train[:, j] = np.logaddexp(
                0.0,
                s_train,
            )

            A_test[:, j] = np.logaddexp(
                0.0,
                s_test,
            )

            fallback_count += 1

            rows.append({
                "feature_index": j,
                "fallback": (
                    "directional_softplus_z"
                ),
                "n_hc": len(hc),
                "n_mci": len(mci),
            })

            continue

        mu_hc = float(
            np.mean(hc)
        )

        mu_mci = float(
            np.mean(mci)
        )

        sigma_hc = float(
            max(
                np.std(hc, ddof=1),
                min_sigma,
            )
        )

        sigma_mci = float(
            max(
                np.std(mci, ddof=1),
                min_sigma,
            )
        )

        llr_train = (
            gaussian_logpdf(
                s_train,
                mu_mci,
                sigma_mci,
            )
            - gaussian_logpdf(
                s_train,
                mu_hc,
                sigma_hc,
            )
        )

        llr_test = (
            gaussian_logpdf(
                s_test,
                mu_mci,
                sigma_mci,
            )
            - gaussian_logpdf(
                s_test,
                mu_hc,
                sigma_hc,
            )
        )

        A_train[:, j] = np.logaddexp(
            0.0,
            llr_train,
        )

        A_test[:, j] = np.logaddexp(
            0.0,
            llr_test,
        )

        rows.append({
            "feature_index": j,
            "fallback": "",
            "n_hc": len(hc),
            "n_mci": len(mci),
            "mu_hc": mu_hc,
            "mu_mci": mu_mci,
            "sigma_hc": sigma_hc,
            "sigma_mci": sigma_mci,
        })

    stats = {
        "disease_likelihood_params": rows,
        "disease_likelihood_fallback_count":
            fallback_count,
    }

    return (
        A_train + epsilon,
        A_test + epsilon,
        stats,
    )


def empirical_percentile_transform(
    A_train,
    A_test,
):
    """Map burdens to empirical percentiles using training data."""

    A_train = np.asarray(
        A_train,
        dtype=float,
    )

    A_test = np.asarray(
        A_test,
        dtype=float,
    )

    P_train = np.zeros_like(
        A_train,
        dtype=float,
    )

    P_test = np.zeros_like(
        A_test,
        dtype=float,
    )

    for j in range(
        A_train.shape[1]
    ):
        reference = np.sort(
            A_train[
                np.isfinite(
                    A_train[:, j]
                ),
                j,
            ]
        )

        if len(reference) == 0:
            continue

        P_train[:, j] = (
            np.searchsorted(
                reference,
                A_train[:, j],
                side="right",
            )
            / len(reference)
        )

        P_test[:, j] = (
            np.searchsorted(
                reference,
                A_test[:, j],
                side="right",
            )
            / len(reference)
        )

    return P_train, P_test


def normalize_burdens_to_probability(
    A,
    epsilon,
):
    """Normalize non-negative burden vectors to probability vectors."""

    A = np.asarray(
        A,
        dtype=float,
    )

    A = np.where(
        np.isfinite(A)
        & (A > 0.0),
        A,
        0.0,
    )

    row_sums = A.sum(
        axis=1,
        keepdims=True,
    )

    bad = (
        ~np.isfinite(
            row_sums[:, 0]
        )
        | (
            row_sums[:, 0]
            <= epsilon
        )
    )

    P = np.zeros_like(
        A,
        dtype=float,
    )

    good = ~bad

    if np.any(good):
        P[good] = (
            A[good]
            / np.maximum(
                row_sums[good],
                epsilon,
            )
        )

    if (
        np.any(bad)
        and A.shape[1] > 0
    ):
        P[bad] = (
            1.0 / A.shape[1]
        )

    return P, int(
        np.sum(bad)
    )


def probability_embedding(
    X_train,
    X_test,
    y_train,
    directions,
    embedding,
    epsilon,
    unknown_policy="abs_z",
    min_sigma=1e-3,
):
    """Construct train-safe probability embeddings."""

    if embedding == "raw_volume":

        mins = np.nanmin(
            X_train,
            axis=0,
            keepdims=True,
        )

        shift = np.minimum(
            mins,
            0.0,
        )

        A_train = (
            X_train
            - shift
            + epsilon
        )

        A_test = (
            X_test
            - shift
            + epsilon
        )

        stats = {
            "hc_mean": None,
            "hc_std": None,
            "train_min_shift":
                shift.ravel().tolist(),
        }

    else:

        mean, std = (
            fit_hc_normative_stats(
                X_train,
                y_train,
                epsilon,
            )
        )

        z_train = (
            X_train - mean
        ) / std

        z_test = (
            X_test - mean
        ) / std

        if embedding == "disease_likelihood":

            (
                A_train,
                A_test,
                likelihood_stats,
            ) = disease_likelihood_burden(
                z_train,
                z_test,
                y_train,
                directions,
                epsilon,
                unknown_policy=unknown_policy,
                min_sigma=min_sigma,
            )

        elif (
            embedding
            == "rank_normalized_directional"
        ):

            raw_train = burden_from_z(
                z_train,
                directions,
                embedding,
                0.0,
                unknown_policy=unknown_policy,
            )

            raw_test = burden_from_z(
                z_test,
                directions,
                embedding,
                0.0,
                unknown_policy=unknown_policy,
            )

            (
                A_train,
                A_test,
            ) = empirical_percentile_transform(
                raw_train,
                raw_test,
            )

            A_train = (
                A_train + epsilon
            )

            A_test = (
                A_test + epsilon
            )

            likelihood_stats = {}

        else:

            A_train = burden_from_z(
                z_train,
                directions,
                embedding,
                epsilon,
                unknown_policy=unknown_policy,
            )

            A_test = burden_from_z(
                z_test,
                directions,
                embedding,
                epsilon,
                unknown_policy=unknown_policy,
            )

            likelihood_stats = {}

        stats = {
            "hc_mean": mean.tolist(),
            "hc_std": std.tolist(),
            "train_min_shift": None,
        }

        stats.update(
            likelihood_stats
        )

    (
        P_train,
        train_uniform_count,
    ) = normalize_burdens_to_probability(
        A_train,
        epsilon,
    )

    (
        P_test,
        test_uniform_count,
    ) = normalize_burdens_to_probability(
        A_test,
        epsilon,
    )

    stats[
        "train_uniform_probability_count"
    ] = train_uniform_count

    stats[
        "test_uniform_probability_count"
    ] = test_uniform_count

    return (
        P_train,
        P_test,
        A_train,
        A_test,
        stats,
    )