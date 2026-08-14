"""Summaries, paired comparisons, and figures for MBA experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon

from .config import METRICS
from .io import atomic_csv


def summarize(frame):
    """Summarize repeated outer-fold performance."""

    rows = []

    groups = [
        "embedding",
        "tensor_layout",
        "ordering_type",
        "representation",
        "classifier",
    ]

    for keys, group in frame.groupby(
        groups
    ):
        row = dict(
            zip(
                groups,
                keys,
            )
        )

        row["n_outer_folds"] = len(
            group
        )

        row["n_repeats"] = (
            group.repeat.nunique()
        )

        for metric in METRICS:

            mean = group[
                metric
            ].mean()

            std = group[
                metric
            ].std(ddof=1)

            error = (
                1.96
                * std
                / math.sqrt(
                    len(group)
                )
            )

            row[
                f"{metric}_mean"
            ] = mean

            row[
                f"{metric}_std"
            ] = std

            row[
                f"{metric}_ci95_low"
            ] = mean - error

            row[
                f"{metric}_ci95_high"
            ] = mean + error

        rows.append(row)

    return pd.DataFrame(rows)


def comparison_outputs(
    frame,
    out,
    seed,
):
    """Generate paired representation comparisons."""

    comparisons = [
        ("Q2", "Q1"),
        ("theta_Q2", "theta_Q1"),
        ("delta_theta_3", "theta_Q2"),
        (
            "theta_Q2_delta3",
            "theta_Q2",
        ),
        ("P", "Q2"),
    ]

    rng = np.random.default_rng(
        seed
    )

    rows = []

    for (
        embedding,
        layout,
        classifier,
    ), group in frame.groupby(
        [
            "embedding",
            "tensor_layout",
            "classifier",
        ]
    ):

        repeat_means = (
            group
            .groupby(
                [
                    "repeat",
                    "representation",
                ]
            )[
                [
                    "auc",
                    "balanced_accuracy",
                ]
            ]
            .mean()
            .reset_index()
        )

        available = set(
            repeat_means.representation
        )

        for left, right in comparisons:

            if not {
                left,
                right,
            } <= available:
                continue

            left_frame = (
                repeat_means[
                    repeat_means.representation
                    == left
                ]
                .set_index("repeat")
            )

            right_frame = (
                repeat_means[
                    repeat_means.representation
                    == right
                ]
                .set_index("repeat")
            )

            common = (
                left_frame.index
                .intersection(
                    right_frame.index
                )
            )

            for metric in [
                "auc",
                "balanced_accuracy",
            ]:

                difference = (
                    left_frame.loc[
                        common,
                        metric,
                    ]
                    - right_frame.loc[
                        common,
                        metric,
                    ]
                ).to_numpy()

                bootstraps = np.array([
                    rng.choice(
                        difference,
                        len(difference),
                        replace=True,
                    ).mean()
                    for _ in range(2000)
                ])

                try:
                    p_value = float(
                        wilcoxon(
                            difference
                        ).pvalue
                    )
                except ValueError:
                    p_value = 1.0

                rows.append({
                    "embedding":
                        embedding,
                    "tensor_layout":
                        layout,
                    "classifier":
                        classifier,
                    "comparison":
                        f"{left} minus {right}",
                    "left_representation":
                        left,
                    "right_representation":
                        right,
                    "metric":
                        metric,
                    "mean_paired_difference":
                        difference.mean(),
                    "bootstrap_ci95_low":
                        np.quantile(
                            bootstraps,
                            0.025,
                        ),
                    "bootstrap_ci95_high":
                        np.quantile(
                            bootstraps,
                            0.975,
                        ),
                    "wilcoxon_p":
                        p_value,
                    "n_repeats":
                        len(difference),
                    "dependence_note": (
                        "Repeated-CV inference is descriptive; "
                        "folds are dependent."
                    ),
                })

    comparisons_frame = (
        pd.DataFrame(rows)
    )

    if not comparisons_frame.empty:

        order = np.argsort(
            comparisons_frame[
                "wilcoxon_p"
            ].to_numpy()
        )

        adjusted = np.empty(
            len(comparisons_frame)
        )

        running = 0.0

        for rank_index, index in enumerate(
            order
        ):
            value = (
                (
                    len(
                        comparisons_frame
                    )
                    - rank_index
                )
                * comparisons_frame.iloc[
                    index
                ].wilcoxon_p
            )

            running = max(
                running,
                value,
            )

            adjusted[index] = min(
                running,
                1.0,
            )

        comparisons_frame[
            "holm_adjusted_p"
        ] = adjusted

    atomic_csv(
        comparisons_frame,
        out
        / "paired_representation_comparisons.csv",
    )

    ordering_summary = summarize(
        frame
    )

    atomic_csv(
        ordering_summary,
        out
        / "ordering_sensitivity_summary.csv",
    )

    embedding_summary = (
        ordering_summary
        .groupby(
            [
                "embedding",
                "representation",
                "classifier",
            ]
        )[
            [
                "auc_mean",
                "balanced_accuracy_mean",
            ]
        ]
        .mean()
        .reset_index()
    )

    atomic_csv(
        embedding_summary,
        out
        / "embedding_comparison_summary.csv",
    )

    biological = (
        frame[
            frame.ordering_type
            == "biological"
        ]
        .groupby(
            [
                "repeat",
                "representation",
                "classifier",
            ]
        )[
            [
                "auc",
                "balanced_accuracy",
            ]
        ]
        .mean()
    )

    random = (
        frame[
            frame.ordering_type
            == "random"
        ]
        .groupby(
            [
                "repeat",
                "representation",
                "classifier",
            ]
        )[
            [
                "auc",
                "balanced_accuracy",
            ]
        ]
        .mean()
    )

    control_rows = []

    for index in biological.index.intersection(
        random.index
    ):
        for metric in [
            "auc",
            "balanced_accuracy",
        ]:
            control_rows.append({
                "repeat": index[0],
                "representation": index[1],
                "classifier": index[2],
                "metric": metric,
                "biological":
                    biological.loc[
                        index,
                        metric,
                    ],
                "random_mean":
                    random.loc[
                        index,
                        metric,
                    ],
                "difference":
                    biological.loc[
                        index,
                        metric,
                    ]
                    - random.loc[
                        index,
                        metric,
                    ],
            })

    atomic_csv(
        pd.DataFrame(
            control_rows
        ),
        out
        / "biological_vs_random_ordering_comparisons.csv",
    )


def make_figures(
    frame,
    loss,
    diagnostics,
    out,
):
    """Generate lightweight diagnostic figures."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    def plot_metric(
        metric,
        representations,
        stem,
    ):
        present = [
            representation
            for representation
            in representations
            if representation
            in set(
                frame.representation
            )
        ]

        if not present:
            return

        summary = (
            frame[
                frame.representation
                .isin(present)
            ]
            .groupby(
                "representation"
            )[metric]
            .agg(["mean", "std"])
            .reindex(present)
        )

        fig, ax = plt.subplots(
            figsize=(9, 5)
        )

        ax.bar(
            summary.index,
            summary["mean"],
            yerr=summary["std"],
            capsize=3,
        )

        ax.set_ylabel(metric)

        ax.tick_params(
            axis="x",
            rotation=35,
        )

        fig.tight_layout()

        fig.savefig(
            out / f"{stem}.png",
            dpi=180,
        )

        fig.savefig(
            out / f"{stem}.pdf",
        )

        plt.close(fig)

    plot_metric(
        "auc",
        [
            "P",
            "Q1",
            "Q2",
            "Q3",
            "theta_Q1",
            "theta_Q2",
            "theta_P",
            "delta_theta_2",
            "delta_theta_3",
        ],
        "mba_representation_auc",
    )

    plot_metric(
        "balanced_accuracy",
        [
            "P",
            "Q1",
            "Q2",
            "Q3",
            "theta_Q1",
            "theta_Q2",
            "theta_P",
            "delta_theta_2",
            "delta_theta_3",
        ],
        "mba_representation_bacc",
    )

    plot_metric(
        "balanced_accuracy",
        [
            "theta_Q2",
            "delta_theta_3",
            "theta_Q2_delta3",
        ],
        "residual_higher_order_gain",
    )

    if not loss.empty:

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        loss.boxplot(
            column=[
                "KL_P_Q1",
                "KL_P_Q2",
            ],
            by="label",
            ax=ax,
        )

        fig.suptitle("")
        fig.tight_layout()

        fig.savefig(
            out
            / "projection_loss_by_diagnosis.png",
            dpi=180,
        )

        fig.savefig(
            out
            / "projection_loss_by_diagnosis.pdf",
        )

        plt.close(fig)

    if not diagnostics.empty:

        convergence = (
            diagnostics
            .groupby(
                "body_order"
            )
            .n_iterations
            .mean()
        )

        fig, ax = plt.subplots()

        ax.bar(
            convergence.index.astype(str),
            convergence,
        )

        ax.set_xlabel(
            "MBA body"
        )

        ax.set_ylabel(
            "Mean iterations"
        )

        fig.tight_layout()

        fig.savefig(
            out
            / "convergence_diagnostics.png",
            dpi=180,
        )

        fig.savefig(
            out
            / "convergence_diagnostics.pdf",
        )

        plt.close(fig)


__all__ = [
    "comparison_outputs",
    "make_figures",
    "summarize",
]
