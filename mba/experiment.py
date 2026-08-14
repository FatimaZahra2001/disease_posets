"""Main orchestration for the official-MBA MRI experiment.

This module contains the high-level experimental workflow for the structural
MRI many-body approximation (MBA) analysis.


"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
)

from .config import (
    experiment_config_hash,
    hash_obj,
)

from .data import (
    fitted_evidence,
    load_data,
)

from .io import (
    atomic_csv,
    atomic_json,
)

from .models import (
    attach_raw_baselines,
    candidates,
    metrics,
    pipe,
    rank,
    score,
    threshold,
)

from .official_backend import (
    import_true_pymba,
)

from .representations import (
    build_bank,
)

from .results import (
    comparison_outputs,
    make_figures,
    summarize,
)

from .tensor import (
    mapping_frames,
    orderings,
)

from .validation import (
    validation_gate,
)

# ---------------------------------------------------------------------------
# Small orchestration helpers
# ---------------------------------------------------------------------------


def _resolve_layouts(args):
    """Expand requested tensor layouts and validate their names."""

    available = orderings()
    chosen = []

    for name in args.layouts:
        if name == "controls":
            chosen.extend(
                layout
                for layout in available
                if layout.startswith(
                    ("system_", "stage_", "random_")
                )
            )
        else:
            chosen.append(name)

    # Preserve order while removing duplicates.
    chosen = list(dict.fromkeys(chosen))

    if args.max_layouts is not None:
        chosen = chosen[: args.max_layouts]

    unknown = set(chosen) - set(available)

    if unknown:
        raise ValueError(
            f"Unknown tensor layouts: {sorted(unknown)}"
        )

    return chosen


def _restore_previous_results(out):
    """Restore checkpointed result tables for a resumed experiment."""

    tables = {
        "fold_results.csv": [],
        "subject_level_predictions.csv": [],
        "mba_projection_diagnostics.csv": [],
        "residual_identity_checks.csv": [],
        "projection_loss_subject_summary.csv": [],
        "inner_cv_results.csv": [],
        "selected_coordinates.csv": [],
    }

    for filename, rows in tables.items():
        path = out / filename

        if path.exists():
            rows.extend(
                pd.read_csv(path).to_dict("records")
            )

    return tables


def _save_checkpoint_tables(out, tables):
    """Atomically write all fold-level checkpoint tables."""

    for filename, rows in tables.items():
        atomic_csv(
            pd.DataFrame(rows),
            out / filename,
        )


def _completed_outer_folds(fold_rows):
    """Return the set of outer folds already completed."""

    if not fold_rows:
        return set()

    return set(
        pd.DataFrame(fold_rows)
        .outer_fold_id
        .unique()
    )


def _select_best_inner_configuration(rows):
    """Select the best inner-CV hyperparameter configuration.

    Configurations are ranked by:
        1. balanced accuracy,
        2. ROC-AUC,
        3. smaller feature budget,
        4. smaller regularisation parameter C.
    """

    return sorted(
        rows,
        key=lambda row: (
            -row["balanced_accuracy"],
            -row["auc"],
            row["selected_k"],
            row["C"],
        ),
    )[0]


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run(args):
    """Run the complete official-MBA structural-MRI experiment."""

    # ------------------------------------------------------------------
    # Optional validation-only execution
    # ------------------------------------------------------------------

    if args.validation_gate:
        return validation_gate(args)

    # ------------------------------------------------------------------
    # Output directory and checkpoint compatibility
    # ------------------------------------------------------------------

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if (
        any(out.iterdir())
        and not args.resume
        and not args.overwrite
    ):
        raise RuntimeError(
            "Output directory is non-empty. "
            "Use --resume or --overwrite."
        )

    cfg_hash = experiment_config_hash(args)
    checkpoint = out / "checkpoint_progress.json"

    if args.resume and checkpoint.exists():
        saved = json.loads(
            checkpoint.read_text()
        )

        saved_hash = saved.get("config_hash")

        if saved_hash != cfg_hash:
            raise RuntimeError(
                "Checkpoint configuration mismatch: "
                f"saved={saved_hash!r}, "
                f"requested={cfg_hash!r}. "
                "Use a new output directory."
            )

    # ------------------------------------------------------------------
    # Dataset and biological tensor mapping
    # ------------------------------------------------------------------

    (
        data,
        y,
        subject_ids,
        X,
        nodes,
        node_table,
        features,
    ) = load_data(args)

    mapping, ordering_frame = mapping_frames(
        node_table
    )

    atomic_csv(
        mapping,
        out / "biological_tensor_mapping.csv",
    )

    atomic_csv(
        ordering_frame,
        out / "tensor_orderings.csv",
    )

    layouts = _resolve_layouts(args)

    # Direction of disease-related change for each biological node.
    directions = (
        node_table
        .set_index("node")
        .loc[nodes, "direction"]
        .tolist()
    )

    # ------------------------------------------------------------------
    # Repeated nested cross-validation
    # ------------------------------------------------------------------

    outer_cv = RepeatedStratifiedKFold(
        n_splits=args.outer_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )

    tables = {
        "fold_results.csv": [],
        "subject_level_predictions.csv": [],
        "mba_projection_diagnostics.csv": [],
        "residual_identity_checks.csv": [],
        "projection_loss_subject_summary.csv": [],
        "inner_cv_results.csv": [],
        "selected_coordinates.csv": [],
    }

    if args.resume:
        tables = _restore_previous_results(out)

    fold_rows = tables["fold_results.csv"]
    prediction_rows = tables[
        "subject_level_predictions.csv"
    ]
    diagnostic_rows = tables[
        "mba_projection_diagnostics.csv"
    ]
    identity_rows = tables[
        "residual_identity_checks.csv"
    ]
    loss_rows = tables[
        "projection_loss_subject_summary.csv"
    ]
    inner_rows = tables[
        "inner_cv_results.csv"
    ]
    selected_rows = tables[
        "selected_coordinates.csv"
    ]

    completed = _completed_outer_folds(
        fold_rows
    )

    total_outer_folds = (
        args.n_repeats
        * args.outer_splits
    )

    # ==================================================================
    # OUTER CROSS-VALIDATION LOOP
    # ==================================================================

    for outer_id, (outer_train, outer_test) in enumerate(
        outer_cv.split(X, y),
        start=1,
    ):
        if outer_id in completed:
            continue

        repeat = (
            (outer_id - 1)
            // args.outer_splits
            + 1
        )

        fold = (
            (outer_id - 1)
            % args.outer_splits
            + 1
        )

        print(
            f"Outer {outer_id}/{total_outer_folds}",
            flush=True,
        )

        inner_cv = StratifiedKFold(
            n_splits=args.inner_splits,
            shuffle=True,
            random_state=(
                args.seed
                + repeat * 1009
                + fold * 97
            ),
        )

        best_configs = {}

        # ==============================================================
        # EMBEDDING × TENSOR-LAYOUT LOOP
        # ==============================================================

        for embedding in args.embeddings:
            for layout in layouts:

                tensor_order = orderings()[layout]

                ordering_type = (
                    ordering_frame.loc[
                        ordering_frame.ordering_id
                        == layout,
                        "ordering_type",
                    ]
                    .iloc[0]
                )

                # ------------------------------------------------------
                # Inner CV: hyperparameter and feature-budget selection
                # ------------------------------------------------------

                oof_predictions = {}

                for inner_id, (
                    inner_train,
                    inner_validation,
                ) in enumerate(
                    inner_cv.split(
                        X[outer_train],
                        y[outer_train],
                    )
                ):
                    train_idx = outer_train[
                        inner_train
                    ]

                    validation_idx = outer_train[
                        inner_validation
                    ]

                    (
                        P_train,
                        P_validation,
                        evidence_train,
                        evidence_validation,
                        _,
                    ) = fitted_evidence(
                        X[train_idx],
                        X[validation_idx],
                        y[train_idx],
                        directions,
                        embedding,
                        args,
                    )

                    train_bank = build_bank(
                        P_train,
                        evidence_train,
                        tensor_order,
                        args,
                        f"inner_{outer_id}_{inner_id}_train",
                        subject_ids[train_idx],
                        y[train_idx],
                    )

                    validation_bank = build_bank(
                        P_validation,
                        evidence_validation,
                        tensor_order,
                        args,
                        f"inner_{outer_id}_{inner_id}_val",
                        subject_ids[validation_idx],
                        y[validation_idx],
                    )

                    attach_raw_baselines(
                        train_bank,
                        validation_bank,
                        data.loc[
                            train_idx,
                            features,
                        ].to_numpy(float),
                        data.loc[
                            validation_idx,
                            features,
                        ].to_numpy(float),
                    )

                    # --------------------------------------------------
                    # Representation × classifier × hyperparameter grid
                    # --------------------------------------------------

                    for representation in args.representations:

                        X_train = train_bank[0][
                            representation
                        ]

                        X_validation = (
                            validation_bank[0][
                                representation
                            ]
                        )

                        feature_ranking = rank(
                            X_train,
                            y[train_idx],
                        )

                        for classifier in args.classifiers:
                            for C in args.c_grid:
                                for k in candidates(
                                    X_train.shape[1],
                                    args.k,
                                ):
                                    key = (
                                        embedding,
                                        layout,
                                        representation,
                                        classifier,
                                        C,
                                        k,
                                    )

                                    oof_predictions.setdefault(
                                        key,
                                        np.full(
                                            len(outer_train),
                                            np.nan,
                                        ),
                                    )

                                    selected = (
                                        feature_ranking[:k]
                                    )

                                    model = pipe(
                                        classifier,
                                        C,
                                        args.seed
                                        + outer_id
                                        + inner_id,
                                    )

                                    model.fit(
                                        X_train[:, selected],
                                        y[train_idx],
                                    )

                                    scores = score(
                                        model,
                                        X_validation[
                                            :, selected
                                        ],
                                    )

                                    oof_predictions[key][
                                        inner_validation
                                    ] = scores

                # ------------------------------------------------------
                # Evaluate complete inner-CV predictions
                # ------------------------------------------------------

                for key, scores in oof_predictions.items():

                    decision_threshold = threshold(
                        y[outer_train],
                        scores,
                    )

                    inner_metrics, _ = metrics(
                        y[outer_train],
                        scores,
                        decision_threshold,
                    )

                    inner_rows.append({
                        "outer_fold_id": outer_id,
                        "repeat": repeat,
                        "fold": fold,
                        "embedding": key[0],
                        "tensor_layout": key[1],
                        "representation": key[2],
                        "classifier": key[3],
                        "C": key[4],
                        "selected_k": key[5],
                        "threshold": decision_threshold,
                        **inner_metrics,
                    })

                # ------------------------------------------------------
                # Select best configuration for each representation
                # ------------------------------------------------------

                for representation in args.representations:
                    for classifier in args.classifiers:

                        candidate_rows = [
                            row
                            for row in inner_rows
                            if (
                                row["outer_fold_id"]
                                == outer_id
                                and row["embedding"]
                                == embedding
                                and row["tensor_layout"]
                                == layout
                                and row["representation"]
                                == representation
                                and row["classifier"]
                                == classifier
                            )
                        ]

                        best_configs[
                            (
                                embedding,
                                layout,
                                representation,
                                classifier,
                            )
                        ] = (
                            _select_best_inner_configuration(
                                candidate_rows
                            )
                        )

                # ------------------------------------------------------
                # Refit disease representation on full outer training set
                # ------------------------------------------------------

                (
                    P_train,
                    P_test,
                    evidence_train,
                    evidence_test,
                    _,
                ) = fitted_evidence(
                    X[outer_train],
                    X[outer_test],
                    y[outer_train],
                    directions,
                    embedding,
                    args,
                )

                train_bank = build_bank(
                    P_train,
                    evidence_train,
                    tensor_order,
                    args,
                    f"outer_{outer_id}_train",
                    subject_ids[outer_train],
                    y[outer_train],
                )

                test_bank = build_bank(
                    P_test,
                    evidence_test,
                    tensor_order,
                    args,
                    f"outer_{outer_id}_test",
                    subject_ids[outer_test],
                    y[outer_test],
                )

                attach_raw_baselines(
                    train_bank,
                    test_bank,
                    data.loc[
                        outer_train,
                        features,
                    ].to_numpy(float),
                    data.loc[
                        outer_test,
                        features,
                    ].to_numpy(float),
                )

                # ------------------------------------------------------
                # MBA diagnostics and projection-loss outputs
                # ------------------------------------------------------

                diagnostic_frames = [
                    (train_bank[2], diagnostic_rows),
                    (test_bank[2], diagnostic_rows),
                    (train_bank[3], identity_rows),
                    (test_bank[3], identity_rows),
                    (test_bank[4], loss_rows),
                ]

                for frame, target in diagnostic_frames:

                    frame = frame.assign(
                        repeat=repeat,
                        fold=fold,
                        outer_fold_id=outer_id,
                        embedding=embedding,
                        tensor_layout=layout,
                        ordering_type=ordering_type,
                    )

                    target.extend(
                        frame.to_dict("records")
                    )

                # ------------------------------------------------------
                # Final outer-fold evaluation
                # ------------------------------------------------------

                for representation in args.representations:

                    X_train = train_bank[0][
                        representation
                    ]

                    X_test = test_bank[0][
                        representation
                    ]

                    coordinate_names = train_bank[1][
                        representation
                    ]

                    for classifier in args.classifiers:

                        best = best_configs[
                            (
                                embedding,
                                layout,
                                representation,
                                classifier,
                            )
                        ]

                        feature_ranking = rank(
                            X_train,
                            y[outer_train],
                        )

                        selected = feature_ranking[
                            : int(
                                best["selected_k"]
                            )
                        ]

                        model = pipe(
                            classifier,
                            float(best["C"]),
                            args.seed + outer_id,
                        )

                        model.fit(
                            X_train[:, selected],
                            y[outer_train],
                        )

                        scores = score(
                            model,
                            X_test[:, selected],
                        )

                        outer_metrics, predictions = (
                            metrics(
                                y[outer_test],
                                scores,
                                float(
                                    best["threshold"]
                                ),
                            )
                        )

                        classifier_model = (
                            model.named_steps["clf"]
                        )

                        if hasattr(
                            classifier_model,
                            "coef_",
                        ):
                            coefficients = np.ravel(
                                classifier_model.coef_
                            )
                        else:
                            coefficients = np.full(
                                len(selected),
                                np.nan,
                            )

                        # ----------------------------------------------
                        # Fold-level result
                        # ----------------------------------------------

                        fold_rows.append({
                            "repeat": repeat,
                            "outer_fold": fold,
                            "outer_fold_id": outer_id,
                            "subject_count_train": len(
                                outer_train
                            ),
                            "subject_count_test": len(
                                outer_test
                            ),
                            "embedding": embedding,
                            "tensor_layout": layout,
                            "ordering_type": ordering_type,
                            "ordering_id": layout,
                            "representation": representation,
                            "mba_body": (
                                int(representation[-1])
                                if representation
                                in {"Q1", "Q2", "Q3"}
                                else np.nan
                            ),
                            "classifier": classifier,
                            "candidate_dimension": (
                                X_train.shape[1]
                            ),
                            "finite_dimension": int(
                                np.isfinite(X_train)
                                .all(axis=0)
                                .sum()
                            ),
                            "selected_k": len(selected),
                            "selected_dimension": len(
                                selected
                            ),
                            "nonzero_coefficients": int(
                                np.sum(
                                    np.abs(coefficients)
                                    > 1e-12
                                )
                            ),
                            "threshold": best[
                                "threshold"
                            ],
                            **outer_metrics,
                            "converged_fraction": float(
                                train_bank[2]
                                .optimizer_success
                                .mean()
                            ),
                            "random_seed": args.seed,
                        })

                        # ----------------------------------------------
                        # Subject-level predictions
                        # ----------------------------------------------

                        for j, subject_index in enumerate(
                            outer_test
                        ):
                            prediction_rows.append({
                                "subject_id": (
                                    subject_ids[
                                        subject_index
                                    ]
                                ),
                                "label": y[
                                    subject_index
                                ],
                                "repeat": repeat,
                                "fold": fold,
                                "outer_fold_id": (
                                    outer_id
                                ),
                                "embedding": embedding,
                                "tensor_layout": layout,
                                "ordering_type": (
                                    ordering_type
                                ),
                                "representation": (
                                    representation
                                ),
                                "classifier": classifier,
                                "score": scores[j],
                                "threshold": best[
                                    "threshold"
                                ],
                                "prediction": (
                                    predictions[j]
                                ),
                            })

                        # ----------------------------------------------
                        # Selected coordinate information
                        # ----------------------------------------------

                        for j, coordinate_index in enumerate(
                            selected
                        ):
                            if X_train.shape[1] == 26:
                                tensor_index = (
                                    np.unravel_index(
                                        coordinate_index + 1,
                                        (3, 3, 3),
                                    )
                                )

                                coordinate_order = int(
                                    sum(
                                        value > 0
                                        for value
                                        in tensor_index
                                    )
                                )

                                axis_components = str(
                                    tensor_index
                                )
                            else:
                                coordinate_order = (
                                    np.nan
                                )
                                axis_components = ""

                            coefficient = (
                                coefficients[j]
                            )

                            if coefficient > 0:
                                coefficient_sign = (
                                    "positive"
                                )
                            elif coefficient < 0:
                                coefficient_sign = (
                                    "negative"
                                )
                            else:
                                coefficient_sign = "zero"

                            selected_rows.append({
                                "repeat": repeat,
                                "fold": fold,
                                "embedding": embedding,
                                "tensor_layout": layout,
                                "representation": (
                                    representation
                                ),
                                "classifier": classifier,
                                "coordinate": (
                                    coordinate_names[
                                        coordinate_index
                                    ]
                                ),
                                "coordinate_index": int(
                                    coordinate_index
                                ),
                                "coordinate_order": (
                                    coordinate_order
                                ),
                                "axis_components": (
                                    axis_components
                                ),
                                "biological_label": (
                                    "tensor coordinate"
                                ),
                                "selection_frequency": 1.0,
                                "cohens_d": np.nan,
                                "coefficient": coefficient,
                                "coefficient_sign": (
                                    coefficient_sign
                                ),
                            })

        # --------------------------------------------------------------
        # Checkpoint after each completed outer fold
        # --------------------------------------------------------------

        _save_checkpoint_tables(
            out,
            tables,
        )

        completed_ids = sorted(
            set(
                row["outer_fold_id"]
                for row in fold_rows
            )
        )

        atomic_json(
            {
                "completed_outer_fold_ids": (
                    completed_ids
                ),
                "n_completed_outer_folds": len(
                    completed_ids
                ),
                "config_hash": cfg_hash,
            },
            checkpoint,
        )

    # ==================================================================
    # FINAL RESULT SUMMARIES
    # ==================================================================

    fold_frame = pd.DataFrame(fold_rows)
    diagnostic_frame = pd.DataFrame(
        diagnostic_rows
    )
    identity_frame = pd.DataFrame(
        identity_rows
    )
    loss_frame = pd.DataFrame(loss_rows)
    selected_frame = pd.DataFrame(
        selected_rows
    )

    summary = summarize(fold_frame)

    atomic_csv(
        summary,
        out
        / "summary_by_representation_classifier.csv",
    )

    convergence_summary = (
        diagnostic_frame
        .groupby(
            [
                "embedding",
                "tensor_layout",
                "body_order",
            ]
        )
        .agg(
            projection_count=(
                "optimizer_success",
                "size",
            ),
            convergence_rate=(
                "optimizer_success",
                "mean",
            ),
            max_eta_mismatch=(
                "active_eta_mismatch_max",
                "max",
            ),
            max_forbidden_theta=(
                "forbidden_theta_max",
                "max",
            ),
        )
        .reset_index()
    )

    atomic_csv(
        convergence_summary,
        out / "mba_convergence_summary.csv",
    )

    atomic_csv(
        identity_frame,
        out / "mba_constraint_checks.csv",
    )

    projection_group_summary = (
        loss_frame
        .groupby(
            [
                "embedding",
                "tensor_layout",
                "label",
            ]
        )[
            [
                "KL_P_Q1",
                "KL_P_Q2",
                "KL_Q2_Q1",
                "gain_body2",
            ]
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    atomic_csv(
        projection_group_summary,
        out / "projection_loss_group_summary.csv",
    )

    coordinate_stability = (
        selected_frame
        .groupby(
            [
                "embedding",
                "tensor_layout",
                "representation",
                "classifier",
                "coordinate",
            ]
        )
        .size()
        .rename("selected_outer_folds")
        .reset_index()
    )

    atomic_csv(
        coordinate_stability,
        out / "coordinate_stability_summary.csv",
    )

    # ------------------------------------------------------------------
    # Statistical comparisons and figures
    # ------------------------------------------------------------------

    comparison_outputs(
        fold_frame,
        out,
        args.seed,
    )

    make_figures(
        fold_frame,
        loss_frame,
        diagnostic_frame,
        out,
    )

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    run_config = {
        "arguments": vars(args),
        "config_hash": cfg_hash,
        "timestamp": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
        "repository_git_commit": (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "official_mba_module_path": (
            import_true_pymba()["module_path"]
        ),
        "official_call": (
            "mproject.MBA_LBFGS(P, body)"
        ),
        "node_mapping_hash": hash_obj(
            node_table.to_dict("records")
        ),
        "tensor_mapping_hash": hash_obj(
            mapping.to_dict("records")
        ),
    }

    atomic_json(
        run_config,
        out / "run_config.json",
    )

    n_completed = len(
        set(
            row["outer_fold_id"]
            for row in fold_rows
        )
    )

    print(
        f"Completed {n_completed}/"
        f"{total_outer_folds} folds; "
        f"outputs={out}"
    )


__all__ = ["run"]
