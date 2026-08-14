from __future__ import annotations

import argparse
import hashlib
import json


EPS = 1e-8

DEFAULT_REPRESENTATIONS = [
    "P",
    "Q1",
    "Q2",
    "Q3",
    "theta_Q1",
    "theta_Q2",
    "theta_P",
    "delta_theta_2",
    "delta_theta_3",
    "residual_Q2",
    "residual_log_Q2",
    "KL",
    "theta_Q2_delta3",
    "node_burden",
    "entropy_P",
    "raw_freesurfer",
    "pca_freesurfer",
]

METRICS = [
    "auc",
    "balanced_accuracy",
    "accuracy",
    "sensitivity",
    "specificity",
]


def hash_obj(x):
    payload = json.dumps(
        x,
        sort_keys=True,
        default=str,
    ).encode()

    return hashlib.sha256(payload).hexdigest()[:16]


def experiment_config_hash(args):
    # Runtime-only controls do not change the scientific experiment.
    excluded = {
        "resume",
        "overwrite",
        "checkpoint_every_folds",
        "n_jobs",
        "validation_gate",
        "projection_retries",
    }

    return hash_obj({
        k: v
        for k, v in vars(args).items()
        if k not in excluded
    })


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Official-MBA "
            "structural-MRI experiment."
        )
    )

    p.add_argument(
        "--csv",
        default=(
            "analysis/freesurfer_all_roi_outputs/"
            "all_freesurfer_roi_features.csv"
        ),
    )

    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--label-col", default="label_str")

    p.add_argument(
        "--outdir",
        default="mba_disease_geometry_results",
    )

    p.add_argument(
        "--embeddings",
        nargs="+",
        default=[
            "abs_z",
            "directional_z_positive",
            "directional_softplus_z",
            "disease_likelihood",
        ],
    )

    p.add_argument(
        "--layouts",
        nargs="+",
        default=[
            "biological_system_block",
            "inherited_compact",
        ],
    )

    p.add_argument("--max-layouts", type=int)

    p.add_argument(
        "--representations",
        nargs="+",
        default=DEFAULT_REPRESENTATIONS,
    )

    p.add_argument(
        "--classifiers",
        nargs="+",
        default=[
            "logreg_l1",
            "logreg_l2",
            "linear_svm",
        ],
    )

    p.add_argument(
        "--c-grid",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0],
    )

    p.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 30],
    )

    p.add_argument("--n-repeats", type=int, default=50)
    p.add_argument("--outer-splits", type=int, default=5)
    p.add_argument("--inner-splits", type=int, default=5)

    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epsilon", type=float, default=EPS)

    p.add_argument(
        "--probability-floor",
        type=float,
        default=1e-7,
    )

    p.add_argument(
        "--min-sigma",
        type=float,
        default=1e-3,
    )

    p.add_argument(
        "--mba-max-iter",
        type=int,
        default=1000,
    )

    p.add_argument(
        "--mba-tol",
        type=float,
        default=1e-8,
    )

    p.add_argument(
        "--projection-retries",
        type=int,
        default=8,
    )

    p.add_argument(
        "--checkpoint-every-folds",
        type=int,
        default=1,
    )

    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
    )

    p.add_argument(
        "--fail-on-nonconvergence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--projection-cache-dir")

    p.add_argument(
        "--validation-gate",
        action="store_true",
    )

    return p

build_arg_parser = build_parser
DEFAULT_REPS = DEFAULT_REPRESENTATIONS
