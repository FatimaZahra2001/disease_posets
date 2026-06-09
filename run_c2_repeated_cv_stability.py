import argparse
import itertools
import json
import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

from scipy.stats import ttest_ind

from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


C2_KEYS = [
    "hippocampus",
    "entorhinal",
    "parahippocampal",
    "amygdala",
    "fusiform",
    "inferiortemporal",
    "middletemporal",
    "superiortemporal",
    "temporalpole",
    "bankssts",
]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(s))
    return re.sub(r"_+", "_", s).strip("_")


def map_labels(y):
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(pd.Series(y).dropna().unique().tolist())
        if set(vals).issubset({0, 1}):
            return y.astype(int)
        if set(vals).issubset({0, 0.5, 1}):
            return y.map({0: 0, 0.5: 1, 1: 1}).astype(int)
        if set(vals).issubset({1, 2}):
            return y.map({1: 0, 2: 1}).astype(int)
        if set(vals).issubset({1, 2, 3}):
            return y.map({1: 0, 2: 1, 3: np.nan})

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

    return y.astype(str).str.upper().str.strip().map(mapping)


def infer_feature_columns(df, id_col, label_col):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    bad_keywords = [
        "has_", "label", "source", "subject", "ptid", "rid",
        "cdglobal", "old_label", "n_extracted", "diagnosis", "dx",
    ]

    cols = []
    for c in numeric:
        if c in [id_col, label_col]:
            continue
        low = c.lower()
        if any(k in low for k in bad_keywords):
            continue
        cols.append(c)

    return cols


def select_c2_features(feature_cols):
    selected = []
    for c in feature_cols:
        low = c.lower()
        if any(k.lower() in low for k in C2_KEYS):
            selected.append(c)
    return selected


def make_positive(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float64)
    min_per_col = np.nanmin(X, axis=0, keepdims=True)
    return X - np.minimum(min_per_col, 0.0) + eps


def normalize_rows_to_prob(X, eps=1e-12):
    X = np.maximum(np.asarray(X, dtype=np.float64), eps)
    return X / np.maximum(X.sum(axis=1, keepdims=True), eps)


def chain_poset_theta(P, eps=1e-12):
    logP = np.log(np.maximum(P, eps))
    return np.diff(logP, axis=1, prepend=np.zeros((logP.shape[0], 1)))


def interaction_features(theta, feature_names, ell):
    n, d = theta.shape

    if ell == 1:
        return theta, feature_names

    combos = list(itertools.combinations(range(d), ell))
    Xell = np.zeros((n, len(combos)), dtype=np.float64)

    names = []
    for j, combo in enumerate(combos):
        Xell[:, j] = np.prod(theta[:, combo], axis=1)
        names.append(" x ".join([feature_names[i] for i in combo]))

    return Xell, names


def poset_features(X_train, X_test, feature_names, ell):
    Ptr = normalize_rows_to_prob(make_positive(X_train))
    Pte = normalize_rows_to_prob(make_positive(X_test))

    th_tr = chain_poset_theta(Ptr)
    th_te = chain_poset_theta(Pte)

    return interaction_features(th_tr, feature_names, ell)[0], interaction_features(th_te, feature_names, ell)[0], interaction_features(th_tr, feature_names, ell)[1]


def shuffled_order(X_train, X_test, feature_names, rng):
    idx = np.arange(len(feature_names))
    rng.shuffle(idx)
    return X_train[:, idx], X_test[:, idx], [feature_names[i] for i in idx]


def select_topk_train_only(X_train, X_test, y_train, names, k):
    if k is None or k <= 0 or X_train.shape[1] <= k:
        return X_train, X_test, names

    selector = SelectKBest(score_func=f_classif, k=k)
    Xtr = selector.fit_transform(X_train, y_train)
    Xte = selector.transform(X_test)

    keep = selector.get_support()
    kept_names = [n for n, keep_i in zip(names, keep) if keep_i]

    return Xtr, Xte, kept_names


def compute_effects(X_train, y_train, names):
    rows = []

    for j, name in enumerate(names):
        hc = X_train[y_train == 0, j]
        mci = X_train[y_train == 1, j]

        hc_mean = np.nanmean(hc)
        mci_mean = np.nanmean(mci)
        hc_std = np.nanstd(hc, ddof=1)
        mci_std = np.nanstd(mci, ddof=1)

        pooled = np.sqrt(
            ((len(hc) - 1) * hc_std**2 + (len(mci) - 1) * mci_std**2)
            / max(len(hc) + len(mci) - 2, 1)
        )

        cohens_d = (mci_mean - hc_mean) / pooled if pooled > 0 else 0.0

        try:
            _, pval = ttest_ind(mci, hc, equal_var=False, nan_policy="omit")
        except Exception:
            pval = np.nan

        rows.append({
            "coordinate": name,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
            "mci_minus_hc": mci_mean - hc_mean,
            "p_value": pval,
        })

    return pd.DataFrame(rows)


def metric_dict(y_true, y_score):
    y_pred = (y_score >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
        "sens_mci": tp / max(tp + fn, 1),
        "spec_hc": tn / max(tn + fp, 1),
    }


def get_models(seed):
    return {
        "linear_svm": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
        "logreg_l2": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            penalty="l2",
            random_state=seed,
        ),
        "logreg_l1": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            penalty="l1",
            C=0.5,
            random_state=seed,
        ),
    }


def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return 1 / (1 + np.exp(-s))


def fit_eval(X_train, X_test, y_train, y_test, model_name, seed):
    model = get_models(seed)[model_name]

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", clone(model)),
    ])

    pipe.fit(X_train, y_train)
    y_score = predict_score(pipe, X_test)
    return metric_dict(y_test, y_score)


def empirical_pvalue(observed, null_values, min_nulls=20):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[~np.isnan(null_values)]

    if len(null_values) < min_nulls:
        return np.nan

    return (1.0 + np.sum(null_values >= observed)) / (len(null_values) + 1.0)


def run(args):
    ensure_dir(args.outdir)

    df = pd.read_csv(args.csv).replace([np.inf, -np.inf], np.nan)
    df[args.label_col] = map_labels(df[args.label_col])
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    feature_cols = infer_feature_columns(df, args.id_col, args.label_col)
    c2_cols = select_c2_features(feature_cols)

    if len(c2_cols) < 4:
        raise ValueError(f"Too few C2 features found: {len(c2_cols)}")

    X = df[c2_cols].values.astype(float)
    y = df[args.label_col].values.astype(int)

    print(f"Subjects: {len(df)}")
    print(f"Labels: {df[args.label_col].value_counts().to_dict()}")
    print(f"C2 features: {len(c2_cols)}")
    print(f"Repeated CV: {args.n_repeats} repeats x {args.n_splits} folds")
    print(f"Nulls per repeat/fold: {args.n_null_posets}")
    print(f"Empirical p-values set to NA unless n_nulls >= {args.min_nulls_for_pvalue}")

    topk = {
        1: args.topk_m1,
        2: args.topk_m2,
        3: args.topk_m3,
    }

    all_metrics = []
    all_null_metrics = []
    all_effects = []
    winner_rows = []

    for repeat in range(args.n_repeats):
        seed = args.seed + repeat
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=seed)

        repeat_fold_metrics = []

        print(f"\nRepeat {repeat + 1}/{args.n_repeats}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train_raw = X[train_idx]
            X_test_raw = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            imputer = SimpleImputer(strategy="median")
            X_train = imputer.fit_transform(X_train_raw)
            X_test = imputer.transform(X_test_raw)

            vt = VarianceThreshold(threshold=args.variance_threshold)
            X_train = vt.fit_transform(X_train)
            X_test = vt.transform(X_test)

            names = [c for c, keep in zip(c2_cols, vt.get_support()) if keep]

            for ell in args.ells:
                try:
                    Xtr_pos, Xte_pos, coord_names = poset_features(X_train, X_test, names, ell)
                    Xtr_pos, Xte_pos, kept_names = select_topk_train_only(
                        Xtr_pos, Xte_pos, y_train, coord_names, topk[ell]
                    )

                    effects = compute_effects(Xtr_pos, y_train, kept_names)
                    effects["repeat"] = repeat
                    effects["fold"] = fold
                    effects["ell"] = f"M{ell}"
                    all_effects.append(effects)

                    for model_name in args.models.split(","):
                        model_name = model_name.strip()

                        md = fit_eval(Xtr_pos, Xte_pos, y_train, y_test, model_name, seed + fold)

                        row = {
                            "repeat": repeat,
                            "fold": fold,
                            "ell": f"M{ell}",
                            "model": model_name,
                            "feature_set": f"C2_MTL_temporal_M{ell}",
                            "is_null": False,
                            "dim": Xtr_pos.shape[1],
                            **md,
                        }

                        all_metrics.append(row)
                        repeat_fold_metrics.append(row)

                    rng_base = np.random.default_rng(seed + fold + ell * 10000)

                    for null_i in range(args.n_null_posets):
                        rng = np.random.default_rng(rng_base.integers(0, 10_000_000))
                        Xtr_shuf, Xte_shuf, names_shuf = shuffled_order(X_train, X_test, names, rng)

                        Xtr_null, Xte_null, null_coord_names = poset_features(
                            Xtr_shuf, Xte_shuf, names_shuf, ell
                        )

                        Xtr_null, Xte_null, null_kept = select_topk_train_only(
                            Xtr_null, Xte_null, y_train, null_coord_names, topk[ell]
                        )

                        for null_model in args.null_models.split(","):
                            null_model = null_model.strip()
                            md_null = fit_eval(
                                Xtr_null, Xte_null, y_train, y_test, null_model, seed + fold + null_i
                            )

                            all_null_metrics.append({
                                "repeat": repeat,
                                "fold": fold,
                                "ell": f"M{ell}",
                                "model": null_model,
                                "null_id": null_i,
                                "feature_set": f"null_C2_MTL_temporal_M{ell}",
                                "is_null": True,
                                "dim": Xtr_null.shape[1],
                                **md_null,
                            })

                except Exception as e:
                    print(f"  Skipping repeat={repeat}, fold={fold}, M{ell}: {e}")

        rep_df = pd.DataFrame(repeat_fold_metrics)

        for model_name in args.models.split(","):
            model_name = model_name.strip()
            sub = rep_df[rep_df["model"] == model_name]

            if len(sub) == 0:
                continue

            order_summary = (
                sub.groupby("ell")
                .agg(mean_auc=("auc", "mean"), mean_bacc=("bacc", "mean"))
                .reset_index()
            )

            if len(order_summary) > 0:
                best_auc = order_summary.sort_values("mean_auc", ascending=False).iloc[0]
                best_bacc = order_summary.sort_values("mean_bacc", ascending=False).iloc[0]

                winner_rows.append({
                    "repeat": repeat,
                    "model": model_name,
                    "winner_auc": best_auc["ell"],
                    "winner_auc_value": best_auc["mean_auc"],
                    "winner_bacc": best_bacc["ell"],
                    "winner_bacc_value": best_bacc["mean_bacc"],
                })

    metrics = pd.DataFrame(all_metrics)
    null_metrics = pd.DataFrame(all_null_metrics)
    winners = pd.DataFrame(winner_rows)

    metrics.to_csv(os.path.join(args.outdir, "observed_fold_metrics.csv"), index=False)
    null_metrics.to_csv(os.path.join(args.outdir, "null_fold_metrics.csv"), index=False)
    winners.to_csv(os.path.join(args.outdir, "winner_by_repeat.csv"), index=False)

    observed_summary = (
        metrics.groupby(["ell", "model"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            bacc_mean=("bacc", "mean"),
            bacc_std=("bacc", "std"),
            sens_mci_mean=("sens_mci", "mean"),
            sens_mci_std=("sens_mci", "std"),
            spec_hc_mean=("spec_hc", "mean"),
            spec_hc_std=("spec_hc", "std"),
            n_splits=("auc", "count"),
        )
        .reset_index()
        .sort_values(["auc_mean", "bacc_mean"], ascending=False)
    )

    observed_summary.to_csv(os.path.join(args.outdir, "observed_repeated_cv_summary.csv"), index=False)

    null_summary = (
        null_metrics.groupby(["ell", "model", "null_id"])
        .agg(
            auc_mean=("auc", "mean"),
            bacc_mean=("bacc", "mean"),
            sens_mci_mean=("sens_mci", "mean"),
        )
        .reset_index()
    )

    null_summary.to_csv(os.path.join(args.outdir, "null_repeated_cv_summary_by_null.csv"), index=False)

    pval_rows = []

    for _, obs in observed_summary.iterrows():
        ell = obs["ell"]
        model = obs["model"]

        null_sub = null_summary[
            (null_summary["ell"] == ell)
            & (null_summary["model"] == model)
        ]

        n_nulls = len(null_sub)

        pval_rows.append({
            "ell": ell,
            "model": model,
            "observed_auc": obs["auc_mean"],
            "null_auc_mean": null_sub["auc_mean"].mean() if n_nulls else np.nan,
            "null_auc_std": null_sub["auc_mean"].std() if n_nulls > 1 else np.nan,
            "empirical_p_auc": empirical_pvalue(obs["auc_mean"], null_sub["auc_mean"], args.min_nulls_for_pvalue),
            "observed_bacc": obs["bacc_mean"],
            "null_bacc_mean": null_sub["bacc_mean"].mean() if n_nulls else np.nan,
            "null_bacc_std": null_sub["bacc_mean"].std() if n_nulls > 1 else np.nan,
            "empirical_p_bacc": empirical_pvalue(obs["bacc_mean"], null_sub["bacc_mean"], args.min_nulls_for_pvalue),
            "observed_sens_mci": obs["sens_mci_mean"],
            "null_sens_mci_mean": null_sub["sens_mci_mean"].mean() if n_nulls else np.nan,
            "null_sens_mci_std": null_sub["sens_mci_mean"].std() if n_nulls > 1 else np.nan,
            "empirical_p_sens_mci": empirical_pvalue(obs["sens_mci_mean"], null_sub["sens_mci_mean"], args.min_nulls_for_pvalue),
            "n_nulls": n_nulls,
            "pvalue_note": (
                "NA: insufficient null samples"
                if n_nulls < args.min_nulls_for_pvalue
                else "computed"
            ),
        })

    pvals = pd.DataFrame(pval_rows).sort_values(["empirical_p_auc", "empirical_p_bacc"], na_position="last")
    pvals.to_csv(os.path.join(args.outdir, "empirical_pvalues_vs_shuffled_null.csv"), index=False)

    winner_auc_counts = winners.groupby(["model", "winner_auc"]).size().reset_index(name="count")
    winner_bacc_counts = winners.groupby(["model", "winner_bacc"]).size().reset_index(name="count")

    winner_auc_counts.to_csv(os.path.join(args.outdir, "winner_counts_by_auc.csv"), index=False)
    winner_bacc_counts.to_csv(os.path.join(args.outdir, "winner_counts_by_bacc.csv"), index=False)

    if all_effects:
        effects = pd.concat(all_effects, ignore_index=True)
        effects.to_csv(os.path.join(args.outdir, "all_coordinate_effects.csv"), index=False)

        stable = (
            effects.groupby(["ell", "coordinate"])
            .agg(
                mean_cohens_d=("cohens_d", "mean"),
                mean_abs_cohens_d=("abs_cohens_d", "mean"),
                mean_p_value=("p_value", "mean"),
                selection_count=("coordinate", "count"),
            )
            .reset_index()
            .sort_values(["selection_count", "mean_abs_cohens_d"], ascending=False)
        )

        stable.to_csv(os.path.join(args.outdir, "stable_coordinates_all_orders.csv"), index=False)

        for ell in ["M1", "M2", "M3"]:
            stable[stable["ell"] == ell].to_csv(
                os.path.join(args.outdir, f"stable_coordinates_{ell}.csv"),
                index=False
            )

    metadata = {
        "csv": args.csv,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "n_subjects": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df[args.label_col].value_counts().to_dict().items()},
        "roi_set": "C2_MTL_temporal",
        "c2_keys": C2_KEYS,
        "n_c2_features": len(c2_cols),
        "n_repeats": args.n_repeats,
        "n_splits": args.n_splits,
        "n_null_posets": args.n_null_posets,
        "min_nulls_for_pvalue": args.min_nulls_for_pvalue,
        "models": args.models,
        "null_models": args.null_models,
        "topk_m1": args.topk_m1,
        "topk_m2": args.topk_m2,
        "topk_m3": args.topk_m3,
    }

    with open(os.path.join(args.outdir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nObserved repeated-CV summary:")
    print(observed_summary.to_string(index=False))

    print("\nWinner counts by AUC:")
    print(winner_auc_counts.to_string(index=False))

    print("\nWinner counts by bACC:")
    print(winner_bacc_counts.to_string(index=False))

    print("\nEmpirical p-values:")
    print(pvals.to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--id_col", default="subject_id")
    parser.add_argument("--label_col", default="label_str")
    parser.add_argument("--outdir", default="c2_repeated_cv_stability")

    parser.add_argument("--n_repeats", type=int, default=100)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_null_posets", type=int, default=100)
    parser.add_argument("--min_nulls_for_pvalue", type=int, default=20)

    parser.add_argument("--ells", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--topk_m1", type=int, default=0)
    parser.add_argument("--topk_m2", type=int, default=200)
    parser.add_argument("--topk_m3", type=int, default=300)

    parser.add_argument("--models", default="linear_svm,logreg_l2,logreg_l1")
    parser.add_argument("--null_models", default="linear_svm,logreg_l2,logreg_l1")

    parser.add_argument("--variance_threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
