#!/usr/bin/env python3

import argparse
import itertools
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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


def infer_feature_columns(df, id_col, label_col, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    exclude_cols.update([id_col, label_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]


def clean_feature_columns(feature_cols, exclude_keywords=None):
    default_keywords = [
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
        default_keywords.extend(exclude_keywords)

    cleaned, removed = [], []

    for c in feature_cols:
        low = c.lower()
        if any(k.lower() in low for k in default_keywords):
            removed.append(c)
        else:
            cleaned.append(c)

    return cleaned, removed


def make_positive(X, eps=1e-8):
    X = np.asarray(X, dtype=np.float64)
    min_per_col = np.nanmin(X, axis=0, keepdims=True)
    return X - np.minimum(min_per_col, 0.0) + eps


def normalize_rows_to_prob(X, eps=1e-12):
    X = np.maximum(np.asarray(X, dtype=np.float64), eps)
    return X / np.maximum(X.sum(axis=1, keepdims=True), eps)


def metric_dict(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "auc": roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan,
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "acc": accuracy_score(y_true, y_pred),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out["sens_mci"] = tp / max(tp + fn, 1)
    out["spec_hc"] = tn / max(tn + fp, 1)

    return out


def bootstrap_metrics(y_true, y_score, n_bootstrap=200, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        rows.append(metric_dict(y_true[idx], y_score[idx]))

    return pd.DataFrame(rows)


# -----------------------------
# Generic tensor poset
# -----------------------------

def pad_or_truncate_features(X, target_dim):
    n, d = X.shape

    if d == target_dim:
        return X

    if d > target_dim:
        return X[:, :target_dim]

    pad = np.zeros((n, target_dim - d), dtype=X.dtype)
    return np.concatenate([X, pad], axis=1)


def theta_from_tensor_probs(P, shape, eps=1e-12):
    n = P.shape[0]
    logP = np.log(np.maximum(P, eps)).reshape((n,) + shape)
    theta = logP.copy()

    for axis in range(1, len(shape) + 1):
        theta = np.diff(
            theta,
            n=1,
            axis=axis,
            prepend=np.take(theta, [0], axis=axis) * 0.0,
        )

    return theta.reshape(n, -1)


def body_order_mask(shape, ell):
    idxs = np.indices(shape)
    nonzero_count = np.sum(idxs != 0, axis=0)
    return (nonzero_count == ell).reshape(-1)


def generic_poset_features(X_train, X_test, ell, fixed_shape):
    target_dim = int(np.prod(fixed_shape))

    Xtr = pad_or_truncate_features(X_train, target_dim)
    Xte = pad_or_truncate_features(X_test, target_dim)

    Ptr = normalize_rows_to_prob(make_positive(Xtr))
    Pte = normalize_rows_to_prob(make_positive(Xte))

    th_tr = theta_from_tensor_probs(Ptr, fixed_shape)
    th_te = theta_from_tensor_probs(Pte, fixed_shape)

    mask = body_order_mask(fixed_shape, ell)

    if mask.sum() == 0:
        raise ValueError(f"No M{ell} parameters for shape {fixed_shape}")

    return th_tr[:, mask], th_te[:, mask], {
        "shape": fixed_shape,
        "dim": int(mask.sum()),
    }


# -----------------------------
# Anatomical poset
# -----------------------------

ANATOMICAL_GROUP_KEYWORDS = {
    "hippocampus": ["hippocampus", "hippo"],
    "entorhinal": ["entorhinal"],
    "parahippocampal": ["parahippocampal", "para_hipp", "parahip"],
    "amygdala": ["amygdala"],
    "temporal": ["temporal", "bankssts", "fusiform"],
    "frontal": ["frontal", "precentral", "pars", "orbitofrontal"],
    "parietal": ["parietal", "postcentral", "precuneus", "supramarginal"],
    "cingulate": ["cingulate"],
    "occipital": ["occipital", "cuneus", "lingual", "pericalcarine"],
    "ventricle": ["ventricle", "ventric", "lat_vent", "inf_lat_vent"],
    "white_matter": ["white_matter", "wm_", "hypointensities"],
    "corpus_callosum": ["cc_", "corpus"],
    "cerebellum": ["cerebellum"],
    "subcortical": [
        "thalamus",
        "caudate",
        "putamen",
        "pallidum",
        "accumbens",
        "ventraldc",
        "brain_stem",
    ],
}


def assign_anatomical_groups(feature_names):
    groups = []

    for f in feature_names:
        low = f.lower()
        assigned = None

        for group, keys in ANATOMICAL_GROUP_KEYWORDS.items():
            if any(k.lower() in low for k in keys):
                assigned = group
                break

        groups.append(assigned)

    return np.array(groups, dtype=object)


def anatomical_summary_features(X, feature_names):
    groups = assign_anatomical_groups(feature_names)

    ordered_groups = [
        "hippocampus",
        "entorhinal",
        "parahippocampal",
        "amygdala",
        "temporal",
        "frontal",
        "parietal",
        "cingulate",
        "occipital",
        "ventricle",
        "white_matter",
        "corpus_callosum",
        "cerebellum",
        "subcortical",
    ]

    unique_groups = [g for g in ordered_groups if g in groups]

    feats, names = [], []

    for g in unique_groups:
        idx = np.where(groups == g)[0]

        if len(idx) == 0:
            continue

        Xg = X[:, idx]

        feats.append(np.nanmean(Xg, axis=1))
        names.append(f"{g}_mean")

        feats.append(np.nanstd(Xg, axis=1))
        names.append(f"{g}_std")

        feats.append(np.nanmin(Xg, axis=1))
        names.append(f"{g}_min")

        feats.append(np.nanmax(Xg, axis=1))
        names.append(f"{g}_max")

    if not feats:
        raise ValueError("No anatomical groups were detected after removing 'other'.")

    return np.vstack(feats).T, names, unique_groups


def chain_poset_theta(P, eps=1e-12):
    logP = np.log(np.maximum(P, eps))
    return np.diff(logP, axis=1, prepend=np.zeros((logP.shape[0], 1)))


def anatomical_interaction_features(theta, ell):
    n, d = theta.shape

    if ell == 1:
        return theta, [(i,) for i in range(d)]

    combos = list(itertools.combinations(range(d), ell))
    Xell = np.zeros((n, len(combos)), dtype=np.float64)

    for j, combo in enumerate(combos):
        Xell[:, j] = np.prod(theta[:, combo], axis=1)

    return Xell, combos


def anatomical_poset_features(X_train, X_test, feature_names, ell):
    Xtr_sum, anat_names, groups = anatomical_summary_features(X_train, feature_names)
    Xte_sum, _, _ = anatomical_summary_features(X_test, feature_names)

    Ptr = normalize_rows_to_prob(make_positive(Xtr_sum))
    Pte = normalize_rows_to_prob(make_positive(Xte_sum))

    th_tr = chain_poset_theta(Ptr)
    th_te = chain_poset_theta(Pte)

    Xtr_ell, combo_names = anatomical_interaction_features(th_tr, ell)
    Xte_ell, _ = anatomical_interaction_features(th_te, ell)

    readable_names = [
        " x ".join([anat_names[i] for i in combo])
        for combo in combo_names
    ]

    return Xtr_ell, Xte_ell, {
        "anat_feature_names": anat_names,
        "groups": groups,
        "interaction_names": readable_names,
        "dim": Xtr_ell.shape[1],
    }


def shuffled_feature_names(feature_names, rng):
    shuffled = list(feature_names)
    rng.shuffle(shuffled)
    return shuffled


# -----------------------------
# Feature selection and effects
# -----------------------------

def select_topk_train_only(X_train, X_test, y_train, feature_names, k):
    if k is None or k <= 0 or X_train.shape[1] <= k:
        return X_train, X_test, feature_names, None

    selector = SelectKBest(score_func=f_classif, k=k)
    Xtr = selector.fit_transform(X_train, y_train)
    Xte = selector.transform(X_test)

    keep = selector.get_support()
    kept_names = [n for n, keep_i in zip(feature_names, keep) if keep_i]

    return Xtr, Xte, kept_names, selector


def compute_group_effects(X_train, y_train, feature_names):
    rows = []

    for j, name in enumerate(feature_names):
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
            "interaction": name,
            "hc_mean": hc_mean,
            "mci_mean": mci_mean,
            "mci_minus_hc": mci_mean - hc_mean,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
            "p_value": pval,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Models
# -----------------------------

def get_classifiers(seed=42):
    models = {
        "logreg": LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=seed,
        ),
        "rf": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
        )
    except Exception:
        pass

    return models


def filter_classifiers(classifiers, names):
    if names == "all":
        return classifiers

    wanted = [x.strip() for x in names.split(",")]
    return {k: v for k, v in classifiers.items() if k in wanted}


def predict_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))

    return model.predict(X)


def fit_eval_feature_set(X_train, X_test, y_train, y_test, feature_set_name, classifiers, seed):
    results, preds = [], {}

    for clf_name, clf in classifiers.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clone(clf)),
        ])

        pipe.fit(X_train, y_train)
        y_score = predict_scores(pipe, X_test)
        md = metric_dict(y_test, y_score)

        results.append({
            "feature_set": feature_set_name,
            "classifier": clf_name,
            **md,
        })

        preds[clf_name] = y_score

    return results, preds


# -----------------------------
# Empirical p-values
# -----------------------------

def empirical_pvalue(observed, null_values, higher_is_better=True):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[~np.isnan(null_values)]

    if len(null_values) == 0 or np.isnan(observed):
        return np.nan

    if higher_is_better:
        return (1.0 + np.sum(null_values >= observed)) / (len(null_values) + 1.0)

    return (1.0 + np.sum(null_values <= observed)) / (len(null_values) + 1.0)


def compute_null_pvalues(summary):
    rows = []

    obs = summary[summary["feature_set"].str.startswith("anatomical_poset_M")].copy()
    nulls = summary[summary["feature_set"].str.startswith("null_shuffle_")].copy()

    for _, r in obs.iterrows():
        fs = r["feature_set"]
        clf = r["classifier"]

        if "_M1" in fs:
            ell_tag = "M1"
        elif "_M2" in fs:
            ell_tag = "M2"
        elif "_M3" in fs:
            ell_tag = "M3"
        else:
            continue

        matching_nulls = nulls[
            (nulls["classifier"] == clf)
            & (nulls["feature_set"].str.contains(f"_{ell_tag}"))
        ]

        if len(matching_nulls) == 0:
            continue

        rows.append({
            "observed_feature_set": fs,
            "classifier": clf,
            "ell": ell_tag,
            "observed_auc": r["auc_mean"],
            "null_auc_mean": matching_nulls["auc_mean"].mean(),
            "null_auc_std": matching_nulls["auc_mean"].std(),
            "empirical_p_auc": empirical_pvalue(r["auc_mean"], matching_nulls["auc_mean"], True),
            "observed_bacc": r["bacc_mean"],
            "null_bacc_mean": matching_nulls["bacc_mean"].mean(),
            "null_bacc_std": matching_nulls["bacc_mean"].std(),
            "empirical_p_bacc": empirical_pvalue(r["bacc_mean"], matching_nulls["bacc_mean"], True),
            "observed_sens_mci": r["sens_mci_mean"],
            "null_sens_mci_mean": matching_nulls["sens_mci_mean"].mean(),
            "null_sens_mci_std": matching_nulls["sens_mci_mean"].std(),
            "empirical_p_sens_mci": empirical_pvalue(r["sens_mci_mean"], matching_nulls["sens_mci_mean"], True),
            "n_nulls": len(matching_nulls),
        })

    return pd.DataFrame(rows).sort_values(["empirical_p_auc", "empirical_p_bacc"])


# -----------------------------
# Figures
# -----------------------------

def save_barh(df, label_col, value_col, title, xlabel, path, top_n=25):
    if df.empty:
        return

    top = df.head(top_n).copy()

    plt.figure(figsize=(11, max(4, 0.32 * len(top))))
    plt.barh(top[label_col][::-1], top[value_col][::-1])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_summary_figures(summary, pvals, outdir):
    observed = summary[summary["is_null"] == False].copy()

    if not observed.empty:
        top_auc = observed.sort_values("auc_mean", ascending=False).head(20)
        save_barh(
            top_auc,
            "feature_set",
            "auc_mean",
            "Observed representations ranked by AUC",
            "Mean AUC across folds",
            os.path.join(outdir, "fig_observed_auc_ranking.png"),
            top_n=20,
        )

        top_bacc = observed.sort_values("bacc_mean", ascending=False).head(20)
        save_barh(
            top_bacc,
            "feature_set",
            "bacc_mean",
            "Observed representations ranked by balanced accuracy",
            "Mean balanced accuracy across folds",
            os.path.join(outdir, "fig_observed_bacc_ranking.png"),
            top_n=20,
        )

    if not pvals.empty:
        pvals_plot = pvals.copy()
        pvals_plot["comparison"] = (
            pvals_plot["observed_feature_set"]
            + " / "
            + pvals_plot["classifier"]
        )

        pvals_plot = pvals_plot.sort_values("empirical_p_auc")
        save_barh(
            pvals_plot,
            "comparison",
            "empirical_p_auc",
            "Empirical p-values: anatomical vs shuffled null",
            "Empirical p-value for AUC",
            os.path.join(outdir, "fig_empirical_pvalues_auc.png"),
            top_n=20,
        )


def make_null_distribution_figures(summary, outdir):
    observed = summary[summary["is_null"] == False].copy()
    nulls = summary[summary["is_null"] == True].copy()

    for ell in ["M1", "M2", "M3"]:
        obs_ell = observed[
            observed["feature_set"].str.contains(f"_{ell}")
            & observed["classifier"].eq("svm_rbf")
        ]

        null_ell = nulls[
            nulls["feature_set"].str.contains(f"_{ell}")
            & nulls["classifier"].eq("svm_rbf")
        ]

        if obs_ell.empty or null_ell.empty:
            continue

        obs_auc = obs_ell.iloc[0]["auc_mean"]
        obs_bacc = obs_ell.iloc[0]["bacc_mean"]

        plt.figure(figsize=(8, 5))
        plt.hist(null_ell["auc_mean"], bins=20, alpha=0.8)
        plt.axvline(obs_auc, linestyle="--", linewidth=2)
        plt.xlabel("Null shuffled-poset AUC")
        plt.ylabel("Count")
        plt.title(f"{ell}: anatomical AUC vs shuffled null")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"fig_null_distribution_auc_{ell}.png"), dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.hist(null_ell["bacc_mean"], bins=20, alpha=0.8)
        plt.axvline(obs_bacc, linestyle="--", linewidth=2)
        plt.xlabel("Null shuffled-poset balanced accuracy")
        plt.ylabel("Count")
        plt.title(f"{ell}: anatomical balanced accuracy vs shuffled null")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"fig_null_distribution_bacc_{ell}.png"), dpi=300)
        plt.close()


# -----------------------------
# Main
# -----------------------------

def run_experiments(args):
    ensure_dir(args.outdir)

    fixed_shape = tuple(int(x) for x in args.generic_shape.split(","))
    topk_by_ell = {
        1: args.topk_m1,
        2: args.topk_m2,
        3: args.topk_m3,
    }

    df = pd.read_csv(args.csv).replace([np.inf, -np.inf], np.nan)

    df[args.label_col] = map_labels(df[args.label_col])
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)

    feature_cols = (
        args.feature_cols.split(",")
        if args.feature_cols
        else infer_feature_columns(
            df,
            args.id_col,
            args.label_col,
            exclude_cols=args.exclude_cols.split(",") if args.exclude_cols else [],
        )
    )

    feature_cols, removed_cols = clean_feature_columns(
        feature_cols,
        exclude_keywords=args.exclude_keywords.split(",") if args.exclude_keywords else [],
    )

    print(f"Loaded {len(df)} subjects")
    print(f"Labels: {df[args.label_col].value_counts().to_dict()}")
    print(f"Using {len(feature_cols)} numeric ROI features after metadata/QC exclusion")
    print(f"Removed {len(removed_cols)} metadata/QC columns")
    print("Removed vague anatomical bucket: other")
    print(f"Fixed generic tensor shape: {fixed_shape}")
    print(f"Random/shuffled null posets: {args.n_null_posets}")
    print(f"Null classifiers: {args.null_classifiers}")

    X = df[feature_cols].values.astype(np.float64)
    y = df[args.label_col].values.astype(int)

    ids = (
        df[args.id_col].astype(str).values
        if args.id_col in df.columns
        else np.arange(len(df)).astype(str)
    )

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    full_classifiers = get_classifiers(args.seed)
    observed_classifiers = filter_classifiers(full_classifiers, args.classifiers)
    null_classifiers = filter_classifiers(full_classifiers, args.null_classifiers)

    all_rows = []
    all_predictions = []
    saved_anat_interactions = defaultdict(list)
    fold_feature_logs = []

    rng_master = np.random.default_rng(args.seed)
    null_seeds = rng_master.integers(0, 10_000_000, size=args.n_null_posets)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\nFold {fold}/{args.n_splits}")

        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_imp = imputer.fit_transform(X_train_raw)
        X_test_imp = imputer.transform(X_test_raw)

        vt = VarianceThreshold(threshold=args.variance_threshold)
        X_train_imp = vt.fit_transform(X_train_imp)
        X_test_imp = vt.transform(X_test_imp)

        selected_feature_cols = [
            c for c, keep in zip(feature_cols, vt.get_support())
            if keep
        ]

        print(f"  Features after variance filter: {len(selected_feature_cols)}")

        fold_feature_logs.append({
            "fold": fold,
            "n_features_before_variance_filter": len(feature_cols),
            "n_features_after_variance_filter": len(selected_feature_cols),
        })

        fold_sets = {}

        fold_sets["raw_roi"] = (
            X_train_imp,
            X_test_imp,
            {"dim": X_train_imp.shape[1]},
        )

        n_pca = min(args.pca_components, X_train_imp.shape[1], X_train_imp.shape[0] - 1)

        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca)),
        ])

        Xtr_pca = pca_pipe.fit_transform(X_train_imp)
        Xte_pca = pca_pipe.transform(X_test_imp)

        fold_sets[f"pca_{Xtr_pca.shape[1]}"] = (
            Xtr_pca,
            Xte_pca,
            {"dim": Xtr_pca.shape[1]},
        )

        for ell in args.ells:
            try:
                Xtr_g, Xte_g, meta = generic_poset_features(
                    X_train_imp,
                    X_test_imp,
                    ell,
                    fixed_shape,
                )

                fold_sets[f"generic_poset_M{ell}"] = (
                    Xtr_g,
                    Xte_g,
                    meta,
                )

                print(f"  generic M{ell}: dim={meta['dim']}")

            except Exception as e:
                print(f"  Skipping generic M{ell}: {e}")

        for ell in args.ells:
            try:
                Xtr_a, Xte_a, meta = anatomical_poset_features(
                    X_train_imp,
                    X_test_imp,
                    selected_feature_cols,
                    ell,
                )

                original_dim = Xtr_a.shape[1]
                names = meta["interaction_names"]

                Xtr_a, Xte_a, kept_names, selector = select_topk_train_only(
                    Xtr_a,
                    Xte_a,
                    y_train,
                    names,
                    topk_by_ell.get(ell),
                )

                fs_name = f"anatomical_poset_M{ell}"

                if selector is not None:
                    fs_name += f"_top{Xtr_a.shape[1]}"

                fold_sets[fs_name] = (
                    Xtr_a,
                    Xte_a,
                    {
                        **meta,
                        "dim_original": original_dim,
                        "dim": Xtr_a.shape[1],
                    },
                )

                print(
                    f"  anatomical M{ell}: "
                    f"original_dim={original_dim} used_dim={Xtr_a.shape[1]}"
                )

                effect_df = compute_group_effects(Xtr_a, y_train, kept_names)
                effect_df["fold"] = fold
                effect_df["ell"] = f"M{ell}"

                for _, rr in effect_df.iterrows():
                    saved_anat_interactions[f"M{ell}"].append(rr.to_dict())

            except Exception as e:
                print(f"  Skipping anatomical M{ell}: {e}")

        for null_i, null_seed in enumerate(null_seeds):
            rng = np.random.default_rng(int(null_seed) + fold)
            shuffled_names = shuffled_feature_names(selected_feature_cols, rng)

            for ell in args.ells:
                try:
                    Xtr_n, Xte_n, meta_n = anatomical_poset_features(
                        X_train_imp,
                        X_test_imp,
                        shuffled_names,
                        ell,
                    )

                    names_n = meta_n["interaction_names"]

                    Xtr_n, Xte_n, kept_names_n, selector_n = select_topk_train_only(
                        Xtr_n,
                        Xte_n,
                        y_train,
                        names_n,
                        topk_by_ell.get(ell),
                    )

                    fs_name = f"null_shuffle_s{null_i:03d}_M{ell}"

                    if selector_n is not None:
                        fs_name += f"_top{Xtr_n.shape[1]}"

                    fold_sets[fs_name] = (
                        Xtr_n,
                        Xte_n,
                        {
                            "dim": Xtr_n.shape[1],
                            "null_seed": int(null_seed),
                        },
                    )

                except Exception as e:
                    print(f"  Skipping null {null_i} M{ell}: {e}")

        for fs_name, (Xtr_fs, Xte_fs, meta) in fold_sets.items():
            is_null = fs_name.startswith("null_shuffle_")
            classifiers = null_classifiers if is_null else observed_classifiers

            rows, preds = fit_eval_feature_set(
                Xtr_fs,
                Xte_fs,
                y_train,
                y_test,
                fs_name,
                classifiers,
                args.seed + fold,
            )

            for row in rows:
                row.update({
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "dim": Xtr_fs.shape[1],
                    "is_null": is_null,
                })
                all_rows.append(row)

            for clf_name, y_score in preds.items():
                for sid, yt, ys in zip(ids[test_idx], y_test, y_score):
                    all_predictions.append({
                        "fold": fold,
                        "subject_id": sid,
                        "feature_set": fs_name,
                        "classifier": clf_name,
                        "y_true": int(yt),
                        "y_score": float(ys),
                        "is_null": is_null,
                    })

    results = pd.DataFrame(all_rows)
    preds = pd.DataFrame(all_predictions)

    results_path = os.path.join(args.outdir, "fold_metrics.csv")
    preds_path = os.path.join(args.outdir, "test_predictions.csv")
    feature_log_path = os.path.join(args.outdir, "fold_feature_filter_log.csv")

    results.to_csv(results_path, index=False)
    preds.to_csv(preds_path, index=False)
    pd.DataFrame(fold_feature_logs).to_csv(feature_log_path, index=False)

    summary = (
        results.groupby(["feature_set", "classifier", "is_null"])
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            bacc_mean=("bacc", "mean"),
            bacc_std=("bacc", "std"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            sens_mci_mean=("sens_mci", "mean"),
            spec_hc_mean=("spec_hc", "mean"),
            dim_mean=("dim", "mean"),
        )
        .reset_index()
        .sort_values(
            ["is_null", "auc_mean", "bacc_mean"],
            ascending=[True, False, False],
        )
    )

    summary_path = os.path.join(args.outdir, "summary_metrics.csv")
    summary.to_csv(summary_path, index=False)

    observed_summary = summary[summary["is_null"] == False].copy()
    null_summary = summary[summary["is_null"] == True].copy()

    observed_summary.to_csv(
        os.path.join(args.outdir, "observed_summary_metrics.csv"),
        index=False,
    )

    null_summary.to_csv(
        os.path.join(args.outdir, "null_shuffle_summary_metrics.csv"),
        index=False,
    )

    pvals = compute_null_pvalues(summary)
    pvals_path = os.path.join(
        args.outdir,
        "empirical_pvalues_anatomical_vs_shuffled_null.csv",
    )
    pvals.to_csv(pvals_path, index=False)

    boot_rows = []
    observed_preds = preds[preds["is_null"] == False].copy()

    for (fs, clf), g in observed_preds.groupby(["feature_set", "classifier"]):
        y_true = g["y_true"].values
        y_score = g["y_score"].values

        if len(np.unique(y_true)) < 2:
            continue

        bdf = bootstrap_metrics(y_true, y_score, args.n_bootstrap, args.seed)

        if len(bdf) == 0:
            continue

        boot_rows.append({
            "feature_set": fs,
            "classifier": clf,
            "auc_boot_mean": bdf["auc"].mean(),
            "auc_boot_std": bdf["auc"].std(),
            "auc_boot_ci_low": bdf["auc"].quantile(0.025),
            "auc_boot_ci_high": bdf["auc"].quantile(0.975),
            "bacc_boot_mean": bdf["bacc"].mean(),
            "bacc_boot_std": bdf["bacc"].std(),
            "bacc_boot_ci_low": bdf["bacc"].quantile(0.025),
            "bacc_boot_ci_high": bdf["bacc"].quantile(0.975),
        })

    boot = (
        pd.DataFrame(boot_rows)
        .sort_values(["auc_boot_mean", "bacc_boot_mean"], ascending=False)
    )

    boot_path = os.path.join(args.outdir, "bootstrap_stability_observed.csv")
    boot.to_csv(boot_path, index=False)

    for key, rows in saved_anat_interactions.items():
        if not rows:
            continue

        idf = pd.DataFrame(rows)

        all_path = os.path.join(args.outdir, f"all_fold_disease_coordinates_{key}.csv")
        idf.to_csv(all_path, index=False)

        agg = (
            idf.groupby("interaction")
            .agg(
                mean_cohens_d=("cohens_d", "mean"),
                mean_abs_cohens_d=("abs_cohens_d", "mean"),
                mean_mci_minus_hc=("mci_minus_hc", "mean"),
                mean_p_value=("p_value", "mean"),
                selection_count=("interaction", "count"),
            )
            .reset_index()
            .sort_values(
                ["selection_count", "mean_abs_cohens_d"],
                ascending=False,
            )
        )

        out_path = os.path.join(args.outdir, f"top_disease_coordinates_{key}.csv")
        agg.to_csv(out_path, index=False)

        save_barh(
            agg,
            "interaction",
            "mean_abs_cohens_d",
            f"Top HC-MCI disease coordinates {key}",
            "Mean absolute Cohen's d across folds",
            os.path.join(args.outdir, f"fig_top_disease_coordinates_{key}.png"),
            top_n=args.top_interactions,
        )

    make_summary_figures(summary, pvals, args.outdir)
    make_null_distribution_figures(summary, args.outdir)

    metadata = {
        "csv": args.csv,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "n_subjects": int(len(df)),
        "label_counts": {
            str(k): int(v)
            for k, v in df[args.label_col].value_counts().to_dict().items()
        },
        "n_features_after_metadata_exclusion": int(len(feature_cols)),
        "removed_metadata_qc_columns": removed_cols,
        "removed_anatomical_bucket": "other",
        "generic_shape": fixed_shape,
        "n_null_posets": args.n_null_posets,
        "null_definition": "random permutation of anatomical feature names while keeping data matrix fixed",
        "topk_m1": args.topk_m1,
        "topk_m2": args.topk_m2,
        "topk_m3": args.topk_m3,
        "ells": args.ells,
        "classifiers": list(observed_classifiers.keys()),
        "null_classifiers": list(null_classifiers.keys()),
        "saved_figures": [
            "fig_observed_auc_ranking.png",
            "fig_observed_bacc_ranking.png",
            "fig_empirical_pvalues_auc.png",
            "fig_null_distribution_auc_M1.png",
            "fig_null_distribution_auc_M2.png",
            "fig_null_distribution_auc_M3.png",
            "fig_null_distribution_bacc_M1.png",
            "fig_null_distribution_bacc_M2.png",
            "fig_null_distribution_bacc_M3.png",
            "fig_top_disease_coordinates_M1.png",
            "fig_top_disease_coordinates_M2.png",
            "fig_top_disease_coordinates_M3.png",
        ],
    }

    metadata_path = os.path.join(args.outdir, "experiment_metadata.json")

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {pvals_path}")
    print(f"  {boot_path}")
    print(f"  {results_path}")
    print(f"  {preds_path}")
    print(f"  {metadata_path}")

    print("\nTop observed summary:")
    print(observed_summary.head(25).to_string(index=False))

    print("\nEmpirical p-values:")
    print(pvals.head(20).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--id_col", default="subject_id")
    parser.add_argument("--label_col", default="label_str")
    parser.add_argument("--feature_cols", default=None)
    parser.add_argument("--exclude_cols", default=None)
    parser.add_argument("--exclude_keywords", default=None)

    parser.add_argument("--outdir", default="poset_hc_mci_results_no_other_nulls")

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--ells", type=int, nargs="+", default=[1, 2, 3])

    parser.add_argument("--generic_shape", default="8,8,8")

    parser.add_argument("--topk_m1", type=int, default=0)
    parser.add_argument("--topk_m2", type=int, default=200)
    parser.add_argument("--topk_m3", type=int, default=300)

    parser.add_argument("--n_null_posets", type=int, default=100)
    parser.add_argument("--classifiers", default="all")
    parser.add_argument("--null_classifiers", default="svm_rbf")

    parser.add_argument("--top_interactions", type=int, default=30)
    parser.add_argument("--variance_threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)