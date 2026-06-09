import argparse
import itertools
import json
import os
import re
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
        "HC": 0, "CN": 0, "CONTROL": 0, "NORMAL": 0,
        "COGNITIVELY NORMAL": 0,
        "MCI": 1, "EMCI": 1, "LMCI": 1, "SMC": 1,
    }
    return y.astype(str).str.upper().str.strip().map(mapping)


def infer_feature_columns(df, id_col, label_col, exclude_cols=None):
    exclude_cols = set(exclude_cols or [])
    exclude_cols.update([id_col, label_col])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]


def clean_feature_columns(feature_cols, exclude_keywords=None):
    default_keywords = [
        "has_", "label", "source", "subject", "ptid", "rid",
        "cdglobal", "old_label", "n_extracted", "diagnosis", "dx",
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


# ============================================================
# ROI-set definitions
# ============================================================

ROI_SETS = {
    "C1_MTL_core": [
        "hippocampus",
        "entorhinal",
        "parahippocampal",
        "amygdala",
    ],
    "C2_MTL_temporal": [
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
    ],
    "C3_memory_default_mode": [
        "hippocampus",
        "entorhinal",
        "parahippocampal",
        "amygdala",
        "posteriorcingulate",
        "isthmuscingulate",
        "precuneus",
        "inferiorparietal",
        "supramarginal",
        "temporal",
        "fusiform",
    ],
    "C4_disconnection": [
        "cc_",
        "corpus",
        "white_matter",
        "wm_",
        "hypointensities",
        "ventricle",
        "ventric",
        "cingulate",
        "temporal",
        "thalamus",
        "caudate",
        "putamen",
        "pallidum",
        "accumbens",
        "ventraldc",
    ],
    "C5_full_disease_network": [
        "hippocampus",
        "entorhinal",
        "parahippocampal",
        "amygdala",
        "temporal",
        "fusiform",
        "bankssts",
        "posteriorcingulate",
        "isthmuscingulate",
        "precuneus",
        "inferiorparietal",
        "supramarginal",
        "cc_",
        "corpus",
        "white_matter",
        "wm_",
        "hypointensities",
        "ventricle",
        "ventric",
        "thalamus",
        "caudate",
        "putamen",
        "pallidum",
        "accumbens",
        "ventraldc",
    ],
}


def select_roi_set_features(feature_cols, roi_set_name):
    keys = ROI_SETS[roi_set_name]
    selected = []
    for c in feature_cols:
        low = c.lower()
        if any(k.lower() in low for k in keys):
            selected.append(c)
    return selected


def add_asymmetry_features(df, feature_cols):
    """
    Adds asymmetry features for left/right pairs when names match common FreeSurfer patterns.
    """
    out = df.copy()
    added = []

    colset = set(feature_cols)

    for c in feature_cols:
        candidates = []

        if "left_" in c.lower():
            candidates.append(re.sub("left_", "Right_", c, flags=re.IGNORECASE))
            candidates.append(re.sub("Left_", "Right_", c))
        if "right_" in c.lower():
            continue

        if c.startswith("lh_"):
            candidates.append("rh_" + c[3:])
        if c.startswith("rh_"):
            continue

        right = None
        for cand in candidates:
            if cand in colset:
                right = cand
                break

        if right is None:
            continue

        left = c
        base = safe_name(left.replace("Left_", "").replace("left_", "").replace("lh_", ""))

        l = out[left].astype(float)
        r = out[right].astype(float)

        mean_name = f"asym_mean_{base}"
        diff_name = f"asym_diff_LminusR_{base}"
        ratio_name = f"asym_ratio_LminusR_over_sum_{base}"
        min_name = f"asym_min_{base}"
        max_name = f"asym_max_{base}"

        out[mean_name] = (l + r) / 2.0
        out[diff_name] = l - r
        out[ratio_name] = (l - r) / (np.abs(l) + np.abs(r) + 1e-8)
        out[min_name] = np.minimum(l, r)
        out[max_name] = np.maximum(l, r)

        added.extend([mean_name, diff_name, ratio_name, min_name, max_name])

    return out, added


# ============================================================
# Poset coordinate construction
# ============================================================

def chain_poset_theta(P, eps=1e-12):
    logP = np.log(np.maximum(P, eps))
    return np.diff(logP, axis=1, prepend=np.zeros((logP.shape[0], 1)))


def interaction_features(theta, ell):
    n, d = theta.shape

    if ell == 1:
        return theta, [(i,) for i in range(d)]

    combos = list(itertools.combinations(range(d), ell))
    Xell = np.zeros((n, len(combos)), dtype=np.float64)

    for j, combo in enumerate(combos):
        Xell[:, j] = np.prod(theta[:, combo], axis=1)

    return Xell, combos


def roi_level_poset_features(X_train, X_test, feature_names, ell):
    """
    ROI-level poset:
    no coarse group summaries. each selected ROI feature is a node.
    """
    Ptr = normalize_rows_to_prob(make_positive(X_train))
    Pte = normalize_rows_to_prob(make_positive(X_test))

    th_tr = chain_poset_theta(Ptr)
    th_te = chain_poset_theta(Pte)

    Xtr_ell, combos = interaction_features(th_tr, ell)
    Xte_ell, _ = interaction_features(th_te, ell)

    names = [
        " x ".join([feature_names[i] for i in combo])
        for combo in combos
    ]

    return Xtr_ell, Xte_ell, names


def shuffled_feature_order(X_train, X_test, feature_names, rng):
    idx = np.arange(len(feature_names))
    rng.shuffle(idx)
    return X_train[:, idx], X_test[:, idx], [feature_names[i] for i in idx]


# ============================================================
# Selection, effects, models
# ============================================================

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
            "coordinate": name,
            "hc_mean": hc_mean,
            "mci_mean": mci_mean,
            "mci_minus_hc": mci_mean - hc_mean,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
            "p_value": pval,
        })

    return pd.DataFrame(rows)


def get_classifiers(seed=42):
    models = {
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
        "linear_svm": SVC(
            kernel="linear",
            probability=True,
            class_weight="balanced",
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


def empirical_pvalue(observed, null_values):
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[~np.isnan(null_values)]
    if len(null_values) == 0 or np.isnan(observed):
        return np.nan
    return (1.0 + np.sum(null_values >= observed)) / (len(null_values) + 1.0)


def compute_null_pvalues(summary):
    rows = []

    obs = summary[
        (summary["is_null"] == False)
        & (summary["feature_set"].str.startswith("C"))
    ].copy()

    nulls = summary[summary["is_null"] == True].copy()

    for _, r in obs.iterrows():
        fs = r["feature_set"]
        clf = r["classifier"]

        roi_set = fs.split("__")[0]

        if "__M1" in fs:
            ell = "M1"
        elif "__M2" in fs:
            ell = "M2"
        elif "__M3" in fs:
            ell = "M3"
        else:
            continue

        matching = nulls[
            (nulls["classifier"] == clf)
            & (nulls["feature_set"].str.startswith(f"null_{roi_set}_"))
            & (nulls["feature_set"].str.contains(f"__{ell}"))
        ]

        if len(matching) == 0:
            continue

        rows.append({
            "observed_feature_set": fs,
            "roi_set": roi_set,
            "classifier": clf,
            "ell": ell,
            "observed_auc": r["auc_mean"],
            "null_auc_mean": matching["auc_mean"].mean(),
            "null_auc_std": matching["auc_mean"].std(),
            "empirical_p_auc": empirical_pvalue(r["auc_mean"], matching["auc_mean"]),
            "observed_bacc": r["bacc_mean"],
            "null_bacc_mean": matching["bacc_mean"].mean(),
            "null_bacc_std": matching["bacc_mean"].std(),
            "empirical_p_bacc": empirical_pvalue(r["bacc_mean"], matching["bacc_mean"]),
            "observed_sens_mci": r["sens_mci_mean"],
            "null_sens_mci_mean": matching["sens_mci_mean"].mean(),
            "null_sens_mci_std": matching["sens_mci_mean"].std(),
            "empirical_p_sens_mci": empirical_pvalue(r["sens_mci_mean"], matching["sens_mci_mean"]),
            "n_nulls": len(matching),
        })

    return pd.DataFrame(rows).sort_values(["empirical_p_auc", "empirical_p_bacc"])


# ============================================================
# Figures
# ============================================================

def save_barh(df, label_col, value_col, title, xlabel, path, top_n=25):
    if df.empty:
        return

    top = df.head(top_n).copy()
    plt.figure(figsize=(12, max(4, 0.35 * len(top))))
    plt.barh(top[label_col][::-1], top[value_col][::-1])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_figures(summary, pvals, outdir):
    observed = summary[summary["is_null"] == False].copy()
    observed["label"] = observed["feature_set"] + " / " + observed["classifier"]

    save_barh(
        observed.sort_values("auc_mean", ascending=False),
        "label",
        "auc_mean",
        "Observed ROI-level posets ranked by AUC",
        "Mean AUC across folds",
        os.path.join(outdir, "fig_observed_auc_ranking.png"),
        top_n=35,
    )

    save_barh(
        observed.sort_values("bacc_mean", ascending=False),
        "label",
        "bacc_mean",
        "Observed ROI-level posets ranked by balanced accuracy",
        "Mean balanced accuracy across folds",
        os.path.join(outdir, "fig_observed_bacc_ranking.png"),
        top_n=35,
    )

    if pvals is not None and not pvals.empty:
        pvals = pvals.copy()
        pvals["label"] = pvals["observed_feature_set"] + " / " + pvals["classifier"]

        save_barh(
            pvals.sort_values("empirical_p_auc"),
            "label",
            "empirical_p_auc",
            "Empirical p-values: ROI-level posets vs shuffled-order nulls",
            "Empirical p-value for AUC",
            os.path.join(outdir, "fig_empirical_pvalues_auc.png"),
            top_n=35,
        )


# ============================================================
# Main
# ============================================================

def run_experiments(args):
    ensure_dir(args.outdir)

    topk_by_ell = {1: args.topk_m1, 2: args.topk_m2, 3: args.topk_m3}

    roi_sets = list(ROI_SETS.keys())
    if args.roi_sets != "all":
        wanted = [x.strip() for x in args.roi_sets.split(",")]
        roi_sets = [x for x in roi_sets if x in wanted]

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

    if args.add_asymmetry:
        df, added_asym = add_asymmetry_features(df, feature_cols)
        feature_cols = feature_cols + added_asym
    else:
        added_asym = []

    print(f"Loaded {len(df)} subjects")
    print(f"Labels: {df[args.label_col].value_counts().to_dict()}")
    print(f"Using {len(feature_cols)} numeric features")
    print(f"Added asymmetry features: {len(added_asym)}")
    print(f"ROI sets: {roi_sets}")
    print(f"Null shuffled orders per ROI set: {args.n_null_posets}")

    X_all = df[feature_cols].values.astype(float)
    y = df[args.label_col].values.astype(int)
    ids = df[args.id_col].astype(str).values if args.id_col in df.columns else np.arange(len(df)).astype(str)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    full_classifiers = get_classifiers(args.seed)
    observed_classifiers = filter_classifiers(full_classifiers, args.classifiers)
    null_classifiers = filter_classifiers(full_classifiers, args.null_classifiers)

    rng_master = np.random.default_rng(args.seed)
    null_seeds = rng_master.integers(0, 10_000_000, size=args.n_null_posets)

    all_rows, all_predictions = [], []
    saved_coordinates = defaultdict(list)
    roi_set_manifest = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y), start=1):
        print(f"\nFold {fold}/{args.n_splits}")

        X_train_raw = X_all[train_idx]
        X_test_raw = X_all[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_train_imp_all = imputer.fit_transform(X_train_raw)
        X_test_imp_all = imputer.transform(X_test_raw)

        vt = VarianceThreshold(threshold=args.variance_threshold)
        X_train_imp_all = vt.fit_transform(X_train_imp_all)
        X_test_imp_all = vt.transform(X_test_imp_all)

        selected_feature_cols_all = [c for c, keep in zip(feature_cols, vt.get_support()) if keep]

        fold_sets = {}

        # Global baselines
        fold_sets["raw_roi_all"] = (
            X_train_imp_all,
            X_test_imp_all,
            {"dim": X_train_imp_all.shape[1]},
        )

        n_pca = min(args.pca_components, X_train_imp_all.shape[1], X_train_imp_all.shape[0] - 1)
        pca_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_pca)),
        ])
        Xtr_pca = pca_pipe.fit_transform(X_train_imp_all)
        Xte_pca = pca_pipe.transform(X_test_imp_all)
        fold_sets[f"pca_{Xtr_pca.shape[1]}"] = (Xtr_pca, Xte_pca, {"dim": Xtr_pca.shape[1]})

        # ROI-set posets
        for roi_set in roi_sets:
            selected_names = select_roi_set_features(selected_feature_cols_all, roi_set)
            if len(selected_names) < args.min_roi_features:
                print(f"  Skipping {roi_set}: only {len(selected_names)} features")
                continue

            idx = [selected_feature_cols_all.index(c) for c in selected_names]
            Xtr_set = X_train_imp_all[:, idx]
            Xte_set = X_test_imp_all[:, idx]

            roi_set_manifest.append({
                "fold": fold,
                "roi_set": roi_set,
                "n_features": len(selected_names),
                "features": "|".join(selected_names),
            })

            print(f"  {roi_set}: {len(selected_names)} features")

            for ell in args.ells:
                try:
                    Xtr_pos, Xte_pos, coord_names = roi_level_poset_features(
                        Xtr_set, Xte_set, selected_names, ell
                    )

                    original_dim = Xtr_pos.shape[1]

                    Xtr_pos, Xte_pos, kept_names, selector = select_topk_train_only(
                        Xtr_pos, Xte_pos, y_train, coord_names, topk_by_ell.get(ell)
                    )

                    fs_name = f"{roi_set}__M{ell}"
                    if selector is not None:
                        fs_name += f"_top{Xtr_pos.shape[1]}"

                    fold_sets[fs_name] = (
                        Xtr_pos,
                        Xte_pos,
                        {"dim": Xtr_pos.shape[1], "dim_original": original_dim},
                    )

                    effect_df = compute_group_effects(Xtr_pos, y_train, kept_names)
                    effect_df["fold"] = fold
                    effect_df["roi_set"] = roi_set
                    effect_df["ell"] = f"M{ell}"

                    for _, rr in effect_df.iterrows():
                        saved_coordinates[f"{roi_set}__M{ell}"].append(rr.to_dict())

                    print(f"    M{ell}: original_dim={original_dim} used_dim={Xtr_pos.shape[1]}")

                except Exception as e:
                    print(f"    Skipping {roi_set} M{ell}: {e}")

            # Shuffled-order nulls
            for null_i, null_seed in enumerate(null_seeds):
                rng = np.random.default_rng(int(null_seed) + fold)

                Xtr_shuf, Xte_shuf, names_shuf = shuffled_feature_order(
                    Xtr_set, Xte_set, selected_names, rng
                )

                for ell in args.ells:
                    try:
                        Xtr_null, Xte_null, null_coord_names = roi_level_poset_features(
                            Xtr_shuf, Xte_shuf, names_shuf, ell
                        )

                        Xtr_null, Xte_null, kept_null_names, selector_null = select_topk_train_only(
                            Xtr_null, Xte_null, y_train, null_coord_names, topk_by_ell.get(ell)
                        )

                        fs_name = f"null_{roi_set}_s{null_i:03d}__M{ell}"
                        if selector_null is not None:
                            fs_name += f"_top{Xtr_null.shape[1]}"

                        fold_sets[fs_name] = (
                            Xtr_null,
                            Xte_null,
                            {"dim": Xtr_null.shape[1], "null_seed": int(null_seed)},
                        )

                    except Exception:
                        pass

        # Fit all
        for fs_name, (Xtr_fs, Xte_fs, meta) in fold_sets.items():
            is_null = fs_name.startswith("null_")
            classifiers = null_classifiers if is_null else observed_classifiers

            rows, preds = fit_eval_feature_set(
                Xtr_fs, Xte_fs, y_train, y_test,
                fs_name, classifiers, args.seed + fold,
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

    results.to_csv(os.path.join(args.outdir, "fold_metrics.csv"), index=False)
    preds.to_csv(os.path.join(args.outdir, "test_predictions.csv"), index=False)
    pd.DataFrame(roi_set_manifest).to_csv(os.path.join(args.outdir, "roi_set_manifest_by_fold.csv"), index=False)

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
        .sort_values(["is_null", "auc_mean", "bacc_mean"], ascending=[True, False, False])
    )

    summary.to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)
    summary[summary["is_null"] == False].to_csv(os.path.join(args.outdir, "observed_summary_metrics.csv"), index=False)
    summary[summary["is_null"] == True].to_csv(os.path.join(args.outdir, "null_shuffle_summary_metrics.csv"), index=False)

    pvals = compute_null_pvalues(summary)
    pvals.to_csv(os.path.join(args.outdir, "empirical_pvalues_roi_posets_vs_shuffled_order.csv"), index=False)

    all_coord_rows = []

    for key, rows in saved_coordinates.items():
        if not rows:
            continue

        idf = pd.DataFrame(rows)
        idf.to_csv(os.path.join(args.outdir, f"all_fold_coordinates_{safe_name(key)}.csv"), index=False)

        agg = (
            idf.groupby(["roi_set", "ell", "coordinate"])
            .agg(
                mean_cohens_d=("cohens_d", "mean"),
                mean_abs_cohens_d=("abs_cohens_d", "mean"),
                mean_mci_minus_hc=("mci_minus_hc", "mean"),
                mean_p_value=("p_value", "mean"),
                selection_count=("coordinate", "count"),
            )
            .reset_index()
            .sort_values(["selection_count", "mean_abs_cohens_d"], ascending=False)
        )

        agg.to_csv(os.path.join(args.outdir, f"top_coordinates_{safe_name(key)}.csv"), index=False)
        all_coord_rows.append(agg)

        save_barh(
            agg,
            "coordinate",
            "mean_abs_cohens_d",
            f"Top HC-MCI coordinates: {key}",
            "Mean absolute Cohen's d",
            os.path.join(args.outdir, f"fig_top_coordinates_{safe_name(key)}.png"),
            top_n=args.top_interactions,
        )

    if all_coord_rows:
        all_coords = pd.concat(all_coord_rows, ignore_index=True)
        all_coords = all_coords.sort_values(["selection_count", "mean_abs_cohens_d"], ascending=False)
        all_coords.to_csv(os.path.join(args.outdir, "top_coordinates_all_roi_sets_all_orders.csv"), index=False)

    make_figures(summary, pvals, args.outdir)

    metadata = {
        "csv": args.csv,
        "id_col": args.id_col,
        "label_col": args.label_col,
        "n_subjects": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df[args.label_col].value_counts().to_dict().items()},
        "roi_sets": roi_sets,
        "roi_set_definitions": ROI_SETS,
        "add_asymmetry": args.add_asymmetry,
        "n_asymmetry_features_added": len(added_asym),
        "n_null_posets": args.n_null_posets,
        "null_definition": "random shuffle of ROI feature order inside same ROI set before theta/interactions",
        "topk_m1": args.topk_m1,
        "topk_m2": args.topk_m2,
        "topk_m3": args.topk_m3,
        "classifiers": list(observed_classifiers.keys()),
        "null_classifiers": list(null_classifiers.keys()),
    }

    with open(os.path.join(args.outdir, "experiment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nTop observed summary:")
    print(summary[summary["is_null"] == False].head(40).to_string(index=False))

    print("\nTop empirical p-values:")
    print(pvals.head(40).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--id_col", default="subject_id")
    parser.add_argument("--label_col", default="label_str")
    parser.add_argument("--feature_cols", default=None)
    parser.add_argument("--exclude_cols", default=None)
    parser.add_argument("--exclude_keywords", default=None)

    parser.add_argument("--outdir", default="poset_hc_mci_roi_level_posets")

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--pca_components", type=int, default=20)
    parser.add_argument("--ells", type=int, nargs="+", default=[1, 2, 3])

    parser.add_argument("--topk_m1", type=int, default=0)
    parser.add_argument("--topk_m2", type=int, default=200)
    parser.add_argument("--topk_m3", type=int, default=300)

    parser.add_argument("--n_null_posets", type=int, default=100)
    parser.add_argument("--classifiers", default="all")
    parser.add_argument("--null_classifiers", default="linear_svm,svm_rbf")

    parser.add_argument("--roi_sets", default="all")
    parser.add_argument("--add_asymmetry", action="store_true")
    parser.add_argument("--min_roi_features", type=int, default=4)

    parser.add_argument("--top_interactions", type=int, default=30)
    parser.add_argument("--variance_threshold", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiments(args)
