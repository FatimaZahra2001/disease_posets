#!/usr/bin/env python3
"""
Next-generation disease-aware PyMBA MRI poset experiment.

This runner keeps official PyMBA theta/eta coordinate computation unchanged.
The experimental novelty is in embedding phi, structured disease-stage posets,
incremental higher-order order models, stability selection, train-fold
threshold tuning, residual M3 tests, covariate correction, and controls.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from disease_aware_poset_utils import (
    EMBEDDING_DESCRIPTIONS,
    build_faces,
    build_face_tensor,
    build_node_table,
    disease_direction,
    embedding_diagnostic_row,
    fit_preprocess_train_test,
    import_official_pymba,
    motif_manifest_for_roi_set,
    official_coordinate_features,
    parse_face_from_coordinate,
    parse_values,
    probability_embedding,
)
from poset_utils import clean_feature_columns, filter_classifiers, get_classifiers, infer_feature_columns, map_labels, predict_scores


warnings.filterwarnings("ignore")
EPS = 1e-12


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_auc(y_true, score):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, score))


def binary_metrics(y_true, score, threshold):
    pred = (np.asarray(score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": safe_auc(y_true, score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity_MCI": float(tp / max(tp + fn, 1)),
        "specificity_HC": float(tn / max(tn + fp, 1)),
        "selected_threshold": float(threshold),
    }


def tune_threshold(y_train, train_score, strategy):
    if strategy == "fixed_0.5":
        return 0.5
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr = 0.5
    best_val = -np.inf
    for thr in thresholds:
        pred = (train_score >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        val = 0.5 * (sens + spec) if strategy == "train_balanced_accuracy" else sens + spec - 1.0
        if val > best_val:
            best_val = val
            best_thr = thr
    return float(best_thr)


def load_dataset(args):
    df = pd.read_csv(args.csv).replace([np.inf, -np.inf], np.nan)
    df[args.label_col] = map_labels(df[args.label_col])
    df = df.dropna(subset=[args.label_col]).copy()
    df[args.label_col] = df[args.label_col].astype(int)
    if args.subject_col in df.columns:
        df = df.sort_values(args.subject_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    feature_cols = infer_feature_columns(df, id_col=args.subject_col, label_col=args.label_col)
    feature_cols, removed = clean_feature_columns(feature_cols)
    return df, df[args.label_col].to_numpy(dtype=int), feature_cols, removed


def covariate_matrix(df, covariates, fit_idx, all_idx):
    available = [c for c in covariates if c in df.columns]
    missing = [c for c in covariates if c not in df.columns]
    if not available:
        return None, None, missing
    cov = pd.get_dummies(df[available], dummy_na=True, drop_first=True)
    cov = cov.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    X_fit = imputer.fit_transform(cov.iloc[fit_idx])
    X_all = imputer.transform(cov.iloc[all_idx])
    X_fit = np.column_stack([np.ones(X_fit.shape[0]), X_fit])
    X_all = np.column_stack([np.ones(X_all.shape[0]), X_all])
    return X_fit, X_all, missing


def covariate_correct_df(df, feature_cols, y, tr_idx, te_idx, args):
    if args.covariate_correction == "none" or not args.covariates:
        return df, []
    fit_idx = tr_idx if args.covariate_fit_group == "train_all" else tr_idx[y[tr_idx] == 0]
    all_idx = np.concatenate([tr_idx, te_idx])
    X_fit_cov, X_all_cov, missing = covariate_matrix(df, args.covariates, fit_idx, all_idx)
    warnings_out = [f"missing_covariate:{c}" for c in missing]
    if X_fit_cov is None or len(fit_idx) < 3:
        warnings_out.append("covariate_correction_skipped")
        return df, warnings_out
    out = df.copy()
    Y_fit = df.iloc[fit_idx][feature_cols].astype(float).to_numpy()
    Y_all = df.iloc[all_idx][feature_cols].astype(float).to_numpy()
    coef, *_ = np.linalg.lstsq(X_fit_cov, Y_fit, rcond=None)
    pred_all = X_all_cov @ coef
    intercept = coef[0, :]
    resid_all = Y_all - pred_all + intercept
    out.iloc[all_idx, out.columns.get_indexer(feature_cols)] = resid_all
    return out, warnings_out


def cohen_d(values, y):
    x0 = np.asarray(values)[np.asarray(y) == 0]
    x1 = np.asarray(values)[np.asarray(y) == 1]
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    pooled = ((len(x0) - 1) * np.var(x0, ddof=1) + (len(x1) - 1) * np.var(x1, ddof=1)) / max(len(x0) + len(x1) - 2, 1)
    return 0.0 if pooled <= 0 else float((np.mean(x1) - np.mean(x0)) / np.sqrt(pooled))


def build_all_order_spaces(df, feature_cols, y, tr_idx, te_idx, roi_set, embedding, backend, args, faces_override=None):
    X_nodes, node_names, node_table = build_node_table(df, feature_cols, roi_set)
    if X_nodes.shape[1] == 0:
        return None, node_table, None
    Xtr, Xte, kept_nodes, _imp, _vt = fit_preprocess_train_test(X_nodes[tr_idx], X_nodes[te_idx], node_names, args.variance_threshold)
    if not kept_nodes:
        return None, node_table, None
    kept_table = node_table[node_table["node"].isin(kept_nodes)].copy()
    directions = kept_table.set_index("node").loc[kept_nodes, "direction"].tolist()
    Ptr, Pte, Atr, Ate, stats = probability_embedding(
        Xtr, Xte, y[tr_idx], directions, embedding, args.epsilon, args.unknown_policy, min_sigma=args.min_sigma
    )
    faces = faces_override if faces_override is not None else build_faces(roi_set, kept_nodes)
    tensors_tr, specs, layout = build_face_tensor(Ptr, kept_nodes, faces, roi_set)
    tensors_te, _specs_te, _layout_te = build_face_tensor(Pte, kept_nodes, faces, roi_set)
    spaces = {}
    spec_lookup = {}
    for order_name in ["M1", "M2", "M3"]:
        Xtr_theta, names = official_coordinate_features(tensors_tr, specs.get(order_name, []), backend, coordinate="theta")
        Xte_theta, _ = official_coordinate_features(tensors_te, specs.get(order_name, []), backend, coordinate="theta")
        spaces[order_name] = {"Xtr": Xtr_theta, "Xte": Xte_theta, "names": names}
        for spec in specs.get(order_name, []):
            spec_lookup[spec["name"]] = spec
    diag = embedding_diagnostic_row(
        repeat=args._current_repeat,
        fold=args._current_fold,
        roi_set=roi_set,
        embedding=embedding,
        node_names=kept_nodes,
        A_train=Atr,
        P_train=Ptr,
        y_train=y[tr_idx],
        stats=stats,
    )
    meta = {
        "node_table": kept_table,
        "faces": faces,
        "layout": layout,
        "spec_lookup": spec_lookup,
        "embedding_diagnostic": diag,
        "burden_summary": pd.DataFrame({
            "node": kept_nodes,
            "direction": directions,
            "train_mean_burden": np.nanmean(Atr, axis=0),
            "test_mean_burden": np.nanmean(Ate, axis=0),
        }),
    }
    return spaces, kept_table, meta


def assemble_order_model(spaces, order_model):
    empty = {"Xtr": np.zeros((spaces["M1"]["Xtr"].shape[0], 0)), "Xte": np.zeros((spaces["M1"]["Xte"].shape[0], 0)), "names": []}
    if order_model == "M1":
        return spaces["M1"]
    if order_model == "M1_M2":
        keys = ["M1", "M2"]
    elif order_model == "M1_M2_M3":
        keys = ["M1", "M2", "M3"]
    elif order_model == "M3_only":
        return spaces["M3"]
    elif order_model == "residual_M3":
        X12 = np.hstack([spaces["M1"]["Xtr"], spaces["M2"]["Xtr"]])
        X12_te = np.hstack([spaces["M1"]["Xte"], spaces["M2"]["Xte"]])
        X3 = spaces["M3"]["Xtr"]
        X3_te = spaces["M3"]["Xte"]
        if X12.shape[1] == 0 or X3.shape[1] == 0:
            return empty
        ridge = Ridge(alpha=1.0)
        ridge.fit(X12, X3)
        return {"Xtr": X3 - ridge.predict(X12), "Xte": X3_te - ridge.predict(X12_te), "names": [n + "::residual_M3" for n in spaces["M3"]["names"]]}
    else:
        raise ValueError(f"Unknown order_model: {order_model}")
    return {
        "Xtr": np.hstack([spaces[k]["Xtr"] for k in keys]),
        "Xte": np.hstack([spaces[k]["Xte"] for k in keys]),
        "names": sum([spaces[k]["names"] for k in keys], []),
    }


def rank_features(X, y):
    if X.shape[1] == 0:
        return np.array([], dtype=int), np.array([]), np.array([])
    scores, pvals = f_classif(X, y)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    pvals = np.nan_to_num(pvals, nan=1.0, posinf=1.0, neginf=1.0)
    return np.argsort(scores)[::-1], scores, pvals


def select_features(space, ytr, args, base):
    n = space["Xtr"].shape[1]
    if n == 0:
        return np.array([], dtype=int), [], np.array([]), np.array([])
    order, scores, pvals = rank_features(space["Xtr"], ytr)
    if args.selection_strategy == "none":
        return np.arange(n), [], scores, pvals
    if args.selection_strategy == "f_classif":
        k = min(max(parse_values(args.inner_topk, int)), n)
        return order[:k], [], scores, pvals

    counts = np.zeros(n, dtype=float)
    rank_sums = np.zeros(n, dtype=float)
    rank_counts = np.zeros(n, dtype=float)
    inner_topk = parse_values(args.inner_topk, int)
    skf = StratifiedKFold(n_splits=min(args.inner_splits, np.bincount(ytr).min()), shuffle=True, random_state=args.random_state + base["repeat"] * 100 + base["fold"])
    total_votes = 0
    for inner_tr, _inner_va in skf.split(np.zeros(len(ytr)), ytr):
        inner_order, _s, _p = rank_features(space["Xtr"][inner_tr], ytr[inner_tr])
        rank_map = np.empty(n, dtype=float)
        rank_map[inner_order] = np.arange(1, len(inner_order) + 1)
        for k in inner_topk:
            keep = inner_order[: min(k, n)]
            counts[keep] += 1.0
            total_votes += 1
        rank_sums += rank_map
        rank_counts += 1
    freq = counts / max(total_votes, 1)
    mean_rank = rank_sums / np.maximum(rank_counts, 1)
    keep = np.where(freq >= args.stability_threshold)[0]
    if len(keep) == 0:
        keep = np.argsort(freq)[::-1][: min(max(inner_topk), n)]
    keep = keep[np.argsort(freq[keep])[::-1]]
    stability_rows = []
    for idx in keep:
        stability_rows.append({
            **base,
            "coordinate": space["names"][idx],
            "selection_frequency_inner": float(freq[idx]),
            "mean_inner_rank": float(mean_rank[idx]),
        })
    return keep, stability_rows, scores, pvals


def fit_eval(space, ytr, yte, keep, clf, threshold_strategy):
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("clf", clone(clf))])
    Xtr = space["Xtr"][:, keep]
    Xte = space["Xte"][:, keep]
    pipe.fit(Xtr, ytr)
    train_score = predict_scores(pipe, Xtr)
    test_score = predict_scores(pipe, Xte)
    threshold = tune_threshold(ytr, train_score, threshold_strategy)
    return pipe, binary_metrics(yte, test_score, threshold)


def selected_coordinate_rows(space, keep, scores, pvals, ytr, pipe, spec_lookup, base):
    coef = np.ravel(pipe.named_steps["clf"].coef_) if hasattr(pipe.named_steps["clf"], "coef_") else np.full(len(keep), np.nan)
    rows = []
    for pos, idx in enumerate(keep):
        name = space["names"][idx]
        clean = name.replace("::residual_M3", "")
        spec = spec_lookup.get(clean, {})
        vals = space["Xtr"][:, idx]
        mean_hc = float(np.nanmean(vals[ytr == 0])) if np.any(ytr == 0) else np.nan
        mean_mci = float(np.nanmean(vals[ytr == 1])) if np.any(ytr == 1) else np.nan
        rows.append({
            **base,
            "rank": pos + 1,
            "coordinate": name,
            "readable_coordinate": spec.get("readable", name),
            "face": " + ".join(spec.get("face", parse_face_from_coordinate(clean))),
            "f_score": float(scores[idx]) if len(scores) else np.nan,
            "p_value": float(pvals[idx]) if len(pvals) else np.nan,
            "coefficient": float(coef[pos]) if pos < len(coef) and np.isfinite(coef[pos]) else np.nan,
            "abs_coefficient": float(abs(coef[pos])) if pos < len(coef) and np.isfinite(coef[pos]) else np.nan,
            "cohens_d_MCI_minus_HC": cohen_d(vals, ytr),
            "mean_theta_HC": mean_hc,
            "mean_theta_MCI": mean_mci,
            "direction": "higher in MCI" if mean_mci > mean_hc else "higher in HC",
        })
    return rows


def random_faces(node_names, counts, rng):
    faces = {}
    for order_name, count in counts.items():
        ell = int(order_name[1])
        all_faces = list(__import__("itertools").combinations(node_names, ell))
        if not all_faces:
            faces[order_name] = []
            continue
        idx = rng.choice(len(all_faces), size=min(count, len(all_faces)), replace=False)
        faces[order_name] = [all_faces[i] for i in idx]
    return faces


def summarise_metrics(df):
    if df.empty:
        return df
    return (
        df.groupby(["representation", "roi_set", "embedding", "order_model", "selection_strategy", "threshold_strategy", "classifier"], dropna=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            sensitivity_MCI_mean=("sensitivity_MCI", "mean"),
            specificity_HC_mean=("specificity_HC", "mean"),
            selected_threshold_mean=("selected_threshold", "mean"),
            selected_threshold_std=("selected_threshold", "std"),
            n_features_mean=("n_features", "mean"),
            n_folds=("auc", "count"),
        )
        .reset_index()
        .sort_values(["balanced_accuracy_mean", "auc_mean"], ascending=False)
    )


def evaluate_one(space, ytr, yte, classifiers, args, base, spec_lookup):
    metric_rows, selected_rows, stability_rows = [], [], []
    keep, stab, scores, pvals = select_features(space, ytr, args, base)
    stability_rows.extend(stab)
    if len(keep) == 0:
        return metric_rows, selected_rows, stability_rows
    for clf_name, clf in classifiers.items():
        pipe, md = fit_eval(space, ytr, yte, keep, clf, args.threshold_strategy)
        row_base = {**base, "classifier": clf_name, "n_features": int(len(keep))}
        metric_rows.append({**row_base, **md})
        selected_rows.extend(selected_coordinate_rows(space, keep, scores, pvals, ytr, pipe, spec_lookup, row_base))
    return metric_rows, selected_rows, stability_rows


def run(args):
    ensure_dir(args.outdir)
    backend = import_official_pymba()
    df, y, feature_cols, removed_cols = load_dataset(args)
    roi_sets = parse_values(args.roi_sets)
    embeddings = parse_values(args.embeddings)
    order_models = parse_values(args.order_model)
    classifiers = filter_classifiers(get_classifiers(args.random_state), ",".join(args.classifiers) if isinstance(args.classifiers, list) else args.classifiers)
    rskf = RepeatedStratifiedKFold(n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.random_state)

    print(f"Using PyMBA from: {backend['module_path']}", flush=True)
    print(f"Subjects: {len(y)} | HC={(y == 0).sum()} | MCI={(y == 1).sum()} | ROI columns={len(feature_cols)}", flush=True)

    metric_rows, selected_rows, stability_rows = [], [], []
    control_rows, embedding_diag_rows, burden_rows = [], [], []
    node_tables, cross_manifests, stage_manifests, residual_rows = [], [], [], []
    covariate_warnings = set()

    for outer_i, (tr_idx, te_idx) in enumerate(rskf.split(np.zeros(len(y)), y), start=1):
        repeat = ((outer_i - 1) // args.n_splits) + 1
        fold = ((outer_i - 1) % args.n_splits) + 1
        print(f"Repeat {repeat}/{args.n_repeats} | Fold {fold}/{args.n_splits}", flush=True)
        args._current_repeat = repeat
        args._current_fold = fold
        ytr, yte = y[tr_idx], y[te_idx]
        rng = np.random.default_rng(args.random_state + 1009 * outer_i)
        fold_df, cov_warn = covariate_correct_df(df, feature_cols, y, tr_idx, te_idx, args)
        covariate_warnings.update(cov_warn)
        diag_seen = set()

        for roi_set in roi_sets:
            X_nodes_all, node_names_all, _node_table_all = build_node_table(fold_df, feature_cols, roi_set)
            if not node_names_all:
                continue
            observed_faces = build_faces(roi_set, node_names_all)
            if outer_i == 1:
                manifest = motif_manifest_for_roi_set(roi_set, observed_faces)
                if roi_set == "disease_stage_cross_system":
                    cross_manifests.append(manifest)
                if roi_set == "disease_stage_compact":
                    stage_manifests.append(manifest)
            random_face_sets = []
            if args.n_matched_random_motif_controls > 0:
                counts = {k: len(v) for k, v in observed_faces.items()}
                random_face_sets = [random_faces(node_names_all, counts, rng) for _ in range(args.n_matched_random_motif_controls)]
            for embedding in embeddings:
                spaces, node_table, meta = build_all_order_spaces(fold_df, feature_cols, y, tr_idx, te_idx, roi_set, embedding, backend, args)
                if spaces is None:
                    continue
                if outer_i == 1 and node_table is not None and not node_table.empty:
                    node_tables.append(node_table)
                diag_key = (repeat, fold, roi_set, embedding)
                if diag_key not in diag_seen:
                    embedding_diag_rows.append(meta["embedding_diagnostic"])
                    b = meta["burden_summary"].copy()
                    b["repeat"] = repeat
                    b["fold"] = fold
                    b["roi_set"] = roi_set
                    b["embedding"] = embedding
                    burden_rows.extend(b.to_dict("records"))
                    diag_seen.add(diag_key)
                for order_model in order_models:
                    assembled = assemble_order_model(spaces, order_model)
                    base = {
                        "repeat": repeat,
                        "fold": fold,
                        "representation": "observed_disease_motif",
                        "roi_set": roi_set,
                        "embedding": embedding,
                        "order_model": order_model,
                        "selection_strategy": args.selection_strategy,
                        "threshold_strategy": args.threshold_strategy,
                    }
                    rows, sels, stabs = evaluate_one(assembled, ytr, yte, classifiers, args, base, meta["spec_lookup"])
                    metric_rows.extend(rows)
                    selected_rows.extend(sels)
                    stability_rows.extend(stabs)
                    if order_model == "residual_M3":
                        residual_rows.extend(rows)
                    for control_id, faces in enumerate(random_face_sets, start=1):
                        cspaces, _ctable, cmeta = build_all_order_spaces(
                            fold_df, feature_cols, y, tr_idx, te_idx, roi_set, embedding, backend, args, faces_override=faces
                        )
                        if cspaces is None:
                            continue
                        cassembled = assemble_order_model(cspaces, order_model)
                        cbase = {**base, "representation": "matched_random_motif_control", "control_id": control_id}
                        crows, _csels, _cstabs = evaluate_one(cassembled, ytr, yte, classifiers, args, cbase, cmeta["spec_lookup"])
                        control_rows.extend(crows)
                    for control_id in range(1, args.n_label_permutation_controls + 1):
                        perm_ytr = rng.permutation(ytr)
                        pbase = {**base, "representation": "permuted_label_control", "control_id": control_id}
                        keep, _stab, scores, pvals = select_features(assembled, perm_ytr, args, pbase)
                        if len(keep) == 0:
                            continue
                        for clf_name, clf in classifiers.items():
                            pipe, md = fit_eval(assembled, perm_ytr, yte, keep, clf, args.threshold_strategy)
                            control_rows.append({**pbase, "classifier": clf_name, "n_features": int(len(keep)), **md})

    outdir = Path(args.outdir)
    per_fold = pd.DataFrame(metric_rows)
    controls = pd.DataFrame(control_rows)
    per_fold.to_csv(outdir / "per_fold_results.csv", index=False)
    summarise_metrics(per_fold).to_csv(outdir / "repeated_cv_summary.csv", index=False)
    pd.DataFrame(selected_rows).to_csv(outdir / "selected_coordinates_by_fold.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(outdir / "selected_coordinates_by_inner_stability.csv", index=False)
    pd.concat(cross_manifests, ignore_index=True).drop_duplicates().to_csv(outdir / "cross_system_motif_manifest.csv", index=False) if cross_manifests else pd.DataFrame().to_csv(outdir / "cross_system_motif_manifest.csv", index=False)
    pd.concat(stage_manifests, ignore_index=True).drop_duplicates().to_csv(outdir / "disease_stage_manifest.csv", index=False) if stage_manifests else pd.DataFrame().to_csv(outdir / "disease_stage_manifest.csv", index=False)
    summarise_metrics(pd.DataFrame(residual_rows)).to_csv(outdir / "residual_M3_summary.csv", index=False)
    summarise_metrics(controls).to_csv(outdir / "matched_control_summary.csv", index=False)
    per_fold.groupby(["threshold_strategy", "classifier"], dropna=False).agg(
        selected_threshold_mean=("selected_threshold", "mean"),
        selected_threshold_std=("selected_threshold", "std"),
        n_folds=("selected_threshold", "count"),
    ).reset_index().to_csv(outdir / "threshold_tuning_summary.csv", index=False)
    pd.DataFrame(embedding_diag_rows).to_csv(outdir / "embedding_diagnostics.csv", index=False)
    pd.DataFrame(burden_rows).to_csv(outdir / "disease_system_burden_summary.csv", index=False)
    pd.concat(node_tables, ignore_index=True).drop_duplicates().to_csv(outdir / "disease_aware_node_manifest.csv", index=False) if node_tables else pd.DataFrame().to_csv(outdir / "disease_aware_node_manifest.csv", index=False)
    controls.to_csv(outdir / "control_per_fold_results.csv", index=False)
    metadata = {
        "script": Path(__file__).name,
        "coordinate_backend": "official PyMBA unchanged",
        "pymba_module_path": backend["module_path"],
        "pymba_functions": backend["functions"],
        "embedding_descriptions": EMBEDDING_DESCRIPTIONS,
        "removed_feature_columns": removed_cols,
        "covariate_warnings": sorted(covariate_warnings),
        "leakage_checks": [
            "Threshold tuning uses training-fold scores only.",
            "Stability selection uses inner CV inside the outer training fold only.",
            "Residual M3 regression is fit on outer training fold M1+M2 only.",
            "Covariate correction models are fit on outer training fold only.",
            "PyMBA theta/eta coordinate functions are imported and used unchanged.",
        ],
    }
    with open(outdir / "disease_aware_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    summary = summarise_metrics(per_fold)
    print("\n===== Disease-aware PyMBA v2 summary =====", flush=True)
    if not summary.empty:
        print(summary.head(20).to_string(index=False), flush=True)
    print(f"Outputs written to: {args.outdir}", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Disease-aware PyMBA MRI poset v2 experiment.")
    p.add_argument("--csv", required=True)
    p.add_argument("--label_col", default="label_str")
    p.add_argument("--subject_col", default="subject_id")
    p.add_argument("--outdir", default="disease_aware_pymba_v2_results")
    p.add_argument("--roi_sets", nargs="+", default=["disease_stage_cross_system", "disease_stage_compact", "ventricle", "mtl", "cortical", "asymmetry"])
    p.add_argument("--embeddings", nargs="+", default=["directional_softplus_z", "directional_z_positive", "abs_z", "raw_volume"])
    p.add_argument("--order_model", nargs="+", default=["M1", "M1_M2", "M1_M2_M3", "M3_only", "residual_M3"])
    p.add_argument("--selection_strategy", choices=["f_classif", "stability_inner_cv", "none"], default="stability_inner_cv")
    p.add_argument("--inner_splits", type=int, default=5)
    p.add_argument("--stability_threshold", type=float, default=0.6)
    p.add_argument("--inner_topk", nargs="+", default=["5", "10", "20", "30", "50"])
    p.add_argument("--threshold_strategy", choices=["fixed_0.5", "train_balanced_accuracy", "train_youden"], default="train_balanced_accuracy")
    p.add_argument("--classifiers", nargs="+", default=["logreg_l1", "logreg_l2", "linear_svm"])
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_repeats", type=int, default=50)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--epsilon", type=float, default=1e-8)
    p.add_argument("--min_sigma", type=float, default=1e-3)
    p.add_argument("--variance_threshold", type=float, default=1e-12)
    p.add_argument("--unknown_policy", choices=["abs_z", "zero"], default="abs_z")
    p.add_argument("--covariates", nargs="*", default=[])
    p.add_argument("--covariate_correction", choices=["none", "train_linear_residual"], default="none")
    p.add_argument("--covariate_fit_group", choices=["train_all", "train_HC"], default="train_HC")
    p.add_argument("--n_matched_random_motif_controls", type=int, default=0)
    p.add_argument("--n_label_permutation_controls", type=int, default=0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
