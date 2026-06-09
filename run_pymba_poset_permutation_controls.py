from pathlib import Path
import sys
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score

# =========================================================
# PATHS
# =========================================================
PYMBA_DIR = Path("/home/hinata/Documents/phd/tokyo internship/NII/code/pymba")
sys.path.append(str(PYMBA_DIR / "src"))

import transform
from mproject import MBA

CSV_PATH = Path(
    "/home/hinata/Documents/phd/tokyo internship/NII/code/analysis/"
    "freesurfer_all_roi_outputs/all_freesurfer_roi_features.csv"
)

OUT_DIR = Path(
    "/home/hinata/Documents/phd/tokyo internship/NII/code/analysis/"
    "pymba_poset_permutation_controls"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8
RANDOM_STATE = 42

HEMIS = ["L", "R"]
FEATURE_TYPES = ["volume_like", "thickness_like"]

# Number of random generic shuffled posets
N_PERMUTATIONS = 25

# For quick testing, set e.g. MAX_ROIS = 30
MAX_ROIS = None

MODELS_TO_RUN = ["logreg", "svm_rbf", "random_forest"]


# =========================================================
# ROI DISCOVERY
# =========================================================
def clean_roi_name(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def discover_cortical_rois(df):
    rois = []

    for col in df.columns:
        if col.startswith("lh_") and col.endswith("_grayvol_mm3"):
            roi = col[len("lh_"):-len("_grayvol_mm3")]
            rh_col = f"rh_{roi}_grayvol_mm3"
            lh_thick = f"lh_{roi}_thickavg_mm"
            rh_thick = f"rh_{roi}_thickavg_mm"

            if rh_col in df.columns and lh_thick in df.columns and rh_thick in df.columns:
                rois.append(roi)

    return sorted(set(rois))


def discover_subcortical_rois(df):
    rois = []

    for col in df.columns:
        if col.startswith("aseg_Left_") and col.endswith("_volume_mm3"):
            roi = col[len("aseg_Left_"):-len("_volume_mm3")]
            right_col = f"aseg_Right_{roi}_volume_mm3"

            if right_col in df.columns:
                rois.append(f"aseg_{roi}")

    return sorted(set(rois))


def build_cols(df):
    cols = {}
    roi_full_names = {}

    cortical_rois = discover_cortical_rois(df)
    subcortical_rois = discover_subcortical_rois(df)

    all_rois = []

    for roi in cortical_rois:
        roi_id = clean_roi_name(roi)
        all_rois.append(roi_id)
        roi_full_names[roi_id] = roi

        cols[("L", roi_id, "volume_like")] = f"lh_{roi}_grayvol_mm3"
        cols[("R", roi_id, "volume_like")] = f"rh_{roi}_grayvol_mm3"
        cols[("L", roi_id, "thickness_like")] = f"lh_{roi}_thickavg_mm"
        cols[("R", roi_id, "thickness_like")] = f"rh_{roi}_thickavg_mm"

    for roi in subcortical_rois:
        raw = roi.replace("aseg_", "")
        roi_id = clean_roi_name(roi)
        all_rois.append(roi_id)
        roi_full_names[roi_id] = raw

        left_col = f"aseg_Left_{raw}_volume_mm3"
        right_col = f"aseg_Right_{raw}_volume_mm3"

        cols[("L", roi_id, "volume_like")] = left_col
        cols[("R", roi_id, "volume_like")] = right_col

        # Subcortical structures do not have cortical thickness;
        # duplicate volume into second channel.
        cols[("L", roi_id, "thickness_like")] = left_col
        cols[("R", roi_id, "thickness_like")] = right_col

    all_rois = sorted(set(all_rois))

    if MAX_ROIS is not None:
        all_rois = all_rois[:MAX_ROIS]

    return all_rois, roi_full_names, cols


# =========================================================
# DATA
# =========================================================
def get_label(df):
    if "label_str" in df.columns:
        return df["label_str"].astype(str)

    if "label" in df.columns:
        if pd.api.types.is_numeric_dtype(df["label"]):
            return df["label"].map({0: "HC", 1: "MCI"})
        return df["label"].astype(str)

    raise ValueError("No label_str or label column found.")


def get_ordered_values(row, regions, cols):
    values = []

    for hemi in HEMIS:
        for region in regions:
            for ftype in FEATURE_TYPES:
                col = cols[(hemi, region, ftype)]
                val = row[col] if col in row.index else np.nan

                if pd.isna(val):
                    val = 0.0

                values.append(max(float(val), 0.0))

    return np.asarray(values, dtype=float)


def normalise_tensor(P):
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 0.0)
    P = P + EPS
    P = P / P.sum()
    return P


def build_tensor_from_values(values, tensor_shape, permutation=None):
    values = np.asarray(values, dtype=float)

    if permutation is not None:
        values = values[permutation]

    P = values.reshape(tensor_shape)
    return normalise_tensor(P)


# =========================================================
# OFFICIAL PYMBA FEATURES
# =========================================================
def l2_error(P, Q):
    return float(np.linalg.norm((P - Q).ravel()))


def kl_div(P, Q):
    P = np.asarray(P, dtype=float) + EPS
    Q = np.asarray(Q, dtype=float) + EPS
    P = P / P.sum()
    Q = Q / Q.sum()
    return float(np.sum(P * np.log(P / Q)))


def run_mba(P, body):
    Q, theta_l, eta_l, hist = MBA(
        P,
        body,
        lr_search=True,
        Newton=True,
        max_iter=100,
        epsilon_auto=True,
        chi="Tsallis",
        q=1,
        rel_epsilon=1e-6,
        rcond=1e-8,
        maxls=20,
        verbose=False,
    )

    Q = normalise_tensor(Q)

    return Q, theta_l, eta_l, hist


def extract_theta_features_for_tensor(P, prefix):
    feats = {}

    theta_full = transform.theta_from_prob(P, chi="Tsallis", q=1)
    eta_full = transform.eta_from_prob(P, chi="Tsallis", q=1)

    flat = np.asarray(theta_full).ravel()
    for i, val in enumerate(flat):
        feats[f"{prefix}_full_theta_{i}"] = float(val)

    feats[f"{prefix}_full_theta_l2"] = float(np.linalg.norm(flat))
    feats[f"{prefix}_full_eta_l2"] = float(np.linalg.norm(np.asarray(eta_full).ravel()))

    for body in [1, 2, 3]:
        Q, theta_l, eta_l, hist = run_mba(P, body)

        theta_flat = np.asarray(theta_l).ravel()

        for i, val in enumerate(theta_flat):
            feats[f"{prefix}_M{body}_theta_{i}"] = float(val)

        feats[f"{prefix}_M{body}_l2_error"] = l2_error(P, Q)
        feats[f"{prefix}_M{body}_kl"] = kl_div(P, Q)
        feats[f"{prefix}_M{body}_theta_l2"] = float(np.linalg.norm(theta_flat))
        feats[f"{prefix}_M{body}_eta_l2"] = float(np.linalg.norm(np.asarray(eta_l).ravel()))

    return feats


def build_feature_dataframe(df, regions, cols, tensor_shape, representation_name, permutation=None):
    rows = []

    for idx, row in df.iterrows():
        sid = row.get("subject_id", row.get("PTID", f"row_{idx}"))

        values = get_ordered_values(row, regions, cols)
        P = build_tensor_from_values(values, tensor_shape, permutation=permutation)

        feats = extract_theta_features_for_tensor(P, representation_name)
        feats["subject_id"] = sid
        feats["label_str"] = row["label_str"]

        rows.append(feats)

    return pd.DataFrame(rows)


# =========================================================
# MODELS
# =========================================================
def evaluate_feature_set(feature_df, feature_cols, y, rep_name):
    X = feature_df[feature_cols].copy()

    min_class = pd.Series(y).value_counts().min()
    n_splits = min(5, min_class)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "roc_auc": "roc_auc",
    }

    all_models = {
        "logreg": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "svm_rbf": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    models = {k: v for k, v in all_models.items() if k in MODELS_TO_RUN}

    rows = []

    for model_name, model in models.items():
        res = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        rows.append({
            "representation": rep_name,
            "model": model_name,
            "n_features": len(feature_cols),
            "cv_accuracy_mean": np.mean(res["test_accuracy"]),
            "cv_accuracy_std": np.std(res["test_accuracy"]),
            "cv_balanced_accuracy_mean": np.mean(res["test_balanced_accuracy"]),
            "cv_balanced_accuracy_std": np.std(res["test_balanced_accuracy"]),
            "cv_roc_auc_mean": np.mean(res["test_roc_auc"]),
            "cv_roc_auc_std": np.std(res["test_roc_auc"]),
        })

    return pd.DataFrame(rows)


def get_feature_sets(feature_df, prefix):
    return {
        f"{prefix}_full_theta": [
            c for c in feature_df.columns
            if c.startswith(f"{prefix}_full_theta_")
        ],
        f"{prefix}_M1_theta": [
            c for c in feature_df.columns
            if c.startswith(f"{prefix}_M1_theta_")
        ],
        f"{prefix}_M2_theta": [
            c for c in feature_df.columns
            if c.startswith(f"{prefix}_M2_theta_")
        ],
        f"{prefix}_M3_theta": [
            c for c in feature_df.columns
            if c.startswith(f"{prefix}_M3_theta_")
        ],
        f"{prefix}_errors_only": [
            c for c in feature_df.columns
            if c.startswith(f"{prefix}_M")
            and (c.endswith("_kl") or c.endswith("_l2_error"))
        ],
    }


def evaluate_representation(feature_df, prefix, permutation_id):
    y = (feature_df["label_str"] == "MCI").astype(int).values

    feature_sets = get_feature_sets(feature_df, prefix)

    all_results = []

    for rep_name, cols in feature_sets.items():
        if len(cols) == 0:
            continue

        print(f"Evaluating {rep_name} | permutation={permutation_id} | n_features={len(cols)}")

        res = evaluate_feature_set(
            feature_df,
            cols,
            y,
            rep_name,
        )

        res["permutation_id"] = permutation_id
        res["representation_family"] = "anatomical" if prefix == "anat" else "generic"
        res["interaction_level"] = (
            "full" if "_full_" in rep_name else
            "M1" if "_M1_" in rep_name else
            "M2" if "_M2_" in rep_name else
            "M3" if "_M3_" in rep_name else
            "errors"
        )

        all_results.append(res)

    return pd.concat(all_results, ignore_index=True)


# =========================================================
# SUMMARIES
# =========================================================
def best_by_representation(results_df):
    idx = (
        results_df
        .sort_values(
            ["cv_balanced_accuracy_mean", "cv_roc_auc_mean"],
            ascending=False,
        )
        .groupby(["representation", "permutation_id"])
        .head(1)
        .index
    )

    return results_df.loc[idx].copy()


def summarise_permutation_controls(best_df):
    rows = []

    generic = best_df[best_df["representation_family"] == "generic"].copy()
    anat = best_df[best_df["representation_family"] == "anatomical"].copy()

    for level in ["full", "M1", "M2", "M3", "errors"]:
        anat_level = anat[anat["interaction_level"] == level]

        if len(anat_level) == 0:
            continue

        anat_row = anat_level.iloc[0]

        generic_level = generic[generic["interaction_level"] == level]

        if len(generic_level) == 0:
            continue

        generic_scores = generic_level["cv_balanced_accuracy_mean"].values
        generic_auc = generic_level["cv_roc_auc_mean"].values

        rows.append({
            "interaction_level": level,
            "anatomical_best_balanced_accuracy": anat_row["cv_balanced_accuracy_mean"],
            "anatomical_auc": anat_row["cv_roc_auc_mean"],
            "generic_mean_balanced_accuracy": np.mean(generic_scores),
            "generic_std_balanced_accuracy": np.std(generic_scores),
            "generic_min_balanced_accuracy": np.min(generic_scores),
            "generic_max_balanced_accuracy": np.max(generic_scores),
            "generic_mean_auc": np.mean(generic_auc),
            "generic_std_auc": np.std(generic_auc),
            "generic_max_auc": np.max(generic_auc),
            "anat_minus_generic_mean_balanced_accuracy": (
                anat_row["cv_balanced_accuracy_mean"] - np.mean(generic_scores)
            ),
            "anat_percentile_vs_generic_balanced_accuracy": (
                100.0 * np.mean(generic_scores <= anat_row["cv_balanced_accuracy_mean"])
            ),
        })

    return pd.DataFrame(rows)


# =========================================================
# MAIN
# =========================================================
def main():
    df = pd.read_csv(CSV_PATH)

    df["label_str"] = get_label(df)
    df = df[df["label_str"].isin(["HC", "MCI"])].copy()

    if "subject_id" in df.columns:
        df = df[~df["subject_id"].isin(["fsaverage", "test_subject"])].copy()

    for flag in ["has_aseg_stats", "has_lh_aparc_stats", "has_rh_aparc_stats"]:
        if flag in df.columns:
            df = df[df[flag] == 1].copy()

    regions, roi_full_names, cols = build_cols(df)

    needed = sorted(set(cols.values()))
    missing = [c for c in needed if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns in CSV: {missing[:20]} ... total missing={len(missing)}")

    df = df.dropna(subset=["label_str"]).copy()

    tensor_shape = (len(HEMIS), len(regions), len(FEATURE_TYPES))
    n_values = int(np.prod(tensor_shape))

    print("Loaded:", df.shape)
    print(df["label_str"].value_counts())
    print(f"Using ROIs: {len(regions)}")
    print(f"Tensor shape: {tensor_shape}")
    print(f"Tensor entries: {n_values}")
    print(f"Permutation controls: {N_PERMUTATIONS}")

    # -----------------------------------------------------
    # Anatomical representation once
    # -----------------------------------------------------
    print("\n=== Building anatomical representation ===")
    anat_features = build_feature_dataframe(
        df,
        regions,
        cols,
        tensor_shape,
        representation_name="anat",
        permutation=None,
    )

    anat_features.to_csv(
        OUT_DIR / "features_anatomical.csv",
        index=False,
    )

    all_results = []

    anat_results = evaluate_representation(
        anat_features,
        prefix="anat",
        permutation_id=-1,
    )

    all_results.append(anat_results)

    # -----------------------------------------------------
    # Generic shuffled permutations
    # -----------------------------------------------------
    permutation_rows = []

    for perm_id in range(N_PERMUTATIONS):
        seed = RANDOM_STATE + perm_id
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_values)

        print(f"\n=== Generic permutation {perm_id + 1}/{N_PERMUTATIONS} seed={seed} ===")

        generic_features = build_feature_dataframe(
            df,
            regions,
            cols,
            tensor_shape,
            representation_name="generic",
            permutation=perm,
        )

        generic_features.to_csv(
            OUT_DIR / f"features_generic_perm_{perm_id:03d}.csv",
            index=False,
        )

        generic_results = evaluate_representation(
            generic_features,
            prefix="generic",
            permutation_id=perm_id,
        )

        generic_results["permutation_seed"] = seed

        all_results.append(generic_results)

        permutation_rows.append({
            "permutation_id": perm_id,
            "seed": seed,
            "permutation": " ".join(map(str, perm.tolist())),
        })

    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(
        OUT_DIR / "all_model_results_with_permutation_controls.csv",
        index=False,
    )

    permutation_df = pd.DataFrame(permutation_rows)
    permutation_df.to_csv(
        OUT_DIR / "generic_permutation_indices.csv",
        index=False,
    )

    best_df = best_by_representation(results_df)
    best_df.to_csv(
        OUT_DIR / "best_model_per_representation_permutation.csv",
        index=False,
    )

    summary_df = summarise_permutation_controls(best_df)
    summary_df.to_csv(
        OUT_DIR / "anatomical_vs_generic_permutation_summary.csv",
        index=False,
    )

    with open(OUT_DIR / "README_interpretation.txt", "w") as f:
        f.write("Permutation controls for anatomical vs generic pymba posets\n")
        f.write("=========================================================\n\n")
        f.write(f"Input CSV: {CSV_PATH}\n")
        f.write(f"Subjects used: {len(df)}\n")
        f.write(f"ROIs used: {len(regions)}\n")
        f.write(f"Tensor shape: {tensor_shape}\n")
        f.write(f"Number of generic permutations: {N_PERMUTATIONS}\n\n")

        f.write("Anatomical representation:\n")
        f.write("  P(hemisphere, ROI, feature_type)\n")
        f.write("  This preserves anatomical semantics.\n\n")

        f.write("Generic permutation controls:\n")
        f.write("  The same values are deterministically shuffled before reshaping into the same tensor shape.\n")
        f.write("  This preserves information content but destroys the anatomical poset structure.\n\n")

        f.write("Official pymba methods:\n")
        f.write("  theta_from_prob(P), eta_from_prob(P), and MBA(P, body=1/2/3).\n\n")

        f.write("Interpretation:\n")
        f.write("  If anatomical performance exceeds the permutation distribution,\n")
        f.write("  it supports the claim that anatomical poset design preserves disease-relevant interactions.\n\n")

        f.write("Summary:\n")
        f.write(summary_df.to_string(index=False))

    print("\nSaved outputs to:", OUT_DIR)

    print("\nPermutation summary:")
    print(summary_df.to_string(index=False))

    print("\nBest results:")
    print(best_df.sort_values(["representation_family", "interaction_level", "permutation_id"]).to_string(index=False))


if __name__ == "__main__":
    main()
