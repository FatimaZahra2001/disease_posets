from pathlib import Path
import re
import pandas as pd
import numpy as np


# =========================================================
# LOCAL PATHS
# =========================================================
FS_DIR = Path("/home/hinata/freesurfer_subjects")

CDR_CSV = Path(
    "/home/hinata/Documents/phd/ADNI4_analysis/code/speech/CDR_07Nov2025.csv"
)

OLD_LABEL_CSV = Path(
    "/home/hinata/Documents/phd/tokyo internship/NII/code/analysis/"
    "freesurfer_baseline_outputs/cleaned_freesurfer_features.csv"
)

OUT_DIR = Path(
    "/home/hinata/Documents/phd/tokyo internship/NII/code/analysis/"
    "freesurfer_all_roi_outputs"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "all_freesurfer_roi_features.csv"
LABEL_AUDIT_CSV = OUT_DIR / "label_merge_audit.csv"

EXCLUDE_SUBJECTS = {"fsaverage", "test_subject"}


# =========================================================
# HELPERS
# =========================================================
def clean_name(x):
    x = str(x).strip()
    x = x.replace("-", "_")
    x = x.replace("/", "_")
    x = x.replace(".", "_")
    x = re.sub(r"[^A-Za-z0-9_]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalise_subject_id(x):
    """
    Keeps ADNI PTID-style IDs clean, e.g. 021_S_10161.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    x = x.replace("-", "_")
    x = x.replace(" ", "")
    return x


def read_aseg_stats(path):
    feats = {}

    if not path.exists():
        return feats

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 5:
                continue

            try:
                volume = float(parts[3])
                struct = clean_name(parts[4])
                feats[f"aseg_{struct}_volume_mm3"] = volume

            except Exception:
                continue

    return feats


def read_aparc_stats(path, hemi):
    feats = {}

    if not path.exists():
        return feats

    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.split()

            if len(parts) < 10:
                continue

            try:
                roi = clean_name(parts[0])

                feats[f"{hemi}_{roi}_numvert"] = float(parts[1])
                feats[f"{hemi}_{roi}_surfarea_mm2"] = float(parts[2])
                feats[f"{hemi}_{roi}_grayvol_mm3"] = float(parts[3])
                feats[f"{hemi}_{roi}_thickavg_mm"] = float(parts[4])
                feats[f"{hemi}_{roi}_thickstd_mm"] = float(parts[5])
                feats[f"{hemi}_{roi}_meancurv"] = float(parts[6])
                feats[f"{hemi}_{roi}_gauscurv"] = float(parts[7])
                feats[f"{hemi}_{roi}_foldind"] = float(parts[8])
                feats[f"{hemi}_{roi}_curvind"] = float(parts[9])

            except Exception:
                continue

    return feats


# =========================================================
# LABEL LOADING
# =========================================================
def find_id_col(df):
    candidates = [
        "subject_id",
        "PTID",
        "RID",
        "Subject",
        "subject",
        "participant_id",
        "ID",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    # fallback: any column containing PTID-like strings
    for c in df.columns:
        sample = df[c].dropna().astype(str).head(30)
        if sample.str.contains(r"\d{3}_S_\d+", regex=True).any():
            return c

    return None


def infer_label_from_cdglobal(x):
    """
    Clinical label mapping:
      CDGLOBAL = 0   -> HC
      CDGLOBAL = 0.5 -> MCI

    Other values are left as NaN because this analysis is HC vs MCI.
    """
    if pd.isna(x):
        return np.nan

    try:
        val = float(x)
    except Exception:
        return np.nan

    if val == 0.0:
        return "HC"

    if val == 0.5:
        return "MCI"

    return np.nan


def load_cdr_labels():
    if not CDR_CSV.exists():
        print(f"WARNING: CDR label file missing: {CDR_CSV}")
        return pd.DataFrame(columns=["subject_id", "label_str", "CDGLOBAL", "label_source"])

    cdr = pd.read_csv(CDR_CSV)

    id_col = find_id_col(cdr)

    if id_col is None:
        print("WARNING: Could not find subject ID column in CDR CSV.")
        print("CDR columns:", cdr.columns.tolist())
        return pd.DataFrame(columns=["subject_id", "label_str", "CDGLOBAL", "label_source"])

    cdglobal_col = None
    for c in ["CDGLOBAL", "CDGLOBAL_STR", "cdr_global", "global_cdr"]:
        if c in cdr.columns:
            cdglobal_col = c
            break

    if cdglobal_col is None:
        print("WARNING: Could not find CDGLOBAL column in CDR CSV.")
        print("CDR columns:", cdr.columns.tolist())
        return pd.DataFrame(columns=["subject_id", "label_str", "CDGLOBAL", "label_source"])

    out = cdr[[id_col, cdglobal_col]].copy()
    out = out.rename(columns={id_col: "subject_id", cdglobal_col: "CDGLOBAL"})
    out["subject_id"] = out["subject_id"].apply(normalise_subject_id)
    out["label_str"] = out["CDGLOBAL"].apply(infer_label_from_cdglobal)
    out["label_source"] = "CDR"

    out = out.dropna(subset=["subject_id"]).copy()

    # priority: MCI over HC, because we do not want to overwrite an observed impairment with a normal visit.
    out["label_priority"] = out["label_str"].map({"MCI": 2, "HC": 1}).fillna(0)
    out = (
        out.sort_values(["subject_id", "label_priority"], ascending=[True, False])
        .drop_duplicates("subject_id", keep="first")
        .drop(columns=["label_priority"])
        .reset_index(drop=True)
    )

    return out


def infer_old_label_str(row):
    if "label_str" in row and pd.notna(row["label_str"]):
        lab = str(row["label_str"]).strip()

        if lab in ["HC", "CN", "Control", "control", "0"]:
            return "HC"

        if lab in ["MCI", "EMCI", "LMCI", "0.5", "1"]:
            return "MCI"

        return lab

    if "label" in row and pd.notna(row["label"]):
        try:
            lab = int(float(row["label"]))

            if lab == 0:
                return "HC"

            if lab == 1:
                return "MCI"

        except Exception:
            return str(row["label"])

    return np.nan


def load_old_labels():
    if not OLD_LABEL_CSV.exists():
        print(f"WARNING: old label file missing: {OLD_LABEL_CSV}")
        return pd.DataFrame(columns=["subject_id", "old_label_str", "old_label_source"])

    old = pd.read_csv(OLD_LABEL_CSV)

    id_col = find_id_col(old)

    if id_col is None:
        print("WARNING: Could not find subject ID column in old label CSV.")
        return pd.DataFrame(columns=["subject_id", "old_label_str", "old_label_source"])

    keep_cols = [id_col]

    for c in ["label", "label_str", "diagnosis", "DX"]:
        if c in old.columns and c not in keep_cols:
            keep_cols.append(c)

    out = old[keep_cols].copy()
    out = out.rename(columns={id_col: "subject_id"})
    out["subject_id"] = out["subject_id"].apply(normalise_subject_id)

    out["old_label_str"] = out.apply(infer_old_label_str, axis=1)
    out["old_label_source"] = "old_cleaned_freesurfer_features"

    out = out.dropna(subset=["subject_id"]).copy()
    out = out.drop_duplicates("subject_id", keep="first")

    return out[["subject_id", "old_label_str", "old_label_source"]]


def load_labels():
    cdr = load_cdr_labels()
    old = load_old_labels()

    labels = cdr.merge(old, on="subject_id", how="outer")

    labels["label_str"] = labels["label_str"].combine_first(labels["old_label_str"])
    labels["label_source"] = np.where(
        labels["label_str"].notna() & cdr.set_index("subject_id").reindex(labels["subject_id"])["label_str"].reset_index(drop=True).notna(),
        "CDR",
        labels.get("old_label_source", "unknown"),
    )

    labels = labels[["subject_id", "label_str", "CDGLOBAL", "old_label_str", "label_source"]].copy()

    labels = labels.drop_duplicates("subject_id", keep="first")

    return labels


# =========================================================
# MAIN
# =========================================================
def main():
    label_df = load_labels()

    print("\nLoaded label table:")
    print("Shape:", label_df.shape)

    if len(label_df):
        print("\nLabel counts from merged labels:")
        print(label_df["label_str"].value_counts(dropna=False))

    rows = []

    subject_dirs = sorted([
        p for p in FS_DIR.iterdir()
        if p.is_dir() and p.name not in EXCLUDE_SUBJECTS
    ])

    print(f"\nFound subject folders: {len(subject_dirs)}")

    for subj_dir in subject_dirs:
        sid = subj_dir.name

        stats_dir = subj_dir / "stats"

        aseg_path = stats_dir / "aseg.stats"
        lh_path = stats_dir / "lh.aparc.stats"
        rh_path = stats_dir / "rh.aparc.stats"

        if not stats_dir.exists():
            print(f"Skipping {sid}: no stats directory")
            continue

        feats = {
            "subject_id": sid,
            "has_recon_all_done": int(
                (subj_dir / "scripts" / "recon-all.done").exists()
            ),
            "has_aseg_stats": int(aseg_path.exists()),
            "has_lh_aparc_stats": int(lh_path.exists()),
            "has_rh_aparc_stats": int(rh_path.exists()),
        }

        feats.update(read_aseg_stats(aseg_path))
        feats.update(read_aparc_stats(lh_path, "lh"))
        feats.update(read_aparc_stats(rh_path, "rh"))

        rows.append(feats)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No FreeSurfer features extracted.")

    df["subject_id"] = df["subject_id"].apply(normalise_subject_id)

    if len(label_df) > 0:
        df = df.merge(label_df, on="subject_id", how="left")
    else:
        df["label_str"] = np.nan
        df["CDGLOBAL"] = np.nan
        df["old_label_str"] = np.nan
        df["label_source"] = np.nan

    df["n_extracted_features"] = df.drop(
        columns=[
            c for c in [
                "subject_id",
                "label",
                "label_str",
                "diagnosis",
                "DX",
                "CDGLOBAL",
                "old_label_str",
                "label_source",
            ]
            if c in df.columns
        ],
        errors="ignore",
    ).notna().sum(axis=1)

    df = df.sort_values("subject_id").reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)

    audit_cols = [
        "subject_id",
        "label_str",
        "CDGLOBAL",
        "old_label_str",
        "label_source",
        "has_recon_all_done",
        "has_aseg_stats",
        "has_lh_aparc_stats",
        "has_rh_aparc_stats",
        "n_extracted_features",
    ]

    audit_cols = [c for c in audit_cols if c in df.columns]
    df[audit_cols].to_csv(LABEL_AUDIT_CSV, index=False)

    print("\n====================================")
    print("Saved:", OUT_CSV)
    print("Saved audit:", LABEL_AUDIT_CSV)
    print("Shape:", df.shape)

    if "label_str" in df.columns:
        print("\nLabel counts:")
        print(df["label_str"].value_counts(dropna=False))

    print("\nFeature extraction completeness:")
    for c in [
        "has_recon_all_done",
        "has_aseg_stats",
        "has_lh_aparc_stats",
        "has_rh_aparc_stats",
    ]:
        if c in df.columns:
            print(f"\n{c}")
            print(df[c].value_counts(dropna=False).to_string())

    usable = df.copy()
    usable = usable[usable["label_str"].isin(["HC", "MCI"])]

    for flag in ["has_aseg_stats", "has_lh_aparc_stats", "has_rh_aparc_stats"]:
        if flag in usable.columns:
            usable = usable[usable[flag] == 1]

    print("\nUsable HC/MCI subjects with complete stats:")
    print(len(usable))

    if len(usable):
        print(usable["label_str"].value_counts(dropna=False))

    missing_labels = df[df["label_str"].isna()]["subject_id"].tolist()
    print("\nSubjects still missing labels:")
    print(len(missing_labels))
    print(missing_labels)

    print("\nExample columns:")
    print(df.columns[:60].tolist())


if __name__ == "__main__":
    main()