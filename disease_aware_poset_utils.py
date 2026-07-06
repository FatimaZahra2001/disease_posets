import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold


EPS = 1e-12


EMBEDDING_DESCRIPTIONS = {
    "raw_volume": "Training-min shifted raw ROI values normalized to a probability vector.",
    "atrophy_energy": "Hard one-sided HC-normative disease-directional z burden: max(0, direction*z).",
    "directional_z_positive": "Hard one-sided HC-normative disease-directional z burden: max(0, direction*z).",
    "abs_z": "Absolute HC-normative abnormality burden: abs(z).",
    "signed_z": "Disease-directional z burden clipped to nonnegative values before probability conversion.",
    "directional_softplus_z": "Smooth disease-directional burden: logaddexp(0, direction*z).",
    "directional_sigmoid_z": "Bounded disease-directional burden: sigmoid(direction*z).",
    "disease_likelihood": "Per-ROI Gaussian log-likelihood ratio, fitted on the training fold, converted with softplus.",
    "rank_normalized_directional": "Disease-directional burden converted to train-fold empirical percentile scale.",
}


PYMBA_CANDIDATE_DIRS = [
    Path(__file__).resolve().parent / "external" / "gkazunii_pymba" / "src",
    Path(__file__).resolve().parent / "external" / "pymba" / "src",
]


def norm(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def contains_any(text, patterns):
    t = norm(text)
    return any(norm(p) in t for p in patterns)


def is_left_feature(col):
    c = str(col).lower()
    return c.startswith("lh_") or "left" in c or "_lh_" in c


def is_right_feature(col):
    c = str(col).lower()
    return c.startswith("rh_") or "right" in c or "_rh_" in c


DECREASE_KEYWORDS = [
    "hippocampus",
    "amygdala",
    "accumbens",
    "entorhinal",
    "parahippocampal",
    "temporalpole",
    "inferiortemporal",
    "middletemporal",
    "superiortemporal",
    "fusiform",
    "frontalpole",
    "caudalanteriorcingulate",
    "isthmuscingulate",
    "caudalmiddlefrontal",
    "precentral",
    "postcentral",
    "precuneus",
    "cuneus",
    "medialorbitofrontal",
    "grayvol",
    "graymatter",
    "thickavg",
    "thickness",
]


INCREASE_KEYWORDS = [
    "lateralventricle",
    "inferiorlateralventricle",
    "inflateralventricle",
    "inf_lat_vent",
    "lat_vent",
    "3rdventricle",
    "thirdventricle",
    "4thventricle",
    "fourthventricle",
    "5thventricle",
    "fifthventricle",
    "csf",
    "ventricle",
]


def disease_direction(name):
    s = norm(name)
    if any(norm(k) in s for k in INCREASE_KEYWORDS):
        return "increase"
    if any(norm(k) in s for k in DECREASE_KEYWORDS):
        return "decrease"
    return "unknown"


FAMILIES = {
    "mtl": {
        "label": "MTL motif",
        "node_keywords": {
            "hippocampus": ["hippocampus"],
            "amygdala": ["amygdala"],
            "entorhinal": ["entorhinal"],
            "parahippocampal": ["parahippocampal"],
            "temporalpole": ["temporalpole"],
            "inferiortemporal": ["inferiortemporal"],
            "middletemporal": ["middletemporal"],
            "fusiform": ["fusiform"],
        },
        "m2": [
            ("hippocampus", "amygdala"),
            ("hippocampus", "entorhinal"),
            ("hippocampus", "parahippocampal"),
            ("entorhinal", "parahippocampal"),
            ("hippocampus", "temporalpole"),
            ("amygdala", "temporalpole"),
        ],
        "m3": [
            ("entorhinal", "hippocampus", "parahippocampal"),
            ("hippocampus", "amygdala", "temporalpole"),
            ("hippocampus", "entorhinal", "fusiform"),
        ],
    },
    "ventricle": {
        "label": "Ventricle motif",
        "node_keywords": {
            "lateral_ventricle": ["lateral_ventricle", "lat_vent"],
            "inferior_lateral_ventricle": ["inferior_lateral_ventricle", "inf_lat_vent", "inflateralventricle"],
            "third_ventricle": ["3rd_ventricle", "third_ventricle"],
            "fourth_ventricle": ["4th_ventricle", "fourth_ventricle"],
            "fifth_ventricle": ["5th_ventricle", "fifth_ventricle"],
            "csf": ["csf"],
        },
        "m2": [
            ("lateral_ventricle", "inferior_lateral_ventricle"),
            ("lateral_ventricle", "csf"),
            ("inferior_lateral_ventricle", "csf"),
            ("third_ventricle", "lateral_ventricle"),
        ],
        "m3": [
            ("lateral_ventricle", "inferior_lateral_ventricle", "csf"),
            ("lateral_ventricle", "third_ventricle", "csf"),
        ],
    },
    "cortical": {
        "label": "Cortical motif",
        "node_keywords": {
            "frontal_pole": ["frontalpole"],
            "caudal_anterior_cingulate": ["caudalanteriorcingulate"],
            "isthmus_cingulate": ["isthmuscingulate"],
            "caudal_middle_frontal": ["caudalmiddlefrontal"],
            "precentral": ["precentral"],
            "postcentral": ["postcentral"],
            "precuneus": ["precuneus"],
            "cuneus": ["cuneus"],
            "medial_orbitofrontal": ["medialorbitofrontal"],
        },
        "m2": [
            ("caudal_anterior_cingulate", "caudal_middle_frontal"),
            ("isthmus_cingulate", "frontal_pole"),
            ("frontal_pole", "precentral"),
            ("caudal_anterior_cingulate", "precuneus"),
            ("frontal_pole", "medial_orbitofrontal"),
            ("precentral", "postcentral"),
        ],
        "m3": [
            ("frontal_pole", "caudal_anterior_cingulate", "precuneus"),
            ("frontal_pole", "precentral", "postcentral"),
            ("caudal_anterior_cingulate", "frontal_pole", "medial_orbitofrontal"),
        ],
    },
}


ASYMMETRY_BASES = {
    "hippocampus": ["hippocampus"],
    "amygdala": ["amygdala"],
    "accumbens": ["accumbens"],
    "ventricle": ["ventricle", "lat_vent", "inf_lat_vent"],
    "temporal": ["entorhinal", "parahippocampal", "temporalpole", "inferiortemporal", "middletemporal", "fusiform"],
    "precentral": ["precentral"],
    "postcentral": ["postcentral"],
    "precuneus": ["precuneus"],
}


def import_official_pymba():
    for pymba_dir in reversed(PYMBA_CANDIDATE_DIRS):
        if pymba_dir.exists() and str(pymba_dir) not in sys.path:
            sys.path.insert(0, str(pymba_dir))
    import transform

    for required in ["theta_from_prob", "eta_from_prob", "prob_from_theta", "decumsum"]:
        if not hasattr(transform, required):
            raise RuntimeError(f"PyMBA transform is missing {required}")
    probe = np.asarray([[0.2, 0.3], [0.1, 0.4]], dtype=float)
    _ = transform.theta_from_prob(probe, chi="Tsallis", q=1)
    _ = transform.eta_from_prob(probe, chi="Tsallis", q=1)
    return {
        "transform": transform,
        "module_path": str(Path(transform.__file__).resolve().parent),
        "functions": {
            "theta": "transform.theta_from_prob",
            "eta": "transform.eta_from_prob",
            "zeta": "transform.prob_from_theta",
            "mobius": "transform.decumsum",
        },
    }


def parse_values(value, cast=str):
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace(",", " ").split()
    return [cast(x) for x in raw if str(x).strip()]


def match_columns(feature_cols, keywords):
    return [c for c in feature_cols if contains_any(c, keywords)]


def aggregate_regular_family(df, feature_cols, roi_set):
    spec = FAMILIES[roi_set]
    data = {}
    rows = []
    for node, keywords in spec["node_keywords"].items():
        cols = match_columns(feature_cols, keywords)
        if roi_set == "ventricle" and node == "lateral_ventricle":
            cols = [
                c for c in cols
                if not contains_any(c, ["inferior_lateral_ventricle", "inf_lat_vent", "inflateralventricle"])
            ]
        if not cols:
            continue
        data[node] = df[cols].astype(float).mean(axis=1)
        dirs = [disease_direction(c) for c in cols]
        direction = pd.Series(dirs).mode().iloc[0] if dirs else disease_direction(node)
        rows.append({
            "roi_set": roi_set,
            "node": node,
            "direction": direction,
            "n_source_features": len(cols),
            "source_features": ";".join(cols),
        })
    return data, rows


def aggregate_asymmetry_family(df, feature_cols):
    data = {}
    rows = []
    for base, keywords in ASYMMETRY_BASES.items():
        cols = match_columns(feature_cols, keywords)
        left_cols = [c for c in cols if is_left_feature(c)]
        right_cols = [c for c in cols if is_right_feature(c)]
        if not left_cols or not right_cols:
            continue
        left = df[sorted(left_cols)].astype(float).mean(axis=1)
        right = df[sorted(right_cols)].astype(float).mean(axis=1)
        data[f"left_{base}_burden"] = left
        data[f"right_{base}_burden"] = right
        data[f"bilateral_{base}_burden"] = (left + right) / 2.0
        data[f"{base}_asymmetry"] = (left - right).abs()
        for node in [f"left_{base}_burden", f"right_{base}_burden", f"bilateral_{base}_burden", f"{base}_asymmetry"]:
            rows.append({
                "roi_set": "asymmetry",
                "node": node,
                "direction": "unknown" if node.endswith("_asymmetry") else disease_direction(base),
                "n_source_features": len(left_cols) + len(right_cols),
                "source_features": "LEFT:" + ";".join(sorted(left_cols)) + " || RIGHT:" + ";".join(sorted(right_cols)),
            })
    return data, rows


def aggregate_disease_stage_compact_family(df, feature_cols):
    data = {}
    rows = []
    systems = [
        ("MTL_burden", FAMILIES["mtl"]["node_keywords"], "decrease", "Stage 1 medial temporal atrophy burden"),
        ("ventricle_CSF_burden", FAMILIES["ventricle"]["node_keywords"], "increase", "Stage 2 ventricular and CSF expansion burden"),
        ("cortical_frontocingulate_burden", FAMILIES["cortical"]["node_keywords"], "decrease", "Stage 3 cortical/fronto-cingulate decline burden"),
    ]
    for node, node_keywords, direction, interpretation in systems:
        cols = sorted(set(itertools.chain.from_iterable(match_columns(feature_cols, kws) for kws in node_keywords.values())))
        if not cols:
            continue
        data[node] = df[cols].astype(float).mean(axis=1)
        rows.append({
            "roi_set": "disease_stage_compact",
            "node": node,
            "direction": direction,
            "n_source_features": len(cols),
            "source_features": ";".join(cols),
            "biological_interpretation": interpretation,
        })
    asym_data, asym_rows = aggregate_asymmetry_family(df, feature_cols)
    asym_nodes = [n for n in asym_data if n.endswith("_asymmetry")]
    if asym_nodes:
        data["asymmetry_burden"] = pd.DataFrame({n: asym_data[n] for n in asym_nodes}).mean(axis=1)
        rows.append({
            "roi_set": "disease_stage_compact",
            "node": "asymmetry_burden",
            "direction": "unknown",
            "n_source_features": int(sum(r["n_source_features"] for r in asym_rows if str(r["node"]).endswith("_asymmetry"))),
            "source_features": "derived from paired left/right disease burden nodes",
            "biological_interpretation": "Stage 4 hemispheric/system disruption burden",
        })
    return data, rows


def aggregate_disease_stage_cross_system_family(df, feature_cols):
    data = {}
    rows = []
    for family in ["mtl", "ventricle", "cortical"]:
        f_data, f_rows = aggregate_regular_family(df, feature_cols, family)
        data.update(f_data)
        for row in f_rows:
            row = dict(row)
            row["roi_set"] = "disease_stage_cross_system"
            row["biological_interpretation"] = f"{family} regional disease burden"
            rows.append(row)
    stage_data, stage_rows = aggregate_disease_stage_compact_family(df, feature_cols)
    data.update(stage_data)
    for row in stage_rows:
        row = dict(row)
        row["roi_set"] = "disease_stage_cross_system"
        rows.append(row)
    return data, rows


def build_node_table(df, feature_cols, roi_set):
    if roi_set in FAMILIES:
        data, rows = aggregate_regular_family(df, feature_cols, roi_set)
    elif roi_set == "asymmetry":
        data, rows = aggregate_asymmetry_family(df, feature_cols)
    elif roi_set == "combined":
        data, rows = aggregate_combined_family(df, feature_cols)
    elif roi_set == "disease_stage_compact":
        data, rows = aggregate_disease_stage_compact_family(df, feature_cols)
    elif roi_set == "disease_stage_cross_system":
        data, rows = aggregate_disease_stage_cross_system_family(df, feature_cols)
    else:
        raise ValueError(f"Unknown roi_set: {roi_set}")
    if not rows:
        return np.zeros((len(df), 0), dtype=float), [], pd.DataFrame()
    node_table = pd.DataFrame(rows)
    names = node_table["node"].tolist()
    X = pd.DataFrame(data, index=df.index)[names].to_numpy(dtype=float)
    return X, names, node_table


def aggregate_combined_family(df, feature_cols):
    data = {}
    rows = []
    systems = {
        "mtl_atrophy_burden": FAMILIES["mtl"]["node_keywords"],
        "ventricular_csf_expansion_burden": FAMILIES["ventricle"]["node_keywords"],
        "cortical_gray_matter_decline_burden": FAMILIES["cortical"]["node_keywords"],
    }
    for system, node_keywords in systems.items():
        cols = sorted(set(itertools.chain.from_iterable(match_columns(feature_cols, kws) for kws in node_keywords.values())))
        if not cols:
            continue
        data[system] = df[cols].astype(float).mean(axis=1)
        rows.append({
            "roi_set": "combined",
            "node": system,
            "direction": "increase" if "ventricular" in system else "decrease",
            "n_source_features": len(cols),
            "source_features": ";".join(cols),
        })
    asym_data, asym_rows = aggregate_asymmetry_family(df, feature_cols)
    asym_nodes = [n for n in asym_data if n.endswith("_asymmetry")]
    if asym_nodes:
        data["hemisphere_asymmetry_burden"] = pd.DataFrame({n: asym_data[n] for n in asym_nodes}).mean(axis=1)
        rows.append({
            "roi_set": "combined",
            "node": "hemisphere_asymmetry_burden",
            "direction": "unknown",
            "n_source_features": int(sum(r["n_source_features"] for r in asym_rows if str(r["node"]).endswith("_asymmetry"))),
            "source_features": "derived from paired left/right disease burden nodes",
        })
    return data, rows


def build_faces(roi_set, node_names):
    present = set(node_names)
    m1 = [(n,) for n in node_names]
    if roi_set in FAMILIES:
        m2 = [tuple(face) for face in FAMILIES[roi_set]["m2"] if set(face) <= present]
        m3 = [tuple(face) for face in FAMILIES[roi_set]["m3"] if set(face) <= present]
    elif roi_set == "asymmetry":
        bases = sorted({n.replace("left_", "").replace("right_", "").replace("bilateral_", "").replace("_burden", "").replace("_asymmetry", "") for n in node_names})
        m2 = []
        m3 = []
        for base in bases:
            left, right, bilateral, asym = f"left_{base}_burden", f"right_{base}_burden", f"bilateral_{base}_burden", f"{base}_asymmetry"
            if {left, right} <= present:
                m2.append((left, right))
            if {bilateral, asym} <= present:
                m2.append((bilateral, asym))
            if {left, right, asym} <= present:
                m3.append((left, right, asym))
    elif roi_set == "combined":
        ordered = [n for n in [
            "mtl_atrophy_burden",
            "ventricular_csf_expansion_burden",
            "cortical_gray_matter_decline_burden",
            "hemisphere_asymmetry_burden",
        ] if n in present]
        m2 = list(itertools.combinations(ordered, 2))
        m3 = list(itertools.combinations(ordered, 3))
    elif roi_set == "disease_stage_compact":
        ordered = [n for n in [
            "MTL_burden",
            "ventricle_CSF_burden",
            "cortical_frontocingulate_burden",
            "asymmetry_burden",
        ] if n in present]
        m2 = list(itertools.combinations(ordered, 2))
        m3 = list(itertools.combinations(ordered, 3))
    elif roi_set == "disease_stage_cross_system":
        m2, m3 = cross_system_structured_faces(present)
    else:
        raise ValueError(f"Unknown roi_set: {roi_set}")
    return {"M1": m1, "M2": m2, "M3": m3}


def cross_system_structured_faces(present):
    pair_candidates = [
        ("hippocampus", "lateral_ventricle"),
        ("hippocampus", "inferior_lateral_ventricle"),
        ("hippocampus", "csf"),
        ("entorhinal", "inferior_lateral_ventricle"),
        ("amygdala", "csf"),
        ("MTL_burden", "ventricle_CSF_burden"),
        ("MTL_burden", "cortical_frontocingulate_burden"),
        ("MTL_burden", "asymmetry_burden"),
        ("ventricle_CSF_burden", "cortical_frontocingulate_burden"),
        ("ventricle_CSF_burden", "asymmetry_burden"),
        ("cortical_frontocingulate_burden", "asymmetry_burden"),
        ("hippocampus", "caudal_anterior_cingulate"),
        ("hippocampus", "isthmus_cingulate"),
        ("entorhinal", "caudal_anterior_cingulate"),
        ("inferiortemporal", "lateral_ventricle"),
        ("middletemporal", "lateral_ventricle"),
        ("fusiform", "csf"),
        ("temporalpole", "asymmetry_burden"),
        ("precuneus", "ventricle_CSF_burden"),
        ("caudal_anterior_cingulate", "ventricle_CSF_burden"),
    ]
    triple_candidates = [
        ("MTL_burden", "ventricle_CSF_burden", "csf"),
        ("MTL_burden", "cortical_frontocingulate_burden", "ventricle_CSF_burden"),
        ("MTL_burden", "asymmetry_burden", "ventricle_CSF_burden"),
        ("hippocampus", "entorhinal", "inferior_lateral_ventricle"),
        ("hippocampus", "amygdala", "csf"),
        ("inferiortemporal", "caudal_anterior_cingulate", "lateral_ventricle"),
        ("middletemporal", "isthmus_cingulate", "lateral_ventricle"),
        ("hippocampus", "cortical_frontocingulate_burden", "csf"),
        ("hippocampus", "caudal_anterior_cingulate", "inferior_lateral_ventricle"),
        ("entorhinal", "cortical_frontocingulate_burden", "ventricle_CSF_burden"),
        ("amygdala", "asymmetry_burden", "csf"),
        ("temporalpole", "asymmetry_burden", "lateral_ventricle"),
        ("fusiform", "caudal_anterior_cingulate", "csf"),
        ("precuneus", "MTL_burden", "ventricle_CSF_burden"),
        ("MTL_burden", "cortical_frontocingulate_burden", "asymmetry_burden"),
    ]
    m2 = [face for face in pair_candidates if set(face) <= present]
    m3 = [face for face in triple_candidates if set(face) <= present]
    return m2, m3


def motif_manifest_for_roi_set(roi_set, faces_by_order):
    rows = []
    for order_name, faces in faces_by_order.items():
        for face in faces:
            systems = []
            for node in face:
                n = str(node).lower()
                if any(k in n for k in ["hippocampus", "entorhinal", "amygdala", "temporal", "mtl"]):
                    systems.append("MTL/temporal")
                if any(k in n for k in ["ventricle", "csf"]):
                    systems.append("ventricular/CSF")
                if any(k in n for k in ["cingulate", "frontal", "cortical", "precuneus", "precentral", "postcentral"]):
                    systems.append("cortical/fronto-cingulate")
                if "asym" in n:
                    systems.append("asymmetry/system disruption")
            systems = sorted(set(systems)) or ["regional disease burden"]
            rows.append({
                "roi_set": roi_set,
                "motif_name": readable_coordinate(roi_set, face),
                "motif_order": order_name,
                "nodes": " + ".join(face),
                "biological_interpretation": "Cross-system disease convergence motif" if len(systems) > 1 else systems[0],
                "disease_systems_involved": ";".join(systems),
            })
    return pd.DataFrame(rows)


def fit_preprocess_train_test(X_train_raw, X_test_raw, names, variance_threshold=1e-12):
    imputer = SimpleImputer(strategy="median")
    Xtr = imputer.fit_transform(X_train_raw)
    Xte = imputer.transform(X_test_raw)
    if Xtr.shape[1] == 0:
        return Xtr, Xte, [], imputer, None
    vt = VarianceThreshold(variance_threshold)
    try:
        Xtr = vt.fit_transform(Xtr)
        Xte = vt.transform(Xte)
    except ValueError:
        return np.zeros((Xtr.shape[0], 0)), np.zeros((Xte.shape[0], 0)), [], imputer, None
    kept = [n for n, keep in zip(names, vt.get_support()) if keep]
    return Xtr, Xte, kept, imputer, vt


def fit_hc_normative_stats(X_train, y_train, epsilon):
    hc = X_train[np.asarray(y_train) == 0]
    if hc.shape[0] == 0:
        raise RuntimeError("Cannot fit HC normative statistics: this training fold has no HC subjects.")
    mean = np.nanmean(hc, axis=0)
    std = np.nanstd(hc, axis=0, ddof=1 if hc.shape[0] > 1 else 0)
    std = np.where(np.isfinite(std) & (std > epsilon), std, 1.0)
    return mean, std


def direction_sign(direction):
    if direction == "increase":
        return 1.0
    if direction == "decrease":
        return -1.0
    return 0.0


def stable_sigmoid(x):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def burden_from_z(z, directions, embedding, epsilon, unknown_policy="abs_z"):
    directions = list(directions)
    out = np.zeros_like(z, dtype=float)
    for j, direction in enumerate(directions):
        if embedding == "abs_z":
            out[:, j] = np.abs(z[:, j])
        elif embedding == "signed_z":
            if direction == "decrease":
                out[:, j] = -z[:, j]
            elif direction == "increase":
                out[:, j] = z[:, j]
            else:
                out[:, j] = np.abs(z[:, j]) if unknown_policy == "abs_z" else 0.0
            out[:, j] = np.maximum(out[:, j], 0.0)
        elif embedding in {"atrophy_energy", "directional_z_positive", "rank_normalized_directional"}:
            if direction == "decrease":
                out[:, j] = np.maximum(0.0, -z[:, j])
            elif direction == "increase":
                out[:, j] = np.maximum(0.0, z[:, j])
            else:
                out[:, j] = np.abs(z[:, j]) if unknown_policy == "abs_z" else 0.0
        elif embedding == "directional_softplus_z":
            sign = direction_sign(direction)
            if sign:
                out[:, j] = np.logaddexp(0.0, sign * z[:, j])
            else:
                out[:, j] = np.logaddexp(0.0, np.abs(z[:, j])) if unknown_policy == "abs_z" else 0.0
        elif embedding == "directional_sigmoid_z":
            sign = direction_sign(direction)
            if sign:
                out[:, j] = stable_sigmoid(sign * z[:, j])
            else:
                out[:, j] = stable_sigmoid(np.abs(z[:, j])) if unknown_policy == "abs_z" else 0.0
        else:
            raise ValueError(f"Unsupported z-score embedding: {embedding}")
    return out + epsilon


def gaussian_logpdf(x, mu, sigma):
    return -0.5 * np.log(2.0 * np.pi * sigma * sigma) - 0.5 * ((x - mu) / sigma) ** 2


def disease_likelihood_burden(z_train, z_test, y_train, directions, epsilon, unknown_policy="abs_z", min_sigma=1e-3, min_class_samples=3):
    directions = list(directions)
    y_train = np.asarray(y_train)
    A_tr = np.zeros_like(z_train, dtype=float)
    A_te = np.zeros_like(z_test, dtype=float)
    rows = []
    fallback_count = 0
    for j, direction in enumerate(directions):
        sign = direction_sign(direction)
        if sign:
            s_tr = sign * z_train[:, j]
            s_te = sign * z_test[:, j]
        elif unknown_policy == "abs_z":
            s_tr = np.abs(z_train[:, j])
            s_te = np.abs(z_test[:, j])
        else:
            A_tr[:, j] = 0.0
            A_te[:, j] = 0.0
            rows.append({"feature_index": j, "fallback": "unknown_zero"})
            continue

        hc = s_tr[(y_train == 0) & np.isfinite(s_tr)]
        mci = s_tr[(y_train == 1) & np.isfinite(s_tr)]
        if len(hc) < min_class_samples or len(mci) < min_class_samples:
            A_tr[:, j] = np.logaddexp(0.0, s_tr)
            A_te[:, j] = np.logaddexp(0.0, s_te)
            fallback_count += 1
            rows.append({"feature_index": j, "fallback": "directional_softplus_z", "n_hc": len(hc), "n_mci": len(mci)})
            continue

        mu_hc = float(np.mean(hc))
        mu_mci = float(np.mean(mci))
        sig_hc = float(max(np.std(hc, ddof=1), min_sigma))
        sig_mci = float(max(np.std(mci, ddof=1), min_sigma))
        llr_tr = gaussian_logpdf(s_tr, mu_mci, sig_mci) - gaussian_logpdf(s_tr, mu_hc, sig_hc)
        llr_te = gaussian_logpdf(s_te, mu_mci, sig_mci) - gaussian_logpdf(s_te, mu_hc, sig_hc)
        A_tr[:, j] = np.logaddexp(0.0, llr_tr)
        A_te[:, j] = np.logaddexp(0.0, llr_te)
        rows.append({
            "feature_index": j,
            "fallback": "",
            "n_hc": len(hc),
            "n_mci": len(mci),
            "mu_hc": mu_hc,
            "mu_mci": mu_mci,
            "sigma_hc": sig_hc,
            "sigma_mci": sig_mci,
        })
    return A_tr + epsilon, A_te + epsilon, {"disease_likelihood_params": rows, "disease_likelihood_fallback_count": fallback_count}


def normalize_burdens_to_probability(A, epsilon):
    A = np.asarray(A, dtype=float)
    A = np.where(np.isfinite(A) & (A > 0.0), A, 0.0)
    row_sums = A.sum(axis=1, keepdims=True)
    bad = (~np.isfinite(row_sums[:, 0])) | (row_sums[:, 0] <= epsilon)
    P = np.zeros_like(A, dtype=float)
    good = ~bad
    if np.any(good):
        P[good] = A[good] / np.maximum(row_sums[good], epsilon)
    if np.any(bad) and A.shape[1] > 0:
        P[bad] = 1.0 / A.shape[1]
    return P, int(np.sum(bad))


def empirical_percentile_transform(A_train, A_test):
    A_train = np.asarray(A_train, dtype=float)
    A_test = np.asarray(A_test, dtype=float)
    Ptr = np.zeros_like(A_train, dtype=float)
    Pte = np.zeros_like(A_test, dtype=float)
    for j in range(A_train.shape[1]):
        ref = np.sort(A_train[np.isfinite(A_train[:, j]), j])
        if len(ref) == 0:
            continue
        Ptr[:, j] = np.searchsorted(ref, A_train[:, j], side="right") / len(ref)
        Pte[:, j] = np.searchsorted(ref, A_test[:, j], side="right") / len(ref)
    return Ptr, Pte


def probability_embedding(X_train, X_test, y_train, directions, embedding, epsilon, unknown_policy="abs_z", min_sigma=1e-3):
    if embedding == "raw_volume":
        mins = np.nanmin(X_train, axis=0, keepdims=True)
        shift = np.minimum(mins, 0.0)
        A_tr = X_train - shift + epsilon
        A_te = X_test - shift + epsilon
        stats = {"hc_mean": None, "hc_std": None, "train_min_shift": shift.ravel().tolist()}
    else:
        mean, std = fit_hc_normative_stats(X_train, y_train, epsilon)
        ztr = (X_train - mean) / std
        zte = (X_test - mean) / std
        if embedding == "disease_likelihood":
            A_tr, A_te, likelihood_stats = disease_likelihood_burden(
                ztr,
                zte,
                y_train,
                directions,
                epsilon,
                unknown_policy=unknown_policy,
                min_sigma=min_sigma,
            )
        elif embedding == "rank_normalized_directional":
            raw_tr = burden_from_z(ztr, directions, embedding, 0.0, unknown_policy=unknown_policy)
            raw_te = burden_from_z(zte, directions, embedding, 0.0, unknown_policy=unknown_policy)
            A_tr, A_te = empirical_percentile_transform(raw_tr, raw_te)
            A_tr = A_tr + epsilon
            A_te = A_te + epsilon
            likelihood_stats = {}
        else:
            A_tr = burden_from_z(ztr, directions, embedding, epsilon, unknown_policy=unknown_policy)
            A_te = burden_from_z(zte, directions, embedding, epsilon, unknown_policy=unknown_policy)
            likelihood_stats = {}
        stats = {"hc_mean": mean.tolist(), "hc_std": std.tolist(), "train_min_shift": None}
        stats.update(likelihood_stats)
    P_tr, train_uniform_count = normalize_burdens_to_probability(A_tr, epsilon)
    P_te, test_uniform_count = normalize_burdens_to_probability(A_te, epsilon)
    stats["train_uniform_probability_count"] = train_uniform_count
    stats["test_uniform_probability_count"] = test_uniform_count
    return P_tr, P_te, A_tr, A_te, stats


def embedding_diagnostic_row(repeat, fold, roi_set, embedding, node_names, A_train, P_train, y_train, stats):
    y_train = np.asarray(y_train)
    A_train = np.asarray(A_train, dtype=float)
    P_train = np.asarray(P_train, dtype=float)
    hc = y_train == 0
    mci = y_train == 1

    def group_mean(values, mask):
        if not np.any(mask):
            return np.nan
        return float(np.nanmean(values[mask]))

    total = np.nansum(A_train, axis=1)
    max_burden = np.nanmax(A_train, axis=1) if A_train.shape[1] else np.full(A_train.shape[0], np.nan)
    entropy = -np.nansum(np.where(P_train > 0, P_train * np.log(np.maximum(P_train, EPS)), 0.0), axis=1)
    mean_hc = np.nanmean(A_train[hc], axis=0) if np.any(hc) else np.full(A_train.shape[1], np.nan)
    mean_mci = np.nanmean(A_train[mci], axis=0) if np.any(mci) else np.full(A_train.shape[1], np.nan)
    diff = mean_mci - mean_hc
    order = np.argsort(np.nan_to_num(np.abs(diff), nan=-np.inf))[::-1][:10]
    top = [
        f"{node_names[i]}:{diff[i]:.6g}"
        for i in order
        if i < len(node_names) and np.isfinite(diff[i])
    ]
    return {
        "repeat": repeat,
        "fold": fold,
        "roi_set": roi_set,
        "embedding": embedding,
        "mean_total_burden_HC": group_mean(total, hc),
        "mean_total_burden_MCI": group_mean(total, mci),
        "mean_max_roi_burden_HC": group_mean(max_burden, hc),
        "mean_max_roi_burden_MCI": group_mean(max_burden, mci),
        "entropy_probability_HC_mean": group_mean(entropy, hc),
        "entropy_probability_MCI_mean": group_mean(entropy, mci),
        "top10_nodes_by_mean_burden_difference_MCI_minus_HC": ";".join(top),
        "train_uniform_probability_count": int(stats.get("train_uniform_probability_count", 0)),
        "test_uniform_probability_count": int(stats.get("test_uniform_probability_count", 0)),
        "disease_likelihood_fallback_count": int(stats.get("disease_likelihood_fallback_count", 0)),
    }


def face_to_index(face, node_to_idx):
    idx = [0, 0, 0]
    for axis, node in enumerate(face):
        idx[axis] = node_to_idx[node]
    return tuple(idx)


def readable_node(node):
    return str(node).replace("_", " ")


def readable_coordinate(roi_set, face):
    if roi_set == "asymmetry" and len(face) >= 2:
        clean = [readable_node(x) for x in face]
        return "Asymmetry motif: " + " + ".join(clean)
    label = FAMILIES.get(roi_set, {"label": "Disease-system motif"})["label"]
    return f"{label}: " + " + ".join(readable_node(x) for x in face)


def coordinate_name(roi_set, order_name, face):
    return "theta(" + roi_set + ":" + order_name + ":" + " x ".join(face) + ")"


def build_face_tensor(P_nodes, node_names, faces_by_order, roi_set, null_permutation=None):
    node_to_idx = {node: i + 1 for i, node in enumerate(node_names)}
    source_idx = {node: i for i, node in enumerate(node_names)}
    shape = (len(node_names) + 1, len(node_names) + 1, len(node_names) + 1)
    tensors = np.full((P_nodes.shape[0], *shape), EPS, dtype=float)
    specs = {}
    for order_name, faces in faces_by_order.items():
        specs[order_name] = []
        for face in faces:
            mapped = tuple(null_permutation.get(n, n) for n in face) if null_permutation else tuple(face)
            if any(n not in node_to_idx or n not in source_idx for n in mapped):
                continue
            idx = face_to_index(mapped, node_to_idx)
            cols = [source_idx[n] for n in mapped]
            tensors[(slice(None), *idx)] = P_nodes[:, cols].mean(axis=1) + EPS
            specs[order_name].append({
                "index": idx,
                "name": coordinate_name(roi_set, order_name, tuple(face)),
                "readable": readable_coordinate(roi_set, tuple(face)),
                "face": tuple(face),
                "mapped_face": mapped,
                "ell": int(order_name[1]),
            })
    totals = tensors.reshape(tensors.shape[0], -1).sum(axis=1)
    tensors = tensors / np.maximum(totals.reshape((tensors.shape[0], 1, 1, 1)), EPS)
    return tensors, specs, {"labels": node_names, "shape": shape}


def official_coordinate_features(tensors, specs, backend, coordinate="theta"):
    if not specs:
        return np.zeros((tensors.shape[0], 0), dtype=float), []
    transform = backend["transform"]
    X = np.zeros((tensors.shape[0], len(specs)), dtype=float)
    for i, P in enumerate(tensors):
        arr = transform.theta_from_prob(P, chi="Tsallis", q=1) if coordinate == "theta" else transform.eta_from_prob(P, chi="Tsallis", q=1)
        for j, spec in enumerate(specs):
            X[i, j] = arr[spec["index"]]
    return X, [spec["name"] for spec in specs]


def parse_face_from_coordinate(name):
    text = str(name)
    if ":" not in text or ")" not in text:
        return []
    inner = text.rsplit(":", 1)[-1].split(")", 1)[0]
    return [x.strip() for x in inner.split(" x ") if x.strip()]
