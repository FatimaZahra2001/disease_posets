"""Fixed 27-node tensor layouts and ordering controls."""

from __future__ import annotations

import numpy as np
import pandas as pd


ORIGINAL_ORDER = [
    "hippocampus", "amygdala", "entorhinal", "parahippocampal",
    "temporalpole", "inferiortemporal", "middletemporal", "fusiform",
    "lateral_ventricle", "inferior_lateral_ventricle", "third_ventricle",
    "fourth_ventricle", "fifth_ventricle", "csf", "frontal_pole",
    "caudal_anterior_cingulate", "isthmus_cingulate",
    "caudal_middle_frontal", "precentral", "postcentral", "precuneus",
    "cuneus", "medial_orbitofrontal", "MTL_burden",
    "ventricle_CSF_burden", "cortical_frontocingulate_burden",
    "asymmetry_burden",
]
ANATOMICAL_ORDER = [
    "hippocampus", "amygdala", "entorhinal", "parahippocampal",
    "temporalpole", "inferiortemporal", "middletemporal", "fusiform",
    "MTL_burden", "lateral_ventricle", "inferior_lateral_ventricle",
    "third_ventricle", "fourth_ventricle", "fifth_ventricle", "csf",
    "ventricle_CSF_burden", "frontal_pole", "caudal_anterior_cingulate",
    "isthmus_cingulate", "caudal_middle_frontal", "precentral",
    "postcentral", "precuneus", "cuneus", "medial_orbitofrontal",
    "cortical_frontocingulate_burden", "asymmetry_burden",
]
DISEASE_PROCESS_ORDER = [
    "entorhinal", "hippocampus", "parahippocampal", "amygdala",
    "temporalpole", "fusiform", "inferiortemporal", "middletemporal",
    "MTL_burden", "inferior_lateral_ventricle", "lateral_ventricle",
    "third_ventricle", "fourth_ventricle", "fifth_ventricle", "csf",
    "ventricle_CSF_burden", "isthmus_cingulate",
    "caudal_anterior_cingulate", "caudal_middle_frontal",
    "medial_orbitofrontal", "frontal_pole", "precuneus", "cuneus",
    "precentral", "postcentral", "cortical_frontocingulate_burden",
    "asymmetry_burden",
]


def tensors_from_evidence(evidence, node_names, ordering):
    """Reorder positive 27-vectors, reshape to 3x3x3, and normalize."""
    if list(node_names) != ORIGINAL_ORDER:
        if set(node_names) != set(ORIGINAL_ORDER):
            raise AssertionError("Expected fixed 27-node identity")
    lookup = {name: i for i, name in enumerate(node_names)}
    arranged = np.asarray(evidence, float)[:, [lookup[name] for name in ordering]]
    if arranged.shape[1] != 27 or np.any(arranged <= 0):
        raise ValueError("Evidence must be positive 27-vectors")
    tensors = arranged.reshape(-1, 3, 3, 3)
    return tensors / tensors.sum(axis=(1, 2, 3), keepdims=True)


BIOLOGICAL_ORDER = list(
    DISEASE_PROCESS_ORDER
)


def system(node):
    """Return the broad disease-system grouping for a node."""

    if node in BIOLOGICAL_ORDER[:9]:
        return "medial_temporal_temporal"

    if node in BIOLOGICAL_ORDER[9:16]:
        return "ventricular_csf"

    if node in BIOLOGICAL_ORDER[16:26]:
        return "cortical_frontocingulate"

    return "cross_system_asymmetry"


def channel(node):
    """Describe the evidence channel represented by a node."""

    if node == "asymmetry_burden":
        return "asymmetry_aggregate"

    if node.endswith("_burden"):
        return "system_composite"

    return "regional_abnormality"


def stage(node):
    """Return the coarse disease-stage grouping used by controls."""

    focal_core = {
        "entorhinal",
        "hippocampus",
        "parahippocampal",
        "amygdala",
    }

    distributed = {
        "MTL_burden",
        "ventricle_CSF_burden",
        "cortical_frontocingulate_burden",
        "asymmetry_burden",
        "precentral",
        "postcentral",
    }

    if node in focal_core:
        return "focal_core"

    if node in distributed:
        return "distributed_composite"

    return "regional"


def orderings(seed=314159):
    """Construct biological, reference, and label-free control orderings."""

    orders = {
        "biological_system_block": list(
            BIOLOGICAL_ORDER
        ),
        "inherited_compact": list(
            ANATOMICAL_ORDER
        ),
        "original_code": list(
            ORIGINAL_ORDER
        ),
    }

    rng = np.random.default_rng(seed)

    # System-preserving controls.
    groups = [
        BIOLOGICAL_ORDER[:9],
        BIOLOGICAL_ORDER[9:16],
        BIOLOGICAL_ORDER[16:26],
        BIOLOGICAL_ORDER[26:],
    ]

    for i in range(10):
        orders[f"system_preserving_{i}"] = [
            node
            for group in groups
            for node in rng.permutation(group)
        ]

    # Stage-preserving controls.
    stage_groups = {
        name: [
            node
            for node in BIOLOGICAL_ORDER
            if stage(node) == name
        ]
        for name in [
            "focal_core",
            "regional",
            "distributed_composite",
        ]
    }

    for i in range(10):
        orders[f"stage_preserving_{i}"] = [
            node
            for group in stage_groups.values()
            for node in rng.permutation(group)
        ]

    # Completely random controls.
    for i in range(25):
        orders[f"random_{i}"] = list(
            rng.permutation(
                BIOLOGICAL_ORDER
            )
        )

    expected = set(BIOLOGICAL_ORDER)

    for name, order in orders.items():
        if (
            len(order) != 27
            or set(order) != expected
        ):
            raise AssertionError(
                f"Invalid tensor ordering: {name}"
            )

    return orders


def _ordering_type(ordering_id):
    if ordering_id == "biological_system_block":
        return "biological"

    if ordering_id == "inherited_compact":
        return "inherited"

    if ordering_id.startswith("system_"):
        return "system_preserving"

    if ordering_id.startswith("stage_"):
        return "stage_preserving"

    if ordering_id.startswith("random_"):
        return "random"

    return "reference"


def mapping_frames(node_table):
    """Create tensor-cell and ordering manifests."""

    lookup = node_table.set_index(
        "node"
    )

    mapping_rows = []
    ordering_rows = []

    for ordering_id, order in orderings().items():
        ordering_type = _ordering_type(
            ordering_id
        )

        for position, node in enumerate(order):

            axis_1, axis_2, axis_3 = (
                np.unravel_index(
                    position,
                    (3, 3, 3),
                )
            )

            ordering_rows.append({
                "ordering_id": ordering_id,
                "ordering_type": ordering_type,
                "position": position,
                "axis_1": axis_1,
                "axis_2": axis_2,
                "axis_3": axis_3,
                "source_node": node,
                "seed": 314159,
            })

            if ordering_id != "biological_system_block":
                continue

            mapping_rows.append({
                "tensor_layout": ordering_id,
                "axis_1_index": axis_1,
                "axis_1_label": [
                    "temporal_block",
                    "ventricular_mixed_block",
                    "cortical_block",
                ][axis_1],
                "axis_2_index": axis_2,
                "axis_2_label":
                    f"within_block_triplet_{axis_2}",
                "axis_3_index": axis_3,
                "axis_3_label":
                    f"within_triplet_rank_{axis_3}",
                "tensor_cell":
                    f"({axis_1},{axis_2},{axis_3})",
                "source_node": node,
                "source_features":
                    lookup.loc[
                        node,
                        "source_features",
                    ],
                "disease_system": system(node),
                "disease_stage": stage(node),
                "evidence_channel": channel(node),
                "hemisphere":
                    "bilateral_or_aggregate",
                "mapping_rationale": (
                    "Prespecified system-block ordering; "
                    "axes 2/3 are indexing ranks, "
                    "not biological constructs."
                ),
                "is_artificial": False,
                "notes": (
                    "Ventricular block includes boundary "
                    "aggregate placements because manifest "
                    "group sizes are 9/7/10/1, not 9/9/9."
                ),
            })

    return (
        pd.DataFrame(mapping_rows),
        pd.DataFrame(ordering_rows),
    )


__all__ = [
    "ANATOMICAL_ORDER",
    "BIOLOGICAL_ORDER",
    "DISEASE_PROCESS_ORDER",
    "ORIGINAL_ORDER",
    "channel",
    "mapping_frames",
    "tensors_from_evidence",
    "orderings",
    "stage",
    "system",
    "tensors_from_evidence",
]
