from __future__ import annotations

import platform
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn

from .config import (
    experiment_config_hash,
    hash_obj,
)
from .official_backend import backend_info


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return None


def build_provenance(
    args,
    node_table,
    tensor_mapping,
):
    backend = backend_info()

    return {
        "arguments": vars(args),
        "config_hash":
            experiment_config_hash(args),
        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "repository_git_commit":
            git_commit(),
        "versions": {
            "python":
                platform.python_version(),
            "numpy":
                np.__version__,
            "pandas":
                pd.__version__,
            "scipy":
                scipy.__version__,
            "sklearn":
                sklearn.__version__,
        },
        "official_mba_module_path":
            backend["module_path"],
        "official_call":
            backend["official_call"],
        "node_mapping_hash":
            hash_obj(
                node_table.to_dict(
                    "records"
                )
            ),
        "tensor_mapping_hash":
            hash_obj(
                tensor_mapping.to_dict(
                    "records"
                )
            ),
    }