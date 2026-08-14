from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd


def atomic_csv(df, path):
    path = Path(path)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    df.to_csv(
        tmp,
        index=False,
    )

    os.replace(tmp, path)


def atomic_json(data, path):
    path = Path(path)

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )

    os.replace(tmp, path)


def read_checkpoint(path):
    path = Path(path)

    if not path.exists():
        return None

    return json.loads(
        path.read_text()
    )


def ensure_output_directory(
    path,
    *,
    resume=False,
    overwrite=False,
):
    path = Path(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        any(path.iterdir())
        and not resume
        and not overwrite
    ):
        raise RuntimeError(
            "Non-empty output directory; "
            "use --resume or --overwrite."
        )

    return path