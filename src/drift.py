"""
Input-distribution drift detection for CamVid images.

Evidently needs numeric feature columns, not raw images, so each image is
reduced to a handful of cheap, explainable summary statistics (per-channel
mean/std, overall brightness, a Laplacian-variance sharpness proxy) rather
than a learned embedding. This is fast (CPU-only, no model forward pass) and
good enough to catch the kind of distribution shift a demo needs to show
(darkened/blurred/OOD input images), without adding a second model to
maintain just for drift checking.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from PIL import Image


def _image_features(path: str) -> dict:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = arr.mean(axis=2)

    # Discrete Laplacian (kernel [[0,1,0],[1,-4,1],[0,1,0]]) via slicing —
    # its variance is a standard blur/sharpness proxy, no scipy/cv2 needed.
    laplacian = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )

    return {
        "mean_r": float(r.mean()), "std_r": float(r.std()),
        "mean_g": float(g.mean()), "std_g": float(g.std()),
        "mean_b": float(b.mean()), "std_b": float(b.std()),
        "brightness": float(gray.mean()),
        "sharpness": float(laplacian.var()),
    }


def extract_features(image_paths: list[str]) -> pd.DataFrame:
    """One row of summary statistics per image."""
    return pd.DataFrame(_image_features(p) for p in image_paths)


@dataclass
class DriftResult:
    passed: bool
    drift_share: float
    drifted_columns: int
    total_columns: int
    report: object  # evidently.core.report.Snapshot — caller can .save_html(...)


def compute_drift(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, drift_share_threshold: float
) -> DriftResult:
    """
    Runs Evidently's DataDriftPreset over the extracted feature columns and
    compares the resulting drift share (fraction of columns that drifted)
    against drift_share_threshold — mirrors the explicit compute-then-compare
    pattern scripts/check_validation_gate.py already uses for the mIoU gate.
    """
    data_definition = DataDefinition()
    reference = Dataset.from_pandas(reference_df, data_definition=data_definition)
    current = Dataset.from_pandas(current_df, data_definition=data_definition)

    report = Report(metrics=[DataDriftPreset(drift_share=drift_share_threshold)])
    result = report.run(reference_data=reference, current_data=current)

    for metric in result.dict()["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            share = metric["value"]["share"]
            count = metric["value"]["count"]
            break
    else:
        raise RuntimeError("Evidently report did not include a DriftedColumnsCount metric.")

    return DriftResult(
        passed=share <= drift_share_threshold,
        drift_share=share,
        drifted_columns=int(count),
        total_columns=len(reference_df.columns),
        report=result,
    )
