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
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from PIL import Image

from src.data.dataset import parse_list


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


def check_drift_against_reference(cfg: dict, current_dir: str) -> DriftResult:
    """
    Builds the reference set from cfg['data']['train_list'] and checks every
    .png in current_dir against it. Shared by scripts/check_drift.py,
    scripts/trigger_retrain_on_drift.py, and scripts/demo_drift_scenario.py
    so the reference-building logic lives in exactly one place.
    """
    data_cfg = cfg["data"]
    reference_paths = [
        img_path
        for img_path, _lbl_path in parse_list(data_cfg["train_list"], data_root=data_cfg["dataset_path"].rstrip("/"))
    ]

    current_paths = sorted(str(p) for p in Path(current_dir).glob("*.png"))
    if not current_paths:
        raise FileNotFoundError(f"No .png images found in '{current_dir}'")

    reference_df = extract_features(reference_paths)
    current_df = extract_features(current_paths)
    threshold = cfg["drift_detection"]["drift_share_threshold"]
    return compute_drift(reference_df, current_df, threshold)
