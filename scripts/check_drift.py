"""
Check whether a batch of "current" images has drifted from the CamVid
training distribution, using Evidently over the summary-statistic features
computed in src/drift.py (see that module's docstring for why raw pixels
aren't used directly).

Reference set: every image in configs/config.yaml's data.train_list.
Current set:   every .png in --current-dir (defaults to the CamVid test
               split — note this actually BLOCKS by default: CamVid's
               train/test split is a real, measurable distribution shift
               (different video sequences, ~8-9/255 brightness difference),
               not a same-distribution sanity check. Point --current-dir at
               a random sample of data/camvid/train/ instead for a clean
               PASSED baseline, or at deliberately corrupted images to
               demonstrate a clearer drift case.)

Saves an HTML report to results/drift_report_<timestamp>.html either way.
Exit code 0 = drift share under threshold (or check disabled), 1 = blocked.
"""
import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import load_config
from src.data.dataset import parse_list
from src.drift import compute_drift, extract_features

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_reference_paths(cfg: dict) -> list[str]:
    data_cfg = cfg["data"]
    pairs = parse_list(data_cfg["train_list"], data_root=data_cfg["dataset_path"].rstrip("/"))
    return [img_path for img_path, _lbl_path in pairs]


def build_current_paths(current_dir: str) -> list[str]:
    paths = sorted(str(p) for p in Path(current_dir).glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No .png images found in '{current_dir}'")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--current-dir",
        default="data/camvid/test",
        help="Directory of images to check for drift against the training set",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    drift_cfg = cfg["drift_detection"]

    if not drift_cfg["enabled"]:
        print("[drift-check] PASSED: drift detection disabled in config — passing by default.")
        sys.exit(0)

    threshold = drift_cfg["drift_share_threshold"]

    print(f"Reference: {cfg['data']['train_list']}")
    reference_paths = build_reference_paths(cfg)
    print(f"  {len(reference_paths)} images")

    print(f"Current:   {args.current_dir}")
    current_paths = build_current_paths(args.current_dir)
    print(f"  {len(current_paths)} images")

    print("Extracting features ...")
    reference_df = extract_features(reference_paths)
    current_df = extract_features(current_paths)

    print("Running Evidently drift report ...")
    result = compute_drift(reference_df, current_df, threshold)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"drift_report_{datetime.now(UTC):%Y%m%d_%H%M%S}.html"
    result.report.save_html(str(report_path))
    print(f"Report saved to {report_path}")

    msg = (
        f"drift_share={result.drift_share:.2f} "
        f"({result.drifted_columns}/{result.total_columns} columns drifted) "
        f"{'<=' if result.passed else '>'} threshold={threshold:.2f}"
    )
    print(f"[drift-check] {'PASSED' if result.passed else 'BLOCKED'}: {msg}")
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
