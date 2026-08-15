"""
Check whether a batch of "current" images has drifted from the CamVid
training distribution, using Evidently over the summary-statistic features
computed in src/drift.py (see that module's docstring for why raw pixels
aren't used directly).

Reference set: every image in configs/config.yaml's data.train_list.
Current set:   every .png in --current-dir (defaults to the CamVid test
               split; note this actually BLOCKS by default, since CamVid's
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
from src.drift import check_drift_against_reference

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


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
        print("[drift-check] PASSED: drift detection disabled in config, passing by default.")
        sys.exit(0)

    print(f"Reference: {cfg['data']['train_list']}")
    print(f"Current:   {args.current_dir}")
    print("Extracting features and running Evidently drift report ...")
    result = check_drift_against_reference(cfg, args.current_dir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / f"drift_report_{datetime.now(UTC):%Y%m%d_%H%M%S}.html"
    result.report.save_html(str(report_path))
    print(f"Report saved to {report_path}")

    threshold = drift_cfg["drift_share_threshold"]
    msg = (
        f"drift_share={result.drift_share:.2f} "
        f"({result.drifted_columns}/{result.total_columns} columns drifted) "
        f"{'<=' if result.passed else '>'} threshold={threshold:.2f}"
    )
    print(f"[drift-check] {'PASSED' if result.passed else 'BLOCKED'}: {msg}")
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
