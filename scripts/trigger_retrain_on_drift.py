"""
Run the drift check (src/drift.py::check_drift_against_reference) and, if it's
blocked, trigger a retrain by calling Prefect's run_deployment API against
training-flow/pspnet-training (see flows/training_flow.py::create_deployment).

This does not wait for or check the training run's outcome — that's a
separate concern already handled inside the flow itself (its own validation
gate and Champion-Challenger promotion decide whether the retrained model is
any good). This script's only job is deciding whether a retrain should start.

Exit code 0 = no drift, or retrain successfully triggered.
Exit code 1 = drift detected but the retrain could not be triggered (Prefect
              unreachable) — this needs attention, unlike a clean "no drift".
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import _is_reachable, load_config
from src.drift import check_drift_against_reference

DEPLOYMENT_NAME = "training-flow/pspnet-training"


def maybe_trigger_retrain(cfg: dict, current_dir: str, config_path: str) -> str:
    result = check_drift_against_reference(cfg, current_dir)

    if result.passed:
        return "no drift, retrain not triggered"

    prefect_api_url = os.environ.get("PREFECT_API_URL", "")
    if not _is_reachable(prefect_api_url):
        return f"drift detected but Prefect unreachable at '{prefect_api_url}', retrain NOT triggered"

    from prefect.deployments import run_deployment

    run_deployment(
        DEPLOYMENT_NAME,
        parameters={"config_path": config_path},
        timeout=0,
    )
    return "drift detected, retrain triggered"


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
    status = maybe_trigger_retrain(cfg, args.current_dir, args.config)
    print(f"[drift-retrain-trigger] {status}")
    sys.exit(1 if "NOT triggered" in status else 0)


if __name__ == "__main__":
    main()
