"""
Block a deployment unless the model version about to be served clears the
mIoU validation gate defined in configs/config.yaml (validation_gate.min_miou).

This re-checks the gate against the MLflow Model Registry at deploy time,
independent of the gate already enforced at registration time in
flows/training_flow.py::register_model_task. A version could in principle
be promoted to a stage after registration, so deploy time is when it
actually matters.

Exit code 0 = gate passed (or disabled), 1 = blocked.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import _is_reachable, get_best_val_miou, load_config


def check_gate(cfg: dict, stage: str) -> tuple[bool, str]:
    gate_cfg = cfg["validation_gate"]
    if not gate_cfg["enabled"]:
        return True, "Validation gate disabled in config — passing by default."

    model_name = cfg["mlflow"]["model_name"]
    miou, version = get_best_val_miou(model_name, stage)
    threshold = gate_cfg["min_miou"]
    passed = miou >= threshold
    msg = (
        f"Model '{model_name}' v{version} best_val_miou={miou:.4f} "
        f"{'>=' if passed else '<'} threshold={threshold:.2f}"
    )
    return passed, msg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", default=os.environ.get("MODEL_STAGE_OR_VERSION", "latest"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])

    if not _is_reachable(tracking_uri):
        print(f"[validation-gate] BLOCKED: MLflow tracking server unreachable at '{tracking_uri}'")
        sys.exit(1)

    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    try:
        passed, msg = check_gate(cfg, args.stage)
    except LookupError as exc:
        print(f"[validation-gate] BLOCKED: {exc}")
        sys.exit(1)

    print(f"[validation-gate] {'PASSED' if passed else 'BLOCKED'}: {msg}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
