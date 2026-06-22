import os
import socket
from urllib.parse import urlparse

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = "configs/config.yaml") -> dict:
    path = config_path if os.path.isabs(config_path) else os.path.join(PROJECT_ROOT, config_path)
    with open(path) as f:
        return yaml.safe_load(f)


def _is_reachable(uri: str, timeout: float = 2.0) -> bool:
    """
    Quick TCP probe so a missing/unreachable MLflow server fails in ~2s
    instead of minutes. mlflow's HTTP client retries connection errors
    with exponential backoff internally, which would otherwise stall
    container startup and trip the Docker/Cloud Run health check.
    """
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError:
        return False


def load_model(cfg: dict, device: torch.device):
    """
    Load the PSPNet model for inference.

    Resolution order:
      1. MLflow Model Registry — models:/<model_name>/<MODEL_STAGE_OR_VERSION>
         (MODEL_STAGE_OR_VERSION env var, defaults to "latest").
         Tracking URI: MLFLOW_TRACKING_URI env var, else configs/config.yaml.
      2. Local checkpoint file — CHECKPOINT_PATH env var, defaults to
         checkpoints/best.pth (the file src/train.py writes)

    Falling back to a local checkpoint keeps the API usable in dev/CI
    environments where no MLflow server is reachable.
    """
    model_name = cfg["mlflow"]["model_name"]
    stage = os.environ.get("MODEL_STAGE_OR_VERSION", "latest")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])

    mlflow_error: Exception | None = None
    if _is_reachable(tracking_uri):
        try:
            import mlflow
            import mlflow.pytorch

            mlflow.set_tracking_uri(tracking_uri)
            model_uri = f"models:/{model_name}/{stage}"
            model = mlflow.pytorch.load_model(model_uri, map_location=device)
            model.to(device).eval()
            return model, f"mlflow:{model_uri}"
        except Exception as exc:
            mlflow_error = exc
    else:
        mlflow_error = ConnectionError(f"MLflow tracking server unreachable at '{tracking_uri}'")

    from src.models.pspnet import build_model

    ckpt_path = os.environ.get("CHECKPOINT_PATH", os.path.join(PROJECT_ROOT, "checkpoints", "best.pth"))
    if not os.path.isfile(ckpt_path):
        raise RuntimeError(
            f"Could not load model from MLflow ({mlflow_error}) "
            f"and no checkpoint found at '{ckpt_path}'."
        ) from mlflow_error

    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, f"checkpoint:{ckpt_path}"
