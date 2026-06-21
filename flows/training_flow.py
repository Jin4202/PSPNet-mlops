"""
Week 2 — Prefect 3 Training Pipeline

Flow execution order:
  load_data_task
    → prepare_model_task
      → train_model_task        (runs inside an MLflow Run)
        → evaluate_model_task
          → register_model_task (includes Validation Gate)

Usage (RunPod):
  # 1) Start worker — Terminal A (server must already be running)
  prefect worker start --pool pspnet-pool --type process

  # 2) Register deployment — one-time setup
  python flows/training_flow.py --deploy

  # 3) Trigger training with a single command — Terminal B
  prefect deployment run training-flow/pspnet-training

  # Or run directly without a Prefect server (dev/debug)
  python flows/training_flow.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import yaml
from prefect import flow, get_run_logger, task

# Add project root to sys.path so src/ is importable from flows/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import get_train_loader, get_val_loader
from src.evaluate import evaluate
from src.models.pspnet import PSPNet
from src.train import train_one_epoch


# ─────────────────────────────────────────────
# Config loader (used outside of tasks)
# ─────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Read a YAML config file and return it as a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Task 1 — Build dataloaders
# ─────────────────────────────────────────────

@task(name="load-data", retries=2, retry_delay_seconds=10)
def load_data_task(cfg: dict) -> tuple:
    """
    Create CamVid train and val dataloaders.
    retries=2 guards against transient file-system or network-mount errors.
    """
    logger = get_run_logger()
    logger.info("Loading CamVid dataloaders ...")

    train_loader = get_train_loader(cfg)
    val_loader   = get_val_loader(cfg)

    logger.info(
        f"  Train batches: {len(train_loader)} | "
        f"Val batches:   {len(val_loader)}"
    )
    return train_loader, val_loader


# ─────────────────────────────────────────────
# Task 2 — Build model / optimizer / device
# ─────────────────────────────────────────────

@task(name="prepare-model")
def prepare_model_task(cfg: dict) -> tuple:
    """
    Instantiate PSPNet, SGD optimizer, and select the compute device.

    Parameter groups:
      - backbone (layer0–layer4): base_lr
      - head (PPM + classifiers):  10 × base_lr
    This differential LR strategy lets the pre-trained backbone fine-tune
    slowly while the newly initialized head learns aggressively.

    Note: large PyTorch objects are passed directly in memory between tasks
    because we use a Process work pool (single process). This avoids
    Prefect serialization overhead for model weights.
    """
    logger = get_run_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    model = PSPNet(
        layers=50,
        num_classes=model_cfg["num_classes"],
        zoom_factor=model_cfg["zoom_factor"],
        pretrained=model_cfg["pretrained_backbone"],
    ).to(device)

    # Split parameters: backbone vs. head
    backbone_params = [
        p for name, p in model.named_parameters()
        if "cls" not in name and "ppm" not in name
    ]
    head_params = [
        p for name, p in model.named_parameters()
        if "cls" in name or "ppm" in name
    ]

    optimizer = torch.optim.SGD(
        [
            {"params": backbone_params, "lr": train_cfg["base_lr"]},
            {"params": head_params,     "lr": train_cfg["base_lr"] * 10},
        ],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )

    logger.info(
        f"Model ready | "
        f"Backbone param groups: {len(backbone_params)} | "
        f"Head param groups: {len(head_params)}"
    )
    return model, optimizer, device


# ─────────────────────────────────────────────
# Task 3 — Training loop (inside an MLflow Run)
# ─────────────────────────────────────────────

@task(name="train-model", timeout_seconds=3600 * 12)  # 12-hour hard cap
def train_model_task(
    cfg: dict,
    model,
    optimizer,
    device: str,
    train_loader,
    val_loader,
) -> tuple:
    """
    Run the full training loop and log everything to MLflow.

    Per epoch:
      - train_one_epoch  → computes train loss + mIoU, applies poly LR decay
      - evaluate         → computes val loss + mIoU on the validation set
      - MLflow metrics   → logged at each epoch step
      - checkpoint       → saved only when val mIoU improves (best model)

    Returns: (best_val_miou, mlflow_run_id)
    The run_id is passed downstream so register_model_task can reference
    the exact artifacts produced by this run.
    """
    logger = get_run_logger()
    train_cfg  = cfg["training"]
    mlflow_cfg = cfg["mlflow"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    best_val_miou  = 0.0
    best_ckpt_path = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log all hyperparameters up front
        mlflow.log_params({
            "epochs":          train_cfg["epochs"],
            "batch_size":      train_cfg["batch_size"],
            "base_lr":         train_cfg["base_lr"],
            "momentum":        train_cfg["momentum"],
            "weight_decay":    train_cfg["weight_decay"],
            "aux_loss_weight": cfg["model"]["aux_loss_weight"],
            "crop_size":       cfg["data"]["crop_size"],
            "architecture":    cfg["model"]["architecture"],
        })

        total_steps   = train_cfg["epochs"] * len(train_loader)
        current_step  = 0

        for epoch in range(1, train_cfg["epochs"] + 1):
            epoch_start = time.time()

            # --- Train one epoch ---
            train_loss, train_miou, current_step = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                cfg=cfg,
                current_step=current_step,
                total_steps=total_steps,
            )

            # --- Validate ---
            val_miou, val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                cfg=cfg,
            )

            epoch_time = time.time() - epoch_start

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_miou": train_miou,
                    "val_loss":   val_loss,
                    "val_miou":   val_miou,
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

            logger.info(
                f"Epoch [{epoch:3d}/{train_cfg['epochs']}] "
                f"train_loss={train_loss:.4f}  train_mIoU={train_miou:.4f}  "
                f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
                f"({epoch_time:.1f}s)"
            )

            # Save checkpoint only when val mIoU improves
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(
                    {
                        "epoch":       epoch,
                        "model_state": model.state_dict(),
                        "optimizer":   optimizer.state_dict(),
                        "val_miou":    val_miou,
                        "config":      cfg,
                    },
                    best_ckpt_path,
                )
                logger.info(f"  ✓ New best val_mIoU={val_miou:.4f} — checkpoint saved")

        # Log final best mIoU and upload checkpoint as MLflow artifact
        mlflow.log_metric("best_val_miou", best_val_miou)
        mlflow.log_artifact(str(best_ckpt_path), artifact_path="checkpoints")

        logger.info(f"Training complete. Best Val mIoU: {best_val_miou:.4f}")

    return best_val_miou, run_id


# ─────────────────────────────────────────────
# Task 4 — Evaluate and summarize results
# ─────────────────────────────────────────────

@task(name="evaluate-model")
def evaluate_model_task(best_val_miou: float, cfg: dict) -> dict:
    """
    Summarize training results and pre-compute the gate decision.

    Keeping evaluation separate from training makes it easy to swap in
    a test-set evaluation or additional metrics later without touching
    the training task.
    """
    logger = get_run_logger()
    gate_cfg = cfg["validation_gate"]

    result = {
        "best_val_miou": best_val_miou,
        "threshold":     gate_cfg["min_miou"],
        "gate_enabled":  gate_cfg["enabled"],
        # Gate passes when mIoU meets threshold, or when gate is disabled
        "gate_passed":   best_val_miou >= gate_cfg["min_miou"] or not gate_cfg["enabled"],
    }

    logger.info(
        f"Evaluation summary:\n"
        f"  Best Val mIoU : {best_val_miou:.4f}\n"
        f"  Gate threshold: {gate_cfg['min_miou']}\n"
        f"  Gate enabled  : {gate_cfg['enabled']}\n"
        f"  Gate passed   : {result['gate_passed']}"
    )
    return result


# ─────────────────────────────────────────────
# Task 5 — Register model (with Validation Gate)
# ─────────────────────────────────────────────

@task(name="register-model")
def register_model_task(eval_result: dict, run_id: str, cfg: dict) -> str:
    """
    Register the model in MLflow Model Registry — only if the gate passes.

    Gate passed → new version registered → returns the model URI
    Gate failed → registration blocked   → raises ValueError
                  (Prefect marks this task as FAILED, flow stops cleanly)

    Why a hard failure instead of a soft warning?
    A ValueError makes the failed run immediately visible in the Prefect UI
    and prevents a bad model from silently reaching production. In Week 4,
    the drift-triggered retraining loop checks run status before promoting.
    """
    logger     = get_run_logger()
    mlflow_cfg = cfg["mlflow"]

    # ── Validation Gate ──────────────────────────────────────────────
    if not eval_result["gate_passed"]:
        msg = (
            f"[Validation Gate BLOCKED] "
            f"Val mIoU {eval_result['best_val_miou']:.4f} "
            f"< threshold {eval_result['threshold']:.2f}. "
            f"Model NOT registered."
        )
        logger.error(msg)
        raise ValueError(msg)

    # ── Register in MLflow Model Registry ───────────────────────────
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    model_uri  = f"runs:/{run_id}/{mlflow_cfg['artifact_path']}"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=mlflow_cfg["model_name"],
    )

    logger.info(
        f"[Validation Gate PASSED] "
        f"Registered '{mlflow_cfg['model_name']}' "
        f"version {registered.version} "
        f"(mIoU={eval_result['best_val_miou']:.4f})"
    )

    return f"models:/{mlflow_cfg['model_name']}/{registered.version}"


# ─────────────────────────────────────────────
# Flow — wire tasks together
# ─────────────────────────────────────────────

@flow(
    name="training-flow",
    description="PSPNet end-to-end training pipeline with MLflow tracking and Validation Gate.",
    log_prints=True,
)
def training_flow(config_path: str = "configs/config.yaml"):
    """
    Single entry point for the entire training pipeline.
    The config_path parameter can be overridden at runtime from the CLI:

      prefect deployment run training-flow/pspnet-training \\
        --param config_path=configs/config_fast.yaml

    In Week 4, the drift detection service will call this flow via the
    Prefect API without any code changes — just a POST to:
      /api/deployments/{deployment_id}/create_flow_run
    """
    logger = get_run_logger()
    logger.info(f"Config: {config_path}")

    cfg = load_config(config_path)

    # Tasks execute sequentially; each receives the output of the previous one
    train_loader, val_loader  = load_data_task(cfg)
    model, optimizer, device  = prepare_model_task(cfg)
    best_val_miou, run_id     = train_model_task(cfg, model, optimizer, device, train_loader, val_loader)
    eval_result               = evaluate_model_task(best_val_miou, cfg)
    model_version_uri         = register_model_task(eval_result, run_id, cfg)

    logger.info(f"Pipeline complete. Model URI: {model_version_uri}")
    return model_version_uri


# ─────────────────────────────────────────────
# Deployment registration
# ─────────────────────────────────────────────

def create_deployment():
    """
    Register a Process work pool deployment with the Prefect server.
    Run this once after the server and work pool are ready.

    Prerequisites:
      prefect server start
      prefect work-pool create pspnet-pool --type process

    Week 4 note:
      The registered deployment ID can be retrieved with:
        prefect deployment ls
      Evidently's drift callback will POST to that deployment to trigger
      automatic retraining without any manual intervention.
    """
    training_flow.from_source(
        source=str(PROJECT_ROOT),
        entrypoint="flows/training_flow.py:training_flow",
    ).deploy(
        name="pspnet-training",
        work_pool_name="pspnet-pool",
        parameters={"config_path": "configs/config.yaml"},
        tags=["pspnet", "mlops", "week2"],
        description="PSPNet training pipeline — triggered manually or by drift detection.",
    )
    print("Deployment 'pspnet-training' registered.")
    print("Start worker : prefect worker start --pool pspnet-pool --type process")
    print("Trigger run  : prefect deployment run training-flow/pspnet-training")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Register Prefect Deployment (requires prefect server to be running)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML",
    )
    args = parser.parse_args()

    if args.deploy:
        create_deployment()
    else:
        # Direct execution — useful for local dev without a Prefect server
        training_flow(config_path=args.config)