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
import sys
import time
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, get_run_logger, task
from torch import nn

# Add project root to sys.path so src/ is importable from flows/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.model_loader import NoChampionError, get_best_val_miou
from src.data.dataset import build_loaders
from src.evaluate import evaluate
from src.models.pspnet import build_model
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

    train_loader, val_loader = build_loaders(cfg)

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
    Instantiate PSPNet, SGD optimizer, criterion, and select the compute device.

    Optimizer is built via model.param_groups(base_lr) — 5 groups
    (layer0+1, layer2, layer3, layer4 at base_lr; PPM+classifiers at
    10 x base_lr) — the same convention src.train.train_one_epoch assumes
    when applying poly LR decay. This differential LR strategy lets the
    pre-trained backbone fine-tune slowly while the newly initialized head
    learns aggressively.

    Note: large PyTorch objects are passed directly in memory between tasks
    because we use a Process work pool (single process). This avoids
    Prefect serialization overhead for model weights.
    """
    logger = get_run_logger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    train_cfg = cfg["training"]

    model = build_model(cfg).to(device)

    optimizer = torch.optim.SGD(
        model.param_groups(train_cfg["base_lr"]),
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_label"])

    logger.info(f"Model ready | param groups: {len(optimizer.param_groups)}")
    return model, optimizer, criterion, device


# ─────────────────────────────────────────────
# Task 3 — Training loop (inside an MLflow Run)
# ─────────────────────────────────────────────

@task(name="train-model", timeout_seconds=3600 * 12)  # 12-hour hard cap
def train_model_task(
    cfg: dict,
    model,
    optimizer,
    criterion,
    device: str,
    train_loader,
    val_loader,
) -> tuple:
    """
    Run the full training loop and log everything to MLflow.

    Per epoch:
      - train_one_epoch  → computes train loss, applies poly LR decay
      - evaluate         → computes val mIoU/allacc on the validation set
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

    save_dir = Path(train_cfg["save_dir"])
    if not save_dir.is_absolute():
        save_dir = PROJECT_ROOT / save_dir

    best_val_miou  = 0.0
    best_ckpt_path = save_dir / "best.pth"
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log all hyperparameters up front
        mlflow.log_params({
            "epochs":       train_cfg["epochs"],
            "batch_size":   train_cfg["batch_size"],
            "base_lr":      train_cfg["base_lr"],
            "momentum":     train_cfg["momentum"],
            "weight_decay": train_cfg["weight_decay"],
            "aux_weight":   train_cfg["aux_weight"],
            "crop_size":    cfg["augmentation"]["crop_size"],
            "arch":         cfg["model"]["arch"],
        })

        total_steps   = train_cfg["epochs"] * len(train_loader)
        current_step  = 0

        for epoch in range(1, train_cfg["epochs"] + 1):
            epoch_start = time.time()

            # --- Train one epoch ---
            train_loss, current_step, lr = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                cfg=cfg,
                current_step=current_step,
                total_steps=total_steps,
            )

            # --- Validate ---
            val_result = evaluate(model=model, loader=val_loader, cfg=cfg, device=device)
            val_miou   = val_result["miou"]

            epoch_time = time.time() - epoch_start

            # Log metrics to MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_miou":   val_miou,
                    "val_allacc": val_result["allacc"],
                    "lr":         lr,
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

            logger.info(
                f"Epoch [{epoch:3d}/{train_cfg['epochs']}] "
                f"train_loss={train_loss:.4f}  "
                f"val_mIoU={val_miou:.4f}  val_allacc={val_result['allacc']:.4f}  "
                f"({epoch_time:.1f}s)"
            )

            # Save checkpoint only when val mIoU improves
            if val_miou > best_val_miou:
                best_val_miou = val_miou
                torch.save(
                    {
                        "epoch":       epoch,
                        "state_dict":  model.state_dict(),
                        "optimizer":   optimizer.state_dict(),
                        "miou":        val_miou,
                        "config":      cfg,
                    },
                    best_ckpt_path,
                )
                logger.info(f"  ✓ New best val_mIoU={val_miou:.4f} — checkpoint saved")

        # Log final best mIoU, then reload the best checkpoint and log it as
        # an MLflow model (not just a raw file) so register_model_task can
        # register it via runs:/{run_id}/{mlflow_cfg['artifact_path']}
        mlflow.log_metric("best_val_miou", best_val_miou)
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["state_dict"])
        mlflow.pytorch.log_model(model, mlflow_cfg["artifact_path"])

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
# Task 6 — Champion-Challenger promotion
# ─────────────────────────────────────────────

@task(name="promote-champion")
def promote_champion_task(eval_result: dict, run_id: str, registered_version: str, cfg: dict) -> str:
    """
    Promote the newly registered version to the "Production" stage, but only
    if it beats the current champion's best_val_miou (or there is no
    champion yet). A model that clears the validation gate still isn't
    necessarily better than what's already serving — this is that check.

    Returns a status string: "promoted (first champion)", "promoted
    (beat prior champion)", or "not promoted (challenger did not beat champion)".

    If a champion version exists at the stage but its run has no
    best_val_miou metric logged, this raises ChampionMetricMissingError
    rather than silently promoting the challenger — a champion that can't be
    compared against is a data problem to fix, not something to promote past.
    """
    logger = get_run_logger()
    cc_cfg = cfg["champion_challenger"]
    model_name = cfg["mlflow"]["model_name"]
    stage = cc_cfg["production_stage"]

    if not cc_cfg["enabled"]:
        logger.info("Champion-Challenger disabled in config — skipping promotion.")
        return "not promoted (champion-challenger disabled)"

    challenger_miou = eval_result["best_val_miou"]
    client = MlflowClient()

    try:
        champion_miou, champion_version = get_best_val_miou(model_name, stage)
    except NoChampionError:
        client.transition_model_version_stage(
            model_name, registered_version, stage=stage, archive_existing_versions=True
        )
        logger.info(
            f"[Champion-Challenger] No prior '{stage}' version — "
            f"v{registered_version} (mIoU={challenger_miou:.4f}) promoted as first champion."
        )
        return "promoted (first champion)"

    if challenger_miou > champion_miou:
        client.transition_model_version_stage(
            model_name, registered_version, stage=stage, archive_existing_versions=True
        )
        logger.info(
            f"[Champion-Challenger] v{registered_version} (mIoU={challenger_miou:.4f}) "
            f"beat champion v{champion_version} (mIoU={champion_miou:.4f}) — promoted."
        )
        return "promoted (beat prior champion)"

    logger.info(
        f"[Champion-Challenger] v{registered_version} (mIoU={challenger_miou:.4f}) "
        f"did not beat champion v{champion_version} (mIoU={champion_miou:.4f}) — not promoted."
    )
    return "not promoted (challenger did not beat champion)"


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
    train_loader, val_loader          = load_data_task(cfg)
    model, optimizer, criterion, device = prepare_model_task(cfg)
    best_val_miou, run_id             = train_model_task(
        cfg, model, optimizer, criterion, device, train_loader, val_loader
    )
    eval_result               = evaluate_model_task(best_val_miou, cfg)
    model_version_uri         = register_model_task(eval_result, run_id, cfg)
    registered_version        = model_version_uri.rsplit("/", 1)[-1]
    promotion_status          = promote_champion_task(eval_result, run_id, registered_version, cfg)

    logger.info(f"Pipeline complete. Model URI: {model_version_uri} ({promotion_status})")
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