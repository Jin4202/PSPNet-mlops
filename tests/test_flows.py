"""
Smoke tests for flows/training_flow.py.

Prefect tasks are invoked via `.fn(...)` so no Prefect server is required.
The training-path tests use a tiny hand-rolled model (real Conv2d params,
so loss.backward() has something to differentiate) instead of the real
PSPNet, and a local sqlite-backed MLflow tracking store instead of a live
server — this is about catching wiring bugs (bad config keys, mismatched
function signatures, wrong return shapes), not validating model quality.
"""
import logging

import pytest
import torch
from torch import nn

from flows import training_flow

IMG_SIZE = 8
NUM_CLASSES = 2


@pytest.fixture(autouse=True)
def _no_prefect_run_context(monkeypatch):
    """
    Tasks are invoked via `.fn(...)`, bypassing real Prefect orchestration,
    so there's no active run context for get_run_logger() to attach to.
    Swap in a plain logger instead of running an ephemeral Prefect server.
    """
    monkeypatch.setattr(training_flow, "get_run_logger", lambda: logging.getLogger("test"))


class _FakeFlowModel(nn.Module):
    """Preserves H, W (like PSPNet's upsampled output) and has real params."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x):
        main = self.conv(x)
        if self.training:
            return main, main.clone()
        return main


def _make_batch(batch_size, all_zero_labels=False):
    imgs = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE)
    if all_zero_labels:
        lbls = torch.zeros(batch_size, IMG_SIZE, IMG_SIZE, dtype=torch.long)
    else:
        lbls = torch.randint(0, NUM_CLASSES, (batch_size, IMG_SIZE, IMG_SIZE))
    return imgs, lbls


def _flow_cfg(tmp_path, gate_enabled=False):
    return {
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "base_lr": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0001,
            "aux_weight": 0.4,
            "poly_power": 0.9,
            "save_dir": str(tmp_path / "checkpoints"),
        },
        "model": {"arch": "pspnet", "num_classes": NUM_CLASSES},
        "augmentation": {"crop_size": IMG_SIZE},
        "data": {"num_classes": NUM_CLASSES, "ignore_label": 255},
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "experiment_name": "pspnet-flow-test",
            "model_name": "pspnet-flow-test-model",
            "artifact_path": "model",
        },
        "validation_gate": {"min_miou": 0.55, "enabled": gate_enabled},
    }


def test_prepare_model_task_builds_five_param_groups():
    cfg = {
        "model": {
            "arch": "pspnet",
            "num_classes": NUM_CLASSES,
            "zoom_factor": 8,
            "ppm_bins": [1, 2, 3, 6],
            "ppm_dim": 512,
        },
        "training": {"base_lr": 0.01, "momentum": 0.9, "weight_decay": 0.0001},
        "data": {"ignore_label": 255},
    }

    _model, optimizer, criterion, device = training_flow.prepare_model_task.fn(cfg)

    assert device == "cpu"  # no GPU in CI
    assert len(optimizer.param_groups) == 5
    assert optimizer.param_groups[-1]["lr"] == cfg["training"]["base_lr"] * 10
    assert isinstance(criterion, nn.CrossEntropyLoss)
    assert criterion.ignore_index == 255


def test_evaluate_model_task_gate_logic():
    cfg = {"validation_gate": {"min_miou": 0.55, "enabled": True}}

    passed = training_flow.evaluate_model_task.fn(best_val_miou=0.6, cfg=cfg)
    failed = training_flow.evaluate_model_task.fn(best_val_miou=0.4, cfg=cfg)

    assert passed["gate_passed"] is True
    assert failed["gate_passed"] is False


def test_register_model_task_blocks_when_gate_failed():
    eval_result = {"gate_passed": False, "best_val_miou": 0.1, "threshold": 0.55}

    try:
        training_flow.register_model_task.fn(eval_result, run_id="fake-run-id", cfg={"mlflow": {}})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_training_pipeline_smoke(tmp_path):
    """train_model_task -> evaluate_model_task -> register_model_task, chained."""
    cfg = _flow_cfg(tmp_path, gate_enabled=False)  # disabled = always passes

    model = _FakeFlowModel(NUM_CLASSES)
    # Bias predictions toward class 0 and use all-zero val labels so mIoU is
    # deterministically > 0 and a checkpoint is guaranteed to be saved.
    with torch.no_grad():
        model.conv.bias[:] = torch.tensor([10.0, -10.0])

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["training"]["base_lr"],
        momentum=cfg["training"]["momentum"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_label"])
    train_loader = [_make_batch(2), _make_batch(2)]
    val_loader = [_make_batch(1, all_zero_labels=True), _make_batch(1, all_zero_labels=True)]

    best_val_miou, run_id = training_flow.train_model_task.fn(
        cfg, model, optimizer, criterion, "cpu", train_loader, val_loader
    )

    assert isinstance(best_val_miou, float)
    assert best_val_miou > 0.0
    ckpt_path = tmp_path / "checkpoints" / "best.pth"
    assert ckpt_path.exists()
    saved = torch.load(ckpt_path, map_location="cpu")
    assert "state_dict" in saved

    eval_result = training_flow.evaluate_model_task.fn(best_val_miou, cfg)
    assert eval_result["gate_passed"] is True

    model_uri = training_flow.register_model_task.fn(eval_result, run_id, cfg)
    assert model_uri == f"models:/{cfg['mlflow']['model_name']}/1"
