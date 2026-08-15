"""
Two things live here:

1. scripts/trigger_retrain_on_drift.py's decision logic, with all collaborators
   mocked (this must never touch a real Prefect API in tests).
2. The local, fast half of "drift → retrain → evaluate → deploy": train → evaluate
   → register → promote, chained in one test, using a real temporary
   sqlite-backed MLflow store (same pattern as tests/test_flows.py and
   tests/test_champion_challenger.py). "Deploy" itself is CI/CD's job, already
   covered elsewhere, and out of scope for a pytest.
"""
import logging

import pytest
import torch
from mlflow.tracking import MlflowClient
from torch import nn

from flows import training_flow
from scripts import trigger_retrain_on_drift as trigger
from tests.test_flows import NUM_CLASSES, _FakeFlowModel, _flow_cfg, _make_batch


@pytest.fixture(autouse=True)
def _no_prefect_run_context(monkeypatch):
    monkeypatch.setattr(training_flow, "get_run_logger", lambda: logging.getLogger("test"))


class _FakeDriftResult:
    def __init__(self, passed):
        self.passed = passed


def test_no_drift_does_not_trigger_retrain(monkeypatch):
    monkeypatch.setattr(trigger, "check_drift_against_reference", lambda cfg, current_dir: _FakeDriftResult(True))
    calls = []
    monkeypatch.setattr("prefect.deployments.run_deployment", lambda *a, **k: calls.append((a, k)))

    status = trigger.maybe_trigger_retrain({}, "some/dir", "configs/config.yaml")

    assert status == "no drift, retrain not triggered"
    assert calls == []


def test_drift_blocked_and_prefect_reachable_triggers_retrain(monkeypatch):
    monkeypatch.setattr(trigger, "check_drift_against_reference", lambda cfg, current_dir: _FakeDriftResult(False))
    monkeypatch.setattr(trigger, "_is_reachable", lambda uri: True)
    calls = []
    monkeypatch.setattr("prefect.deployments.run_deployment", lambda *a, **k: calls.append((a, k)))
    monkeypatch.setenv("PREFECT_API_URL", "http://fake-prefect:4200/api")

    status = trigger.maybe_trigger_retrain({}, "some/dir", "configs/config.yaml")

    assert status == "drift detected, retrain triggered"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == trigger.DEPLOYMENT_NAME
    assert kwargs["parameters"] == {"config_path": "configs/config.yaml"}


def test_drift_blocked_and_prefect_unreachable_does_not_trigger(monkeypatch):
    monkeypatch.setattr(trigger, "check_drift_against_reference", lambda cfg, current_dir: _FakeDriftResult(False))
    monkeypatch.setattr(trigger, "_is_reachable", lambda uri: False)
    calls = []
    monkeypatch.setattr("prefect.deployments.run_deployment", lambda *a, **k: calls.append((a, k)))

    status = trigger.maybe_trigger_retrain({}, "some/dir", "configs/config.yaml")

    assert "NOT triggered" in status
    assert calls == []


def test_full_local_chain_train_evaluate_register_promote(tmp_path):
    """train_model_task -> evaluate_model_task -> register_model_task -> promote_champion_task."""
    cfg = _flow_cfg(tmp_path, gate_enabled=False)
    cfg["champion_challenger"] = {"enabled": True, "production_stage": "Production"}

    model = _FakeFlowModel(NUM_CLASSES)
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
    eval_result = training_flow.evaluate_model_task.fn(best_val_miou, cfg)
    assert eval_result["gate_passed"] is True

    model_uri = training_flow.register_model_task.fn(eval_result, run_id, cfg)
    registered_version = model_uri.rsplit("/", 1)[-1]

    status = training_flow.promote_champion_task.fn(eval_result, run_id, registered_version, cfg)

    assert status == "promoted (first champion)"
    versions = MlflowClient().get_latest_versions(cfg["mlflow"]["model_name"], stages=["Production"])
    assert len(versions) == 1
    assert str(versions[0].version) == registered_version
