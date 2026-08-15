"""
Tests for flows/training_flow.py::promote_champion_task.

Uses a real temporary sqlite-backed MLflow tracking store (same pattern as
tests/test_flows.py) rather than mocks, so the actual MLflow Model Registry
stage-transition behavior is what's under test.
"""
import logging

import mlflow
import mlflow.pytorch
import pytest
from mlflow.tracking import MlflowClient
from torch import nn

from app.model_loader import ChampionMetricMissingError
from flows import training_flow


@pytest.fixture(autouse=True)
def _no_prefect_run_context(monkeypatch):
    monkeypatch.setattr(training_flow, "get_run_logger", lambda: logging.getLogger("test"))


def _register_version(model_name: str, miou: float) -> tuple[str, str]:
    """Create a real MLflow run (with a logged model + best_val_miou metric) and register it."""
    with mlflow.start_run() as run:
        mlflow.log_metric("best_val_miou", miou)
        mlflow.pytorch.log_model(nn.Conv2d(3, 2, 1), "model")
        run_id = run.info.run_id
    registered = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    return registered.version, run_id


def _register_version_without_metric(model_name: str) -> tuple[str, str]:
    """Same as _register_version, but the run never logs best_val_miou (e.g. a version
    promoted manually outside this flow, predating metric-logging)."""
    with mlflow.start_run() as run:
        mlflow.pytorch.log_model(nn.Conv2d(3, 2, 1), "model")
        run_id = run.info.run_id
    registered = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    return registered.version, run_id


def _cfg(tmp_path, model_name: str, enabled: bool = True) -> dict:
    return {
        "mlflow": {
            "tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}",
            "model_name": model_name,
        },
        "champion_challenger": {"enabled": enabled, "production_stage": "Production"},
    }


def _production_version(model_name: str) -> str:
    versions = MlflowClient().get_latest_versions(model_name, stages=["Production"])
    assert len(versions) == 1
    return versions[0].version


def test_first_version_becomes_champion_unconditionally(tmp_path):
    model_name = "cc-test-first"
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    version, run_id = _register_version(model_name, miou=0.50)
    cfg = _cfg(tmp_path, model_name)

    status = training_flow.promote_champion_task.fn(
        {"best_val_miou": 0.50}, run_id, version, cfg
    )

    assert status == "promoted (first champion)"
    assert _production_version(model_name) == version


def test_better_challenger_replaces_champion(tmp_path):
    model_name = "cc-test-better"
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = _cfg(tmp_path, model_name)

    v1, run1 = _register_version(model_name, miou=0.50)
    training_flow.promote_champion_task.fn({"best_val_miou": 0.50}, run1, v1, cfg)

    v2, run2 = _register_version(model_name, miou=0.65)
    status = training_flow.promote_champion_task.fn({"best_val_miou": 0.65}, run2, v2, cfg)

    assert status == "promoted (beat prior champion)"
    assert _production_version(model_name) == v2


def test_worse_challenger_is_not_promoted(tmp_path):
    model_name = "cc-test-worse"
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = _cfg(tmp_path, model_name)

    v1, run1 = _register_version(model_name, miou=0.60)
    training_flow.promote_champion_task.fn({"best_val_miou": 0.60}, run1, v1, cfg)

    v2, run2 = _register_version(model_name, miou=0.45)
    status = training_flow.promote_champion_task.fn({"best_val_miou": 0.45}, run2, v2, cfg)

    assert status == "not promoted (challenger did not beat champion)"
    assert _production_version(model_name) == v1


def test_champion_with_missing_metric_blocks_promotion(tmp_path):
    model_name = "cc-test-missing-metric"
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = _cfg(tmp_path, model_name)

    # A "champion" that exists at Production but was never run through this flow
    # (e.g. promoted manually), so its run has no best_val_miou metric.
    champion_version, _ = _register_version_without_metric(model_name)
    MlflowClient().transition_model_version_stage(model_name, champion_version, stage="Production")

    challenger_version, challenger_run = _register_version(model_name, miou=0.01)

    with pytest.raises(ChampionMetricMissingError):
        training_flow.promote_champion_task.fn(
            {"best_val_miou": 0.01}, challenger_run, challenger_version, cfg
        )

    # The unmeasurable champion must still be the one at Production, not the challenger.
    assert _production_version(model_name) == champion_version


def test_disabled_skips_promotion(tmp_path):
    model_name = "cc-test-disabled"
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    cfg = _cfg(tmp_path, model_name, enabled=False)

    version, run_id = _register_version(model_name, miou=0.50)
    status = training_flow.promote_champion_task.fn(
        {"best_val_miou": 0.50}, run_id, version, cfg
    )

    assert status == "not promoted (champion-challenger disabled)"
    assert MlflowClient().get_latest_versions(model_name, stages=["Production"]) == []
