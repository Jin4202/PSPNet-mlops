import pytest

from scripts import check_validation_gate as gate


class FakeVersion:
    def __init__(self, version, run_id):
        self.version = version
        self.run_id = run_id


class FakeRun:
    def __init__(self, metrics):
        self.data = type("Data", (), {"metrics": metrics})()


class FakeClient:
    def __init__(self, versions_by_stage=None, versions_by_name=None, runs=None):
        self._versions_by_stage = versions_by_stage or {}
        self._versions_by_name = versions_by_name or []
        self._runs = runs or {}

    def get_latest_versions(self, name, stages):
        return self._versions_by_stage.get(stages[0], [])

    def search_model_versions(self, filter_string):
        return self._versions_by_name

    def get_run(self, run_id):
        return self._runs[run_id]


def test_gate_disabled_always_passes(cfg, monkeypatch):
    cfg["validation_gate"]["enabled"] = False
    passed, msg = gate.check_gate(cfg, stage="Production")
    assert passed is True
    assert "disabled" in msg


def test_gate_passes_when_miou_meets_threshold(cfg, monkeypatch):
    version = FakeVersion("3", "run-1")
    client = FakeClient(
        versions_by_stage={"Production": [version]},
        runs={"run-1": FakeRun({"best_val_miou": 0.60})},
    )
    monkeypatch.setattr(gate, "MlflowClient", lambda: client)

    passed, msg = gate.check_gate(cfg, stage="Production")

    assert passed is True
    assert "v3" in msg


def test_gate_blocks_when_miou_below_threshold(cfg, monkeypatch):
    version = FakeVersion("4", "run-2")
    client = FakeClient(
        versions_by_stage={"Production": [version]},
        runs={"run-2": FakeRun({"best_val_miou": 0.30})},
    )
    monkeypatch.setattr(gate, "MlflowClient", lambda: client)

    passed, msg = gate.check_gate(cfg, stage="Production")

    assert passed is False
    assert "<" in msg


def test_gate_uses_highest_version_for_latest_stage(cfg, monkeypatch):
    v1, v2 = FakeVersion("1", "run-1"), FakeVersion("2", "run-2")
    client = FakeClient(
        versions_by_name=[v1, v2],
        runs={"run-2": FakeRun({"best_val_miou": 0.70})},
    )
    monkeypatch.setattr(gate, "MlflowClient", lambda: client)

    passed, msg = gate.check_gate(cfg, stage="latest")

    assert passed is True
    assert "v2" in msg


def test_gate_raises_when_no_version_found(cfg, monkeypatch):
    client = FakeClient(versions_by_stage={})
    monkeypatch.setattr(gate, "MlflowClient", lambda: client)

    with pytest.raises(LookupError, match="No 'Production' version"):
        gate.check_gate(cfg, stage="Production")


def test_gate_raises_when_metric_missing(cfg, monkeypatch):
    version = FakeVersion("5", "run-5")
    client = FakeClient(
        versions_by_stage={"Production": [version]},
        runs={"run-5": FakeRun({})},
    )
    monkeypatch.setattr(gate, "MlflowClient", lambda: client)

    with pytest.raises(LookupError, match="best_val_miou"):
        gate.check_gate(cfg, stage="Production")
