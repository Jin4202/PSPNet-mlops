import contextlib
import socket

import pytest
import torch
import torch.nn as nn

from app import model_loader


def _free_port_that_refuses_connections():
    """Bind and immediately close a socket to get a port nothing is listening on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@contextlib.contextmanager
def _listening_socket():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    try:
        yield s.getsockname()[1]
    finally:
        s.close()


def test_is_reachable_true_for_open_port():
    with _listening_socket() as port:
        assert model_loader._is_reachable(f"http://127.0.0.1:{port}") is True


def test_is_reachable_false_for_closed_port():
    port = _free_port_that_refuses_connections()
    assert model_loader._is_reachable(f"http://127.0.0.1:{port}") is False


def test_is_reachable_false_for_non_http_scheme():
    assert model_loader._is_reachable("file:///tmp/whatever") is False


def test_is_reachable_false_for_missing_hostname():
    assert model_loader._is_reachable("http://") is False


def test_load_model_falls_back_to_checkpoint_when_mlflow_unreachable(monkeypatch, tmp_path, cfg):
    monkeypatch.setattr(model_loader, "_is_reachable", lambda uri, timeout=2.0: False)

    fake_model = nn.Conv2d(3, cfg["model"]["num_classes"], 1)
    monkeypatch.setattr("src.models.pspnet.build_model", lambda cfg: fake_model)

    ckpt_path = tmp_path / "best.pth"
    torch.save({"state_dict": fake_model.state_dict()}, ckpt_path)
    monkeypatch.setenv("CHECKPOINT_PATH", str(ckpt_path))

    model, source = model_loader.load_model(cfg, torch.device("cpu"))

    assert source == f"checkpoint:{ckpt_path}"
    assert not model.training


def test_load_model_raises_when_mlflow_unreachable_and_no_checkpoint(monkeypatch, tmp_path, cfg):
    monkeypatch.setattr(model_loader, "_is_reachable", lambda uri, timeout=2.0: False)
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "does-not-exist.pth"))

    with pytest.raises(RuntimeError):
        model_loader.load_model(cfg, torch.device("cpu"))
