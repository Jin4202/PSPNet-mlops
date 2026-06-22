import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image


@pytest.fixture
def cfg():
    return {
        "augmentation": {
            "short_size": 24,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "model": {
            "num_classes": 11,
            "zoom_factor": 8,
            "ppm_bins": [1, 2, 3, 6],
            "ppm_dim": 512,
        },
        "mlflow": {
            "tracking_uri": "http://localhost:5000",
            "model_name": "pspnet-camvid",
        },
    }


@pytest.fixture
def sample_image():
    """A small non-square RGB image (wider than tall)."""
    arr = np.random.randint(0, 256, size=(20, 40, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


class FakeSegModel(nn.Module):
    """Stand-in for PSPNet: returns logits at a downsampled resolution."""

    def __init__(self, num_classes=11, downsample=8, return_aux=False):
        super().__init__()
        self.num_classes = num_classes
        self.downsample = downsample
        self.return_aux = return_aux

    def forward(self, x):
        b, _, h, w = x.shape
        out_h, out_w = max(1, h // self.downsample), max(1, w // self.downsample)
        main = torch.zeros(b, self.num_classes, out_h, out_w)
        main[:, 0] = 1.0  # class 0 always wins argmax
        if self.return_aux:
            return main, torch.zeros(b, self.num_classes, out_h, out_w)
        return main


@pytest.fixture
def fake_model():
    return FakeSegModel()
