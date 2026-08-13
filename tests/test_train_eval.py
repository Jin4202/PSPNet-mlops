import itertools

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluate import evaluate, iou_cpu
from src.train import poly_lr, train_one_epoch


def test_poly_lr_at_step_zero_equals_base():
    assert poly_lr(base=1.0, step=0, total=100, power=1.0) == 1.0


def test_poly_lr_at_final_step_approaches_zero():
    assert poly_lr(base=1.0, step=100, total=100, power=1.0) == 0.0


def test_poly_lr_is_monotonically_decreasing():
    total = 100
    lrs = [poly_lr(base=1.0, step=s, total=total, power=0.9) for s in range(total)]
    assert all(a >= b for a, b in itertools.pairwise(lrs))


def test_iou_cpu_perfect_prediction():
    pred = np.array([0, 0, 1, 1])
    target = np.array([0, 0, 1, 1])
    inter, union, gt = iou_cpu(pred, target, num_classes=2)
    assert list(inter) == [2, 2]
    assert list(union) == [2, 2]
    assert list(gt) == [2, 2]


def test_iou_cpu_ignores_ignore_label():
    pred = np.array([0, 1, 0])
    target = np.array([0, 255, 1])  # middle pixel ignored
    inter, _union, gt = iou_cpu(pred, target, num_classes=2, ignore=255)
    # only pixel 0 (class 0, correct) and pixel 2 (pred 0, target 1) remain
    assert inter.sum() == 1  # only the class-0 pixel matches
    assert gt.sum() == 2  # 2 valid ground-truth pixels remain


class _ConstantModel(nn.Module):
    """Always predicts class 0 everywhere."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape
        logits = torch.zeros(b, self.num_classes, h, w)
        logits[:, 0] = 1.0
        return logits


def test_evaluate_computes_expected_miou_for_constant_predictor():
    model = _ConstantModel(num_classes=2)
    model.eval()
    imgs = torch.rand(1, 3, 2, 2)
    lbls = torch.zeros(1, 2, 2, dtype=torch.long)  # all class 0 -> perfect match
    loader = [(imgs, lbls)]
    cfg = {"data": {"num_classes": 2, "ignore_label": 255}}

    result = evaluate(model, loader, cfg, device="cpu")

    assert result["miou"] == 1.0
    assert result["allacc"] == pytest.approx(1.0)


class _TrainableModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(3, num_classes, 1)

    def forward(self, x):
        main = self.conv(x)
        if self.training:
            return main, main.clone()
        return main


def test_train_one_epoch_runs_and_advances_step_count():
    num_classes = 2
    model = _TrainableModel(num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    cfg = {"training": {"base_lr": 0.01, "poly_power": 0.9, "aux_weight": 0.4}}
    loader = [
        (torch.rand(2, 3, 4, 4), torch.randint(0, num_classes, (2, 4, 4))),
        (torch.rand(2, 3, 4, 4), torch.randint(0, num_classes, (2, 4, 4))),
    ]

    avg_loss, current_step, lr = train_one_epoch(
        model, loader, optimizer, criterion, "cpu", cfg, current_step=0, total_steps=10
    )

    assert isinstance(avg_loss, float)
    assert current_step == 2  # one step per batch
    assert 0.0 <= lr <= cfg["training"]["base_lr"]
