import torch

from src.models.pspnet import PSPNet, build_model
from src.models.resnet import resnet50

NUM_CLASSES = 3
IMG_SIZE = 32  # small so the real ResNet-50 backbone runs fast on CPU


def _tiny_model():
    return PSPNet(num_classes=NUM_CLASSES, ppm_bins=(1, 2), ppm_dim=32)


def test_resnet50_output_channels():
    model = resnet50()
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    out = model(x)
    assert out.shape[1] == 2048  # layer4 output channels


def test_pspnet_train_mode_returns_main_and_aux_matching_input_size():
    model = _tiny_model()
    model.train()
    x = torch.rand(2, 3, IMG_SIZE, IMG_SIZE)

    main, aux = model(x)

    assert main.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)
    assert aux.shape == (2, NUM_CLASSES, IMG_SIZE, IMG_SIZE)


def test_pspnet_eval_mode_returns_only_main():
    model = _tiny_model()
    model.eval()
    x = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        out = model(x)

    assert torch.is_tensor(out)
    assert out.shape == (1, NUM_CLASSES, IMG_SIZE, IMG_SIZE)


def test_pspnet_param_groups_has_five_groups_with_head_at_10x_lr():
    model = _tiny_model()
    base_lr = 0.01

    groups = model.param_groups(base_lr)

    assert len(groups) == 5
    assert [g["lr"] for g in groups] == [base_lr] * 4 + [base_lr * 10]
    # every parameter should appear in exactly one group
    grouped_params = [id(p) for g in groups for p in g["params"]]
    assert len(grouped_params) == len(set(grouped_params))
    assert len(grouped_params) == len(list(model.parameters()))


def test_build_model_reads_config_keys():
    cfg = {
        "model": {
            "num_classes": NUM_CLASSES,
            "zoom_factor": 8,
            "ppm_bins": [1, 2],
            "ppm_dim": 32,
        }
    }
    model = build_model(cfg)
    assert isinstance(model, PSPNet)
    assert model.main_cls[-1].out_channels == NUM_CLASSES
