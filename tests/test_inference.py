import base64
import io

import numpy as np
import torch
from PIL import Image

from app.inference import CAMVID_PALETTE, mask_to_png_base64, predict_mask, preprocess
from tests.conftest import FakeSegModel


def test_preprocess_shape_and_dtype(sample_image, cfg):
    tensor = preprocess(sample_image, cfg)

    assert tensor.dtype == torch.float32
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3
    # short side (height, since image is wider than tall) must match short_size
    assert tensor.shape[2] == cfg["augmentation"]["short_size"]
    # aspect ratio preserved
    w, h = sample_image.size
    expected_w = round(w * cfg["augmentation"]["short_size"] / h)
    assert tensor.shape[3] == expected_w


def test_preprocess_normalization(cfg):
    mean, std = cfg["augmentation"]["mean"], cfg["augmentation"]["std"]
    short = cfg["augmentation"]["short_size"]
    image = Image.new("RGB", (short, short), color=(128, 128, 128))

    tensor = preprocess(image, cfg)

    expected = (128 / 255.0 - mean[0]) / std[0]
    assert torch.allclose(tensor[0, 0], torch.full((short, short), expected), atol=1e-5)


def test_predict_mask_matches_original_size(sample_image, cfg):
    model = FakeSegModel(num_classes=cfg["model"]["num_classes"])
    device = torch.device("cpu")

    mask = predict_mask(model, sample_image, cfg, device)

    assert mask.shape == (sample_image.height, sample_image.width)
    assert mask.dtype == np.uint8


def test_predict_mask_uses_main_output_when_tuple(sample_image, cfg):
    model = FakeSegModel(num_classes=cfg["model"]["num_classes"], return_aux=True)
    device = torch.device("cpu")

    mask = predict_mask(model, sample_image, cfg, device)

    assert mask.shape == (sample_image.height, sample_image.width)
    # FakeSegModel always biases class 0 in its main output
    assert (mask == 0).all()


def test_mask_to_png_base64_roundtrips(cfg):
    num_classes = cfg["model"]["num_classes"]
    mask = np.tile(np.arange(num_classes, dtype=np.uint8), (4, 1))

    encoded = mask_to_png_base64(mask)
    decoded_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded_bytes))

    assert image.format == "PNG"
    assert image.size == (mask.shape[1], mask.shape[0])
    decoded_arr = np.asarray(image.convert("RGB"))
    expected = CAMVID_PALETTE[mask]
    assert np.array_equal(decoded_arr, expected)


def test_mask_to_png_base64_clips_out_of_range_classes():
    mask = np.array([[0, 99]], dtype=np.uint8)

    encoded = mask_to_png_base64(mask)
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB"))

    assert np.array_equal(decoded[0, 1], CAMVID_PALETTE[-1])
