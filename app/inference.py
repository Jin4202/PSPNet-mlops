import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# CamVid-11 class palette (SegNet-Tutorial convention), order matches
# the README class list: Sky, Building, Pole, Road, Pavement, Tree,
# SignSymbol, Fence, Car, Pedestrian, Bicyclist.
CAMVID_PALETTE = np.array(
    [
        [128, 128, 128],  # Sky
        [128, 0, 0],      # Building
        [192, 192, 128],  # Pole
        [128, 64, 128],   # Road
        [60, 40, 222],    # Pavement
        [128, 128, 0],    # Tree
        [192, 128, 128],  # SignSymbol
        [64, 64, 128],    # Fence
        [64, 0, 128],      # Car
        [64, 64, 0],      # Pedestrian
        [0, 128, 192],    # Bicyclist
    ],
    dtype=np.uint8,
)


def preprocess(image: Image.Image, cfg: dict) -> torch.Tensor:
    """
    Resize so the short side matches the training short_size, then
    normalize with the same mean/std used in ValTransform. No crop —
    inference runs on the full (resized) image.
    """
    a = cfg["augmentation"]
    short = a["short_size"]
    mean, std = a["mean"], a["std"]

    w, h = image.size
    ratio = short / min(w, h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    image = image.resize((new_w, new_h), Image.BILINEAR)

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor


@torch.no_grad()
def predict_mask(model, image: Image.Image, cfg: dict, device: torch.device) -> np.ndarray:
    """Run the model and return a (H, W) array of class indices at the original image size."""
    orig_w, orig_h = image.size
    tensor = preprocess(image, cfg).to(device)

    logits = model(tensor)
    if isinstance(logits, tuple):  # train-mode style (main, aux) output
        logits = logits[0]

    logits = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=True)
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)


def mask_to_png_base64(mask: np.ndarray) -> str:
    """Color-code a class-index mask using the CamVid palette and encode it as a base64 PNG."""
    color = CAMVID_PALETTE[np.clip(mask, 0, len(CAMVID_PALETTE) - 1)]
    buf = io.BytesIO()
    Image.fromarray(color, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
