"""
Demonstrate why drift detection matters: corrupt a batch of real CamVid test
images (darken + blur, simulating low-light/foggy driving conditions), then
measure two things side by side:

  1. mIoU on the original images vs. the corrupted ones (src.evaluate.evaluate)
  2. Whether src.drift.check_drift_against_reference flags the corrupted
     batch as drifted from the training distribution

Both are measured, not assumed — this is a report script, not a gate, so it
always exits 0. Corrupted images and an HTML drift report are left under
--output-dir for inspection afterward.
"""
import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.model_loader import load_config, load_model
from src.data.dataset import ValTransform, parse_list
from src.drift import check_drift_against_reference
from src.evaluate import evaluate


class _PairDataset(Dataset):
    """(image_path, label_path) pairs run through ValTransform, matching CamVid's
    __getitem__ (including the void-class remap) but without parse_list's
    SegNet-specific path rewriting, which doesn't apply to freshly written files."""

    def __init__(self, pairs: list[tuple[str, str]], transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, lbl_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path)
        img, lbl = self.transform(img, lbl)
        lbl[lbl == 11] = 255  # void class → ignore
        return img, lbl


def corrupt_image(img: Image.Image) -> Image.Image:
    """Darken + blur, simulating low-light/foggy driving conditions."""
    img = ImageEnhance.Brightness(img).enhance(0.4)
    return img.filter(ImageFilter.GaussianBlur(radius=3))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--num-images", type=int, default=30)
    parser.add_argument("--output-dir", default="results/drift_demo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, source = load_model(cfg, device)
    print(f"Model: {source}  device={device}")

    data_cfg = cfg["data"]
    pairs = parse_list(data_cfg["test_list"], data_root=data_cfg["dataset_path"].rstrip("/"))[: args.num_images]
    print(f"Using {len(pairs)} images from {data_cfg['test_list']}")

    output_dir = Path(args.output_dir)
    corrupted_dir = output_dir / "corrupted"
    corrupted_dir.mkdir(parents=True, exist_ok=True)

    corrupted_pairs = []
    for img_path, lbl_path in pairs:
        corrupted = corrupt_image(Image.open(img_path).convert("RGB"))
        out_path = corrupted_dir / Path(img_path).name
        corrupted.save(out_path)
        corrupted_pairs.append((str(out_path), lbl_path))

    transform = ValTransform(cfg)
    original_loader = DataLoader(_PairDataset(pairs, transform), batch_size=1)
    corrupted_loader = DataLoader(_PairDataset(corrupted_pairs, transform), batch_size=1)

    print("Evaluating on original images ...")
    original_result = evaluate(model, original_loader, cfg, device)
    print("Evaluating on corrupted images ...")
    corrupted_result = evaluate(model, corrupted_loader, cfg, device)

    delta = corrupted_result["miou"] - original_result["miou"]
    print()
    print(f"mIoU (original)  : {original_result['miou']:.4f}")
    print(f"mIoU (corrupted) : {corrupted_result['miou']:.4f}")
    print(f"Delta            : {delta:+.4f}")

    print()
    print("Checking drift on the corrupted batch ...")
    drift_result = check_drift_against_reference(cfg, str(corrupted_dir))
    report_path = output_dir / f"drift_report_{datetime.now(UTC):%Y%m%d_%H%M%S}.html"
    drift_result.report.save_html(str(report_path))

    threshold = cfg["drift_detection"]["drift_share_threshold"]
    print(
        f"[drift-check] {'PASSED' if drift_result.passed else 'BLOCKED'}: "
        f"drift_share={drift_result.drift_share:.2f} "
        f"({drift_result.drifted_columns}/{drift_result.total_columns} columns drifted) "
        f"{'<=' if drift_result.passed else '>'} threshold={threshold:.2f}"
    )
    print(f"Corrupted images: {corrupted_dir}")
    print(f"Drift report:     {report_path}")


if __name__ == "__main__":
    main()
