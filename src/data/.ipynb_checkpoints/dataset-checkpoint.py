import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


def parse_list(list_path, data_root="data/camvid"):
    items = []
    with open(list_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                # /SegNet/CamVid/train/xxx.png → data/camvid/train/xxx.png
                img = os.path.join(data_root, parts[0].replace("/SegNet/CamVid/", ""))
                lbl = os.path.join(data_root, parts[1].replace("/SegNet/CamVid/", ""))
                items.append((img, lbl))
    return items


class TrainTransform:
    def __init__(self, cfg):
        a = cfg["augmentation"]
        self.short  = a["short_size"]
        self.crop   = a["crop_size"]
        self.scales = (a["scale_min"], a["scale_max"])
        self.angles = (a["rotate_min"], a["rotate_max"])
        self.mean, self.std = a["mean"], a["std"]
        self.ignore = cfg["data"]["ignore_label"]

    def __call__(self, img, lbl):
        w, h = img.size
        ratio = self.short / min(w, h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.BILINEAR)
        lbl = lbl.resize((int(w*ratio), int(h*ratio)), Image.NEAREST)

        if random.random() < 0.5:
            img, lbl = TF.hflip(img), TF.hflip(lbl)

        s = random.uniform(*self.scales)
        w, h = img.size
        img = img.resize((int(w*s), int(h*s)), Image.BILINEAR)
        lbl = lbl.resize((int(w*s), int(h*s)), Image.NEAREST)

        angle = random.uniform(*self.angles)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        lbl = TF.rotate(lbl, angle, interpolation=TF.InterpolationMode.NEAREST, fill=self.ignore)

        w, h = img.size
        pw, ph = max(self.crop - w, 0), max(self.crop - h, 0)
        if pw or ph:
            img = TF.pad(img, (0, 0, pw, ph), fill=0)
            lbl = TF.pad(lbl, (0, 0, pw, ph), fill=self.ignore)

        w, h = img.size
        x = random.randint(0, w - self.crop)
        y = random.randint(0, h - self.crop)
        img = TF.crop(img, y, x, self.crop, self.crop)
        lbl = TF.crop(lbl, y, x, self.crop, self.crop)

        img = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


class ValTransform:
    def __init__(self, cfg):
        a = cfg["augmentation"]
        self.short  = a["short_size"]
        self.crop   = a["crop_size"]
        self.mean, self.std = a["mean"], a["std"]
        self.ignore = cfg["data"]["ignore_label"]

    def __call__(self, img, lbl):
        w, h = img.size
        ratio = self.short / min(w, h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.BILINEAR)
        lbl = lbl.resize((int(w*ratio), int(h*ratio)), Image.NEAREST)

        w, h = img.size
        pw, ph = max(self.crop - w, 0), max(self.crop - h, 0)
        if pw or ph:
            img = TF.pad(img, (0, 0, pw, ph), fill=0)
            lbl = TF.pad(lbl, (0, 0, pw, ph), fill=self.ignore)

        img = TF.center_crop(img, self.crop)
        lbl = TF.center_crop(lbl, self.crop)

        img = TF.normalize(TF.to_tensor(img), self.mean, self.std)
        lbl = torch.from_numpy(np.array(lbl)).long()
        return img, lbl


class CamVid(Dataset):
    def __init__(self, list_path, transform=None):
        self.items = parse_list(list_path)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, lbl_path = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        lbl = Image.open(lbl_path)
        if self.transform:
            img, lbl = self.transform(img, lbl)
        lbl[lbl == 11] = 255  # void class → ignore
        return img, lbl


def build_loaders(cfg):
    tc = cfg["training"]
    dc = cfg["data"]
    train_ds = CamVid(dc["train_list"], TrainTransform(cfg))
    val_ds   = CamVid(dc["val_list"],   ValTransform(cfg))
    train_loader = DataLoader(train_ds, tc["batch_size"], shuffle=True,
                              num_workers=tc["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, 1, shuffle=False,
                              num_workers=tc["num_workers"], pin_memory=True)
    return train_loader, val_loader
