import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet50


class PPM(nn.Module):
    def __init__(self, in_dim, out_dim, bins=(1, 2, 3, 6)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(b),
                nn.Conv2d(in_dim, out_dim, 1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True),
            ) for b in bins
        ])

    def forward(self, x):
        h, w = x.shape[2:]
        parts = [x] + [
            F.interpolate(branch(x), size=(h, w), mode='bilinear', align_corners=True)
            for branch in self.branches
        ]
        return torch.cat(parts, dim=1)


def _cls_head(in_ch, num_classes):
    mid = in_ch // 4
    return nn.Sequential(
        nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid),
        nn.ReLU(inplace=True),
        nn.Dropout2d(0.1),
        nn.Conv2d(mid, num_classes, 1),
    )


class PSPNet(nn.Module):
    def __init__(self, num_classes, zoom_factor=8, ppm_bins=(1,2,3,6), ppm_dim=512):
        super().__init__()
        self.zoom_factor = zoom_factor
        bb = resnet50()
        self.layer0, self.layer1, self.layer2 = bb.layer0, bb.layer1, bb.layer2
        self.layer3, self.layer4              = bb.layer3, bb.layer4
        self.ppm      = PPM(2048, ppm_dim, ppm_bins)
        self.main_cls = _cls_head(2048 + len(ppm_bins) * ppm_dim, num_classes)
        self.aux_cls  = _cls_head(1024, num_classes)
        self._init_heads()

    def _init_heads(self):
        for m in list(self.ppm.modules()) + list(self.main_cls.modules()) + list(self.aux_cls.modules()):
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x);  x3 = x
        x = self.layer4(x)
        x = self.ppm(x)
        main = F.interpolate(self.main_cls(x), size=(H, W), mode='bilinear', align_corners=True)
        if self.training:
            aux = F.interpolate(self.aux_cls(x3), size=(H, W), mode='bilinear', align_corners=True)
            return main, aux
        return main

    def param_groups(self, base_lr):
        return [
            {"params": list(self.layer0.parameters()) + list(self.layer1.parameters()), "lr": base_lr},
            {"params": list(self.layer2.parameters()), "lr": base_lr},
            {"params": list(self.layer3.parameters()), "lr": base_lr},
            {"params": list(self.layer4.parameters()), "lr": base_lr},
            {"params": list(self.ppm.parameters()) +
                       list(self.main_cls.parameters()) +
                       list(self.aux_cls.parameters()), "lr": base_lr * 10},
        ]


def build_model(cfg):
    m = cfg["model"]
    return PSPNet(m["num_classes"], m["zoom_factor"], m["ppm_bins"], m["ppm_dim"])
