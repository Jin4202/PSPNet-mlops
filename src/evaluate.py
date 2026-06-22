import numpy as np
import torch


def iou_cpu(pred, target, num_classes, ignore=255):
    mask   = target != ignore
    pred   = pred[mask].clip(0, num_classes - 1)
    target = target[mask].clip(0, num_classes - 1)
    inter  = pred[pred == target]
    bins   = np.arange(num_classes + 1)
    area_i, _ = np.histogram(inter,   bins)
    area_p, _ = np.histogram(pred,    bins)
    area_t, _ = np.histogram(target,  bins)
    return area_i, area_p + area_t - area_i, area_t


def evaluate(model, loader, cfg, device):
    nc     = cfg["data"]["num_classes"]
    ignore = cfg["data"]["ignore_label"]
    model.eval()

    total_i = np.zeros(nc)
    total_u = np.zeros(nc)
    total_t = np.zeros(nc)

    with torch.no_grad():
        for imgs, lbls in loader:
            pred = model(imgs.to(device)).argmax(1)
            p = pred.cpu().numpy().squeeze()
            t = lbls.numpy().squeeze()
            i, u, gt = iou_cpu(p, t, nc, ignore)
            total_i += i
            total_u += u
            total_t += gt

    valid = total_u > 0
    iou   = np.where(valid, total_i / np.where(valid, total_u, 1), 0)
    miou  = iou[valid].mean()
    allacc = total_i.sum() / (total_t.sum() + 1e-10)

    return {"miou": float(miou), "allacc": float(allacc), "class_iou": iou}
