import os, sys, argparse, yaml, time
import torch
import torch.nn as nn
import mlflow, mlflow.pytorch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.pspnet import build_model
from src.data.dataset import build_loaders
from src.evaluate import evaluate


def poly_lr(base, step, total, power=0.9):
    return base * ((1 - step / total) ** power)


def train(cfg, run_name=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tc = cfg["training"]

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)

    optimizer = torch.optim.SGD(
        model.param_groups(tc["base_lr"]),
        momentum=tc["momentum"],
        weight_decay=tc["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg["data"]["ignore_label"])
    os.makedirs(tc["save_dir"], exist_ok=True)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name or cfg["mlflow"]["run_name"]):
        mlflow.log_params({
            "epochs": tc["epochs"], "batch_size": tc["batch_size"],
            "base_lr": tc["base_lr"], "aux_weight": tc["aux_weight"],
            "arch": cfg["model"]["arch"],
        })

        best_miou, step, max_iter = 0.0, 0, tc["epochs"] * len(train_loader)

        for epoch in range(tc["epochs"]):
            model.train()
            epoch_loss = 0.0

            for imgs, lbls in train_loader:
                imgs, lbls = imgs.to(device), lbls.to(device).long()
                lr = poly_lr(tc["base_lr"], step, max_iter, tc["poly_power"])
                for i, g in enumerate(optimizer.param_groups):
                    g["lr"] = lr if i < 4 else lr * 10
                step += 1

                main, aux = model(imgs)
                loss = criterion(main, lbls) + tc["aux_weight"] * criterion(aux, lbls)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            val      = evaluate(model, val_loader, cfg, device)

            mlflow.log_metrics({
                "train_loss": avg_loss,
                "val_miou":   val["miou"],
                "val_allacc": val["allacc"],
                "lr":         lr,
            }, step=epoch)

            print(f"Epoch {epoch+1}/{tc['epochs']} | loss={avg_loss:.4f} | miou={val['miou']:.4f}")

            if val["miou"] > best_miou:
                best_miou = val["miou"]
                ckpt = os.path.join(tc["save_dir"], "best.pth")
                torch.save({"epoch": epoch+1, "state_dict": model.state_dict(),
                            "miou": best_miou}, ckpt)

        threshold = cfg["evaluation"]["miou_threshold"]
        if best_miou >= threshold:
            ckpt_data = torch.load(os.path.join(tc["save_dir"], "best.pth"), map_location=device)
            model.load_state_dict(ckpt_data["state_dict"])
            mlflow.pytorch.log_model(model, "model",
                                     registered_model_name=cfg["mlflow"]["model_name"])
            print(f"Registered | best miou={best_miou:.4f}")
        else:
            print(f"Skipped registration | miou={best_miou:.4f} < threshold={threshold}")

    return best_miou


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="configs/config.yaml")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    best = train(cfg, args.run_name)
    print(f"Done | best miou={best:.4f}")
