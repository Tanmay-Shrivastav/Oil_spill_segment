import argparse
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import OilSpillDataset
from src.models.baselines import build_baseline
from src.models.concat_unet import ConcatUNet
from src.models.proposed_model import FiLMUNet
from src.utils.losses import CombinedLoss
from src.utils.metrics import compute_confusion, metrics_from_counts


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name, cfg):
    if name in ("unet", "deeplabv3plus", "attn_unet"):
        return build_baseline(name)
    if name == "concat_unet":
        return ConcatUNet()
    if name == "proposed_model":
        return FiLMUNet(film_dropout=cfg["train"]["film_dropout"])
    raise ValueError(f"Unknown model: {name}")


def run_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, train=True, limit=None, desc=""):
    model.train(train)
    use_amp = device.type == "cuda"
    total_loss, steps = 0.0, 0
    tp = fp = fn = tn = 0
    pbar = tqdm(loader, total=limit or len(loader), desc=desc, leave=False, dynamic_ncols=True)
    for i, batch in enumerate(pbar):
        if limit and i >= limit:
            break
        sar, env, mask = batch["sar"].to(device), batch["env"].to(device), batch["mask"].to(device)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(sar, env)
                loss = criterion(logits, mask)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
        total_loss += float(loss.item())
        steps += 1
        btp, bfp, bfn, btn = compute_confusion(logits.detach(), mask)
        tp += btp; fp += bfp; fn += bfn; tn += btn
    m = metrics_from_counts(tp, fp, fn, tn)
    return total_loss / max(steps, 1), m["IoU"], m["Dice"]


def run_training(model_name, data_dir=None, out_dir=None, epochs=None, batch_size=None, limit_batches=None, config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    t_cfg = cfg["train"]
    epochs = epochs or t_cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = batch_size or (2 if device.type == "cpu" else t_cfg["batch_size"])

    patches_csv = str(Path(data_dir) / "processed/patches.csv") if data_dir else cfg["paths"]["patches_csv"]
    env_csv = str(Path(data_dir) / "env_data.csv") if data_dir else cfg["paths"]["env_csv"]
    ckpt_dir = Path(out_dir) / "checkpoints" if out_dir else Path(cfg["paths"]["checkpoints_dir"])
    log_dir = Path(out_dir) / "logs" if out_dir else Path(cfg["paths"]["logs_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_ds = OilSpillDataset(patches_csv, env_csv, "train", augment=True)
    val_ds = OilSpillDataset(patches_csv, env_csv, "val", augment=False)
    nw = 0 if limit_batches else t_cfg["num_workers"]
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    model = build_model(model_name, cfg).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = CombinedLoss(t_cfg["bce_weight"], t_cfg["dice_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=t_cfg["max_lr"], weight_decay=t_cfg["weight_decay"])
    steps_per_epoch = limit_batches or len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=t_cfg["max_lr"], epochs=epochs, steps_per_epoch=steps_per_epoch)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_dice, no_imp, history = -1.0, 0, []
    for epoch in tqdm(range(1, epochs + 1), desc=model_name, leave=False, dynamic_ncols=True):
        tr_loss, _, _ = run_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device, True, limit_batches, f"{model_name} E{epoch} Train")
        va_loss, va_iou, va_dice = run_epoch(model, val_loader, criterion, optimizer, scheduler, scaler, device, False, limit_batches, f"{model_name} E{epoch} Val")
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_iou": va_iou, "val_dice": va_dice})
        tqdm.write(f"[{model_name}] Epoch {epoch}/{epochs} | lr={lr:.2e} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} va_iou={va_iou:.4f} va_dice={va_dice:.4f}")
        if va_dice > best_dice:
            best_dice, no_imp = va_dice, 0
            raw = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(raw.state_dict(), ckpt_dir / f"{model_name}_best.pt")
        else:
            no_imp += 1
            if no_imp >= t_cfg["patience"]:
                tqdm.write(f"[{model_name}] Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(log_dir / f"{model_name}_history.csv", index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--limit_batches", type=int, default=None)
    args = p.parse_args()
    run_training(args.model, args.data_dir, args.out_dir, args.epochs, args.batch_size, args.limit_batches, args.config)


if __name__ == "__main__":
    main()
