import argparse
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import OilSpillDataset
from src.train import build_model
from src.utils.metrics import compute_confusion, metrics_from_counts


def save_overlay(sar, mask, prob, out_path, title):
    vv = sar[0].cpu().numpy()
    gt = mask[0].cpu().numpy()
    pr = (prob[0, 0].cpu().numpy() >= 0.5).astype(np.float32)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    axes[0].imshow(vv, cmap="gray"); axes[0].set_title("SAR VV")
    axes[1].imshow(vv, cmap="gray"); axes[1].imshow(gt, cmap="Reds", alpha=0.45 * (gt > 0)); axes[1].set_title("Ground Truth")
    axes[2].imshow(vv, cmap="gray"); axes[2].imshow(pr, cmap="cool", alpha=0.45 * (pr > 0)); axes[2].set_title(title)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate_model(name, cfg, loader, ckpt_path, vis_dir, device, limit=None, num_vis=6):
    model = build_model(name, cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    use_amp = device.type == "cuda"

    tp = fp = fn = tn = 0
    l_fp = l_tn = 0
    total_ms, n_patches, saved = 0.0, 0, 0

    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        sar, env, mask = batch["sar"].to(device), batch["env"].to(device), batch["mask"].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(sar, env)
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_ms += (time.perf_counter() - t0) * 1000.0
        n_patches += sar.shape[0]

        probs = torch.sigmoid(logits)
        for j in range(sar.shape[0]):
            btp, bfp, bfn, btn = compute_confusion(logits[j : j + 1], mask[j : j + 1])
            tp += btp; fp += bfp; fn += bfn; tn += btn
            if batch["class"][j] == "Lookalike":
                l_fp += bfp; l_tn += btn
            if saved < num_vis and (mask[j].sum() > 0 or batch["class"][j] == "Lookalike"):
                save_overlay(sar[j], mask[j], probs[j : j + 1], vis_dir / f"{name}_{batch['patch_id'][j]}.png", f"{name}")
                saved += 1

    res = metrics_from_counts(tp, fp, fn, tn)
    res["Lookalike_FPR"] = round(l_fp / (l_fp + l_tn + 1e-7), 6)
    res["Inference_ms"] = round(total_ms / max(n_patches, 1), 2)
    res["Model"] = name
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--model", default=None)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--limit_batches", type=int, default=None)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bs = args.batch_size or (2 if device.type == "cpu" else cfg["train"]["batch_size"])

    patches_csv = str(Path(args.data_dir) / "processed/patches.csv") if args.data_dir else cfg["paths"]["patches_csv"]
    env_csv = str(Path(args.data_dir) / "env_data.csv") if args.data_dir else cfg["paths"]["env_csv"]
    ckpt_dir = Path(args.out_dir) / "checkpoints" if args.out_dir else Path(cfg["paths"]["checkpoints_dir"])
    log_dir = Path(args.out_dir) / "logs" if args.out_dir else Path(cfg["paths"]["logs_dir"])
    vis_dir = Path(args.out_dir) / "visuals" if args.out_dir else Path(cfg["paths"]["visuals_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    test_ds = OilSpillDataset(patches_csv, env_csv, "test", augment=False)
    nw = 0 if args.limit_batches else cfg["train"]["num_workers"]
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    models = [args.model] if args.model else cfg["models"]
    rows = []
    for name in models:
        ckpt = ckpt_dir / f"{name}_best.pt"
        if not ckpt.exists():
            print(f"[Skip] Checkpoint not found: {ckpt}")
            continue
        res = evaluate_model(name, cfg, test_loader, ckpt, vis_dir, device, args.limit_batches)
        rows.append(res)
        print(res)

    if rows:
        cols = ["Model", "IoU", "Dice", "Precision", "Recall", "F1", "FPR", "Lookalike_FPR", "Inference_ms"]
        out_csv = log_dir / "comparison_results.csv"
        pd.DataFrame(rows)[cols].to_csv(out_csv, index=False)
        print(f"Saved results to {out_csv}")


if __name__ == "__main__":
    main()
