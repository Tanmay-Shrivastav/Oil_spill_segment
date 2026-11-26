import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import yaml
from tqdm import tqdm


def read_clean_sar(path, clip_min, clip_max):
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=clip_max, neginf=clip_min)
    return np.clip(arr, clip_min, clip_max)


def compute_train_stats(train_df, clip_min, clip_max):
    ch_sum = np.zeros(2, dtype=np.float64)
    ch_sq = np.zeros(2, dtype=np.float64)
    count = 0
    for path in tqdm(train_df["image_path"], desc="Computing train SAR stats"):
        arr = read_clean_sar(path, clip_min, clip_max).astype(np.float64)
        ch_sum += arr.sum(axis=(1, 2))
        ch_sq += (arr ** 2).sum(axis=(1, 2))
        count += arr.shape[1] * arr.shape[2]
    mean = ch_sum / count
    std = np.sqrt(np.maximum(ch_sq / count - mean ** 2, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    p_cfg = cfg["preprocess"]
    ps = p_cfg["patch_size"]
    clip_min, clip_max = p_cfg["sar_clip_min"], p_cfg["sar_clip_max"]

    out_dir = Path(args.out_dir or cfg["paths"]["processed_dir"])
    sar_dir = out_dir / "sar"
    mask_dir = out_dir / "masks"
    sar_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(cfg["paths"]["splits_csv"])
    mean, std = compute_train_stats(df[df["split"] == "train"], clip_min, clip_max)
    stats_path = out_dir / "sar_stats.json"
    with open(stats_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f, indent=2)
    print(f"Train SAR mean={mean.tolist()}, std={std.tolist()}")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tiling 2048x2048 scenes"):
        sar = read_clean_sar(row["image_path"], clip_min, clip_max)
        sar = (sar - mean[:, None, None]) / (std[:, None, None] + 1e-6)
        with rasterio.open(row["mask_path"]) as msrc:
            mask = (msrc.read(1) > 0).astype(np.uint8)

        h, w = mask.shape
        idx = 0
        for r in range(0, h, ps):
            for c in range(0, w, ps):
                patch_id = f"{row['scene_id']}_p{idx:02d}"
                s_path = sar_dir / f"{patch_id}.npy"
                m_path = mask_dir / f"{patch_id}.npy"
                sar_patch = sar[:, r : r + ps, c : c + ps].astype(np.float16)
                mask_patch = mask[r : r + ps, c : c + ps]
                np.save(s_path, sar_patch)
                np.save(m_path, mask_patch)
                rows.append({
                    "patch_id": patch_id,
                    "scene_id": row["scene_id"],
                    "split": row["split"],
                    "class": row["class"],
                    "region": row["region"],
                    "sar_path": s_path.as_posix(),
                    "mask_path": m_path.as_posix(),
                    "oil_pixels": int(mask_patch.sum()),
                })
                idx += 1

    patches_df = pd.DataFrame(rows)
    patches_csv = out_dir / "patches.csv"
    patches_df.to_csv(patches_csv, index=False)
    print(f"Saved {len(patches_df)} patches to {patches_csv}")


if __name__ == "__main__":
    main()
