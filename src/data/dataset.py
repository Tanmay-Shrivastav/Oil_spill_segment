import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


ENV_COLS = ["wind_speed_scaled", "u10_scaled", "v10_scaled", "swh_scaled", "sst_scaled"]


class OilSpillDataset(Dataset):
    def __init__(self, patches_csv, env_csv, split, augment=False):
        patches = pd.read_csv(patches_csv)
        patches = patches[patches["split"] == split].reset_index(drop=True)
        env = pd.read_csv(env_csv)[["scene_id"] + ENV_COLS]
        self.df = patches.merge(env, on="scene_id", how="left")
        self.augment = augment and (split == "train")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sar = np.load(row["sar_path"]).astype(np.float32)
        mask = np.load(row["mask_path"]).astype(np.float32)
        env = row[ENV_COLS].to_numpy(dtype=np.float32)

        if self.augment:
            if np.random.rand() < 0.5:
                sar = np.flip(sar, axis=2).copy()
                mask = np.flip(mask, axis=1).copy()
            if np.random.rand() < 0.5:
                sar = np.flip(sar, axis=1).copy()
                mask = np.flip(mask, axis=0).copy()
            k = np.random.randint(0, 4)
            if k:
                sar = np.rot90(sar, k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k, axes=(0, 1)).copy()

        return {
            "sar": torch.from_numpy(sar),
            "env": torch.from_numpy(env),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "class": row["class"],
            "patch_id": row["patch_id"],
        }
