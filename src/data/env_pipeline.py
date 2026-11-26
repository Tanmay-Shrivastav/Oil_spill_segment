import argparse
import time
import joblib
import numpy as np
import pandas as pd
import requests
import yaml
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def fetch_batch(url, lats, lons, start_date, end_date, hourly_vars, retries=3):
    params = {
        "latitude": ",".join(f"{x:.5f}" for x in lats),
        "longitude": ",".join(f"{x:.5f}" for x in lons),
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": "UTC",
    }
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data if isinstance(data, list) else [data]
            time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return [{} for _ in lats]


def safe_mean(values):
    if not values:
        return np.nan
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    valid = arr[~np.isnan(arr)]
    return float(valid.mean()) if len(valid) > 0 else np.nan


def compute_wind_components(ws_list, wd_list):
    if not ws_list or not wd_list:
        return np.nan, np.nan, np.nan
    ws = np.array([np.nan if v is None else float(v) for v in ws_list], dtype=np.float64)
    wd = np.array([np.nan if v is None else float(v) for v in wd_list], dtype=np.float64)
    mask = ~np.isnan(ws) & ~np.isnan(wd)
    if not np.any(mask):
        return np.nan, np.nan, np.nan
    rad = np.deg2rad(wd[mask])
    u10 = -ws[mask] * np.sin(rad)
    v10 = -ws[mask] * np.cos(rad)
    return float(ws[mask].mean()), float(u10.mean()), float(v10.mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = pd.read_csv(cfg["paths"]["splits_csv"])
    env_cfg = cfg["env"]
    features = env_cfg["features"]

    records = []
    for i in tqdm(range(0, len(df), args.batch_size), desc="Fetching Open-Meteo"):
        batch = df.iloc[i : i + args.batch_size]
        lats, lons = batch["lat"].tolist(), batch["lon"].tolist()
        arch = fetch_batch(
            env_cfg["archive_url"], lats, lons, env_cfg["start_date"],
            env_cfg["end_date"], "wind_speed_10m,wind_direction_10m",
        )
        mar = fetch_batch(
            env_cfg["marine_url"], lats, lons, env_cfg["start_date"],
            env_cfg["end_date"], "wave_height,sea_surface_temperature",
        )
        for (_, row), a, m in zip(batch.iterrows(), arch, mar):
            ah, mh = a.get("hourly", {}), m.get("hourly", {})
            ws, u10, v10 = compute_wind_components(
                ah.get("wind_speed_10m"), ah.get("wind_direction_10m")
            )
            swh = safe_mean(mh.get("wave_height"))
            sst = safe_mean(mh.get("sea_surface_temperature"))
            records.append({
                "scene_id": row["scene_id"], "lat": row["lat"], "lon": row["lon"],
                "region": row["region"], "class": row["class"], "split": row["split"],
                "wind_speed": ws, "u10": u10, "v10": v10, "swh": swh, "sst": sst,
            })
        time.sleep(0.2)

    out_df = pd.DataFrame(records)
    train_mask = out_df["split"] == "train"
    for col in features:
        n_missing = int(out_df[col].isna().sum())
        if n_missing > 0:
            med = float(out_df.loc[train_mask, col].median())
            out_df[col] = out_df[col].fillna(med)
            print(f"[Fallback] Filled {n_missing} missing '{col}' values with train median ({med:.4f})")

    scaler = StandardScaler()
    scaler.fit(out_df.loc[train_mask, features])
    scaled = scaler.transform(out_df[features])
    for idx, col in enumerate(features):
        out_df[f"{col}_scaled"] = scaled[:, idx]

    out_df.to_csv(cfg["paths"]["env_csv"], index=False)
    joblib.dump(scaler, cfg["paths"]["env_scaler"])
    print(f"Saved {len(out_df)} rows to {cfg['paths']['env_csv']} (NaNs: {int(out_df.isna().sum().sum())})")


if __name__ == "__main__":
    main()
