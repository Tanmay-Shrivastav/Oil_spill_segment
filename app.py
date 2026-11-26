"""
Oil-Spill Segmentation — Earth Observation Console
===================================================
Three-tab Streamlit dashboard:
  1. Global Coverage & Geographic Split
  2. Live Inference & FiLM Stress-Test Lab
  3. Cross-Geographical Benchmarks
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import yaml

# ── project imports ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import ENV_COLS
from src.models.proposed_model import FiLMUNet
from src.models.baselines import build_baseline
from src.models.concat_unet import ConcatUNet
from src.utils.metrics import compute_confusion, metrics_from_counts

# ── configuration ────────────────────────────────────────────────────────────
CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = ROOT / CFG["paths"]["checkpoints_dir"]
VIS_DIR = ROOT / CFG["paths"]["visuals_dir"]

REGION_COLORS = {
    "Europe_Med_BlackSea": "#3b82f6",
    "MiddleEast_RedSea_PersianGulf": "#f59e0b",
    "Americas": "#10b981",
    "AsiaPacific_Other": "#ef4444",
}

ENV_LABELS = {
    "wind_speed": ("Wind Speed", "m/s", 0.0, 40.0, 15.0),
    "u10": ("U-Wind (10 m)", "m/s", -15.0, 15.0, 0.0),
    "v10": ("V-Wind (10 m)", "m/s", -15.0, 15.0, 0.0),
    "swh": ("Sig. Wave Height", "m", 0.0, 8.0, 1.0),
    "sst": ("Sea Surface Temp", "°C", -2.0, 35.0, 20.0),
}


# ── data loading (cached) ───────────────────────────────────────────────────
@st.cache_data
def load_splits():
    return pd.read_csv(ROOT / CFG["paths"]["splits_csv"])


@st.cache_data
def load_env():
    return pd.read_csv(ROOT / CFG["paths"]["env_csv"])


@st.cache_data
def load_patches():
    return pd.read_csv(ROOT / CFG["paths"]["patches_csv"])


@st.cache_data
def load_history():
    path = ROOT / CFG["paths"]["logs_dir"] / "proposed_model_history.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_results():
    path = ROOT / CFG["paths"]["logs_dir"] / "comparison_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_env_scaler():
    import joblib
    path = ROOT / CFG["paths"]["env_scaler"]
    return joblib.load(path) if path.exists() else None


# ── model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_model(name: str):
    """Load a model checkpoint; returns None if not found."""
    ckpt = CKPT_DIR / f"{name}_best.pt"
    if not ckpt.exists():
        return None
    if name in ("unet", "deeplabv3plus", "attn_unet"):
        model = build_baseline(name)
    elif name == "concat_unet":
        model = ConcatUNet()
    elif name == "proposed_model":
        model = FiLMUNet(film_dropout=CFG["train"]["film_dropout"])
    else:
        return None
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    model.to(DEVICE).eval()
    return model


# ── inference helpers ────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, sar_np: np.ndarray, env_vec: np.ndarray | None = None):
    """Run single-patch inference. Returns probability map (H, W)."""
    sar_t = torch.from_numpy(sar_np).unsqueeze(0).float().to(DEVICE)
    env_t = torch.from_numpy(env_vec).unsqueeze(0).float().to(DEVICE) if env_vec is not None else None
    logits = model(sar_t, env_t)
    return torch.sigmoid(logits).squeeze().cpu().numpy()


def scale_env_raw(raw_dict: dict, scaler) -> np.ndarray:
    """Scale raw env values using the fitted scaler, returning scaled vector."""
    ordered = [raw_dict[k] for k in ["wind_speed", "u10", "v10", "swh", "sst"]]
    arr = np.array(ordered, dtype=np.float32).reshape(1, -1)
    if scaler is not None:
        arr = scaler.transform(arr).astype(np.float32)
    return arr.flatten()


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oil-Spill EO Console",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS for dark console look ─────────────────────────────────────────
st.markdown("""
<style>
    .block-container {padding-top:1.5rem; padding-bottom:1rem;}
    h1, h2, h3 {font-family:'JetBrains Mono',monospace;}
    div[data-testid="stMetric"] {
        background:#111827; border:1px solid #1e293b;
        border-radius:8px; padding:12px 16px;
    }
    div[data-testid="stMetric"] label {color:#94a3b8 !important; font-size:0.75rem;}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#00e5ff !important;}
    .stTabs [data-baseweb="tab-list"] {gap:0;}
    .stTabs [data-baseweb="tab"] {
        background:#111827; border:1px solid #1e293b;
        color:#94a3b8; padding:8px 20px; font-family:monospace;
    }
    .stTabs [aria-selected="true"] {
        background:#0a0e17 !important; color:#00e5ff !important;
        border-bottom:2px solid #00e5ff;
    }
</style>
""", unsafe_allow_html=True)

# ── header ───────────────────────────────────────────────────────────────────
st.markdown("## 🛰️  Oil-Spill Segmentation — Earth Observation Console")
st.caption(f"Device: `{DEVICE}` · Checkpoint dir: `{CKPT_DIR}`")

# ── tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌍 Global Coverage & Geographic Split",
    "🔬 Live Inference & FiLM Stress-Test Lab",
    "📊 Cross-Geographical Benchmarks",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Global Coverage & Geographic Split
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    splits_df = load_splits()
    env_df = load_env()

    st.markdown("### 🗺️ Scene Locations")

    # ── scatter-geo map ──────────────────────────────────────────────────
    map_df = splits_df.dropna(subset=["lat", "lon"]).copy()
    fig_map = px.scatter_geo(
        map_df,
        lat="lat", lon="lon",
        color="region",
        symbol="class",
        hover_name="scene_id",
        hover_data={"split": True, "region": True, "class": True},
        color_discrete_map=REGION_COLORS,
        projection="natural earth",
        title="SAR Scene Coverage",
    )
    fig_map.update_layout(
        geo=dict(
            bgcolor="#0a0e17",
            lakecolor="#111827",
            landcolor="#1a1a2e",
            oceancolor="#0d1117",
            showocean=True,
            showlakes=True,
            coastlinecolor="#334155",
        ),
        paper_bgcolor="#0a0e17",
        plot_bgcolor="#0a0e17",
        font_color="#e0e0e0",
        legend=dict(bgcolor="#111827", bordercolor="#1e293b"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── summary metrics ──────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Scenes", len(splits_df))
    col2.metric("Regions", splits_df["region"].nunique())
    col3.metric("Classes", splits_df["class"].nunique())
    col4.metric("Train / Val / Test",
                f"{(splits_df['split']=='train').sum()} / "
                f"{(splits_df['split']=='val').sum()} / "
                f"{(splits_df['split']=='test').sum()}")

    st.markdown("---")

    # ── split distribution charts ────────────────────────────────────────
    left, right = st.columns(2)

    with left:
        st.markdown("#### Region × Class Distribution")
        cross = splits_df.groupby(["region", "class"]).size().reset_index(name="count")
        fig_bar = px.bar(
            cross, x="region", y="count", color="class",
            barmode="group",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"],
        )
        fig_bar.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
            font_color="#e0e0e0", height=350,
            xaxis_title="Region", yaxis_title="Count",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown("#### Train / Val / Test Split")
        split_counts = splits_df.groupby(["split", "region"]).size().reset_index(name="count")
        fig_split = px.sunburst(
            split_counts, path=["split", "region"], values="count",
            color="split",
            color_discrete_map={"train": "#3b82f6", "val": "#f59e0b", "test": "#10b981"},
        )
        fig_split.update_layout(
            paper_bgcolor="#0a0e17", font_color="#e0e0e0", height=350,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_split, use_container_width=True)

    # ── environmental feature distributions ──────────────────────────────
    st.markdown("### 🌊 Environmental Feature Distributions by Region")
    raw_env_cols = ["wind_speed", "u10", "v10", "swh", "sst"]
    available_env_cols = [c for c in raw_env_cols if c in env_df.columns]
    if available_env_cols:
        env_feat = st.selectbox("Feature", available_env_cols, key="env_feat_tab1")
        fig_env = px.violin(
            env_df, x="region", y=env_feat, color="region",
            box=True, points="outliers",
            color_discrete_map=REGION_COLORS,
        )
        fig_env.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
            font_color="#e0e0e0", height=350, showlegend=False,
        )
        st.plotly_chart(fig_env, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Live Inference & FiLM Stress-Test Lab
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    patches_df = load_patches()
    env_df2 = load_env()
    scaler = load_env_scaler()

    st.markdown("### ⚔️ Side-by-Side Model Duel")

    # ── model selectors ──────────────────────────────────────────────────
    all_models = CFG["models"]
    available_models = [m for m in all_models if (CKPT_DIR / f"{m}_best.pt").exists()]
    if not available_models:
        st.warning("No model checkpoints found. Train models first.")
        st.stop()

    mc1, mc2 = st.columns(2)
    with mc1:
        model_a_name = st.selectbox("Model A", available_models, index=0, key="model_a")
    with mc2:
        default_b = min(1, len(available_models) - 1)
        model_b_name = st.selectbox("Model B", available_models, index=default_b, key="model_b")

    # ── patch selector ───────────────────────────────────────────────────
    test_patches = patches_df[patches_df["split"] == "test"].reset_index(drop=True)
    if test_patches.empty:
        test_patches = patches_df.head(20)

    patch_id = st.selectbox(
        "Select test patch",
        test_patches["patch_id"].tolist(),
        key="patch_sel",
    )
    patch_row = test_patches[test_patches["patch_id"] == patch_id].iloc[0]

    # ── environment sliders ──────────────────────────────────────────────
    st.markdown("### 🌡️ Environmental Condition Sliders")
    st.caption("Adjust to stress-test FiLM conditioning. Non-FiLM models ignore these.")

    # Get the actual env values for this scene as defaults
    scene_env = env_df2[env_df2["scene_id"] == patch_row["scene_id"]]
    env_defaults = {}
    for key in ENV_LABELS:
        if not scene_env.empty and key in scene_env.columns:
            env_defaults[key] = float(scene_env[key].iloc[0])
        else:
            env_defaults[key] = ENV_LABELS[key][4]

    env_cols_ui = st.columns(5)
    raw_env = {}
    for i, (key, (label, unit, lo, hi, default)) in enumerate(ENV_LABELS.items()):
        with env_cols_ui[i]:
            raw_env[key] = st.slider(
                f"{label} ({unit})",
                min_value=lo, max_value=hi,
                value=env_defaults.get(key, default),
                step=(hi - lo) / 100,
                key=f"env_{key}",
            )

    # ── run inference ────────────────────────────────────────────────────
    if st.button("🚀 Run Inference", type="primary", use_container_width=True):
        # Load SAR and mask
        sar_path = ROOT / patch_row["sar_path"]
        mask_path = ROOT / patch_row["mask_path"]

        if not sar_path.exists() or not mask_path.exists():
            st.error(f"Patch files not found: {sar_path}")
            st.stop()

        sar_np = np.load(str(sar_path)).astype(np.float32)
        mask_np = np.load(str(mask_path)).astype(np.float32)

        # Scale env
        env_scaled = scale_env_raw(raw_env, scaler)

        # Run both models
        res_cols = st.columns(2)
        for col, mname in zip(res_cols, [model_a_name, model_b_name]):
            with col:
                model = load_model(mname)
                if model is None:
                    st.error(f"Checkpoint not found for {mname}")
                    continue

                prob = run_inference(model, sar_np, env_scaled)
                pred = (prob >= 0.5).astype(np.float32)

                # Compute metrics
                logits_t = torch.from_numpy(prob).unsqueeze(0).unsqueeze(0)
                # Convert prob to logit space for compute_confusion
                logits_for_metric = torch.logit(torch.clamp(logits_t, 1e-6, 1 - 1e-6))
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
                tp, fp, fn, tn = compute_confusion(logits_for_metric, mask_t)
                metrics = metrics_from_counts(tp, fp, fn, tn)

                st.markdown(f"#### 🏷️ {mname}")

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("IoU", f"{metrics['IoU']:.4f}")
                m2.metric("Dice", f"{metrics['Dice']:.4f}")
                m3.metric("Precision", f"{metrics['Precision']:.4f}")
                m4, m5, m6 = st.columns(3)
                m4.metric("Recall", f"{metrics['Recall']:.4f}")
                m5.metric("F1", f"{metrics['F1']:.4f}")
                m6.metric("FPR", f"{metrics['FPR']:.6f}")

                # Visualizations: SAR, Ground Truth, Prediction
                fig, axes = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(
                    1, 3, figsize=(10, 3.5)
                )
                vv = sar_np[0]
                axes[0].imshow(vv, cmap="gray")
                axes[0].set_title("SAR VV", color="#e0e0e0")
                axes[0].axis("off")

                axes[1].imshow(vv, cmap="gray")
                axes[1].imshow(mask_np, cmap="Reds", alpha=0.45 * (mask_np > 0))
                axes[1].set_title("Ground Truth", color="#e0e0e0")
                axes[1].axis("off")

                axes[2].imshow(vv, cmap="gray")
                axes[2].imshow(pred, cmap="cool", alpha=0.45 * (pred > 0))
                axes[2].set_title(f"Pred: {mname}", color="#e0e0e0")
                axes[2].axis("off")

                fig.patch.set_facecolor("#0a0e17")
                for ax in axes:
                    ax.set_facecolor("#0a0e17")
                __import__("matplotlib.pyplot", fromlist=["pyplot"]).tight_layout()
                st.pyplot(fig)
                __import__("matplotlib.pyplot", fromlist=["pyplot"]).close(fig)

        # ── FiLM stress-test delta ───────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🧪 FiLM Conditioning Sensitivity")
        st.caption("Shows how the proposed model's prediction changes from default to current env settings.")

        film_model = load_model("proposed_model")
        if film_model is not None and sar_path.exists():
            # Default env from dataset
            if not scene_env.empty:
                default_scaled_cols = [c for c in ENV_COLS if c in scene_env.columns]
                if default_scaled_cols:
                    default_env = scene_env[default_scaled_cols].values.flatten().astype(np.float32)
                else:
                    default_env = np.zeros(5, dtype=np.float32)
            else:
                default_env = np.zeros(5, dtype=np.float32)

            prob_default = run_inference(film_model, sar_np, default_env)
            prob_custom = run_inference(film_model, sar_np, env_scaled)
            delta = prob_custom - prob_default

            dc1, dc2, dc3 = st.columns(3)
            with dc1:
                st.markdown("**Default Env Prediction**")
                fig1, ax1 = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(4, 4))
                ax1.imshow(sar_np[0], cmap="gray")
                ax1.imshow((prob_default >= 0.5).astype(float), cmap="cool",
                          alpha=0.45 * (prob_default >= 0.5))
                ax1.axis("off")
                fig1.patch.set_facecolor("#0a0e17")
                ax1.set_facecolor("#0a0e17")
                st.pyplot(fig1)
                __import__("matplotlib.pyplot", fromlist=["pyplot"]).close(fig1)

            with dc2:
                st.markdown("**Custom Env Prediction**")
                fig2, ax2 = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(4, 4))
                ax2.imshow(sar_np[0], cmap="gray")
                ax2.imshow((prob_custom >= 0.5).astype(float), cmap="cool",
                          alpha=0.45 * (prob_custom >= 0.5))
                ax2.axis("off")
                fig2.patch.set_facecolor("#0a0e17")
                ax2.set_facecolor("#0a0e17")
                st.pyplot(fig2)
                __import__("matplotlib.pyplot", fromlist=["pyplot"]).close(fig2)

            with dc3:
                st.markdown("**Probability Delta**")
                fig3, ax3 = __import__("matplotlib.pyplot", fromlist=["pyplot"]).subplots(figsize=(4, 4))
                im = ax3.imshow(delta, cmap="RdBu_r", vmin=-1, vmax=1)
                ax3.axis("off")
                fig3.patch.set_facecolor("#0a0e17")
                ax3.set_facecolor("#0a0e17")
                fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                st.pyplot(fig3)
                __import__("matplotlib.pyplot", fromlist=["pyplot"]).close(fig3)

            st.metric(
                "Mean Abs Probability Shift",
                f"{np.abs(delta).mean():.4f}",
                delta=f"{delta.mean():+.4f} avg signed shift",
            )
    else:
        st.info("👆 Select a patch and click **Run Inference** to start the duel.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Cross-Geographical Benchmarks
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    results_df = load_results()
    history_df = load_history()
    patches_df3 = load_patches()

    st.markdown("### 📈 Model Comparison")

    if results_df.empty:
        st.warning("No comparison results found. Run `python src/evaluate.py` first.")
    else:
        # ── radar chart ──────────────────────────────────────────────────
        radar_metrics = ["IoU", "Dice", "Precision", "Recall", "F1"]
        available_radar = [m for m in radar_metrics if m in results_df.columns]

        if available_radar:
            fig_radar = go.Figure()
            for _, row in results_df.iterrows():
                vals = [row[m] for m in available_radar]
                vals.append(vals[0])  # close the polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=vals,
                    theta=available_radar + [available_radar[0]],
                    fill="toself",
                    name=row["Model"],
                    opacity=0.6,
                ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#111827",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b"),
                    angularaxis=dict(gridcolor="#1e293b"),
                ),
                paper_bgcolor="#0a0e17",
                font_color="#e0e0e0",
                height=420,
                title="Metric Radar — All Models",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── bar charts ───────────────────────────────────────────────────
        bar_left, bar_right = st.columns(2)

        with bar_left:
            st.markdown("#### IoU & Dice Comparison")
            melt_df = results_df.melt(
                id_vars=["Model"],
                value_vars=[m for m in ["IoU", "Dice"] if m in results_df.columns],
                var_name="Metric", value_name="Score",
            )
            fig_iou = px.bar(
                melt_df, x="Model", y="Score", color="Metric",
                barmode="group",
                color_discrete_sequence=["#00e5ff", "#f59e0b"],
            )
            fig_iou.update_layout(
                paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
                font_color="#e0e0e0", height=320,
            )
            st.plotly_chart(fig_iou, use_container_width=True)

        with bar_right:
            st.markdown("#### False Positive Rate")
            fpr_cols = [c for c in ["FPR", "Lookalike_FPR"] if c in results_df.columns]
            if fpr_cols:
                fpr_melt = results_df.melt(
                    id_vars=["Model"], value_vars=fpr_cols,
                    var_name="Metric", value_name="Rate",
                )
                fig_fpr = px.bar(
                    fpr_melt, x="Model", y="Rate", color="Metric",
                    barmode="group",
                    color_discrete_sequence=["#ef4444", "#f97316"],
                )
                fig_fpr.update_layout(
                    paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
                    font_color="#e0e0e0", height=320,
                )
                st.plotly_chart(fig_fpr, use_container_width=True)

        # ── inference latency ────────────────────────────────────────────
        if "Inference_ms" in results_df.columns:
            st.markdown("#### ⏱️ Inference Latency")
            fig_lat = px.bar(
                results_df, x="Model", y="Inference_ms",
                color_discrete_sequence=["#8b5cf6"],
                labels={"Inference_ms": "Latency (ms)"},
            )
            fig_lat.update_layout(
                paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
                font_color="#e0e0e0", height=280,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

        # ── results table ────────────────────────────────────────────────
        st.markdown("#### 📋 Full Results Table")
        st.dataframe(
            results_df.style.format({
                c: "{:.4f}" for c in results_df.columns if c != "Model"
            }).set_properties(**{"background-color": "#111827", "color": "#e0e0e0"}),
            use_container_width=True,
        )

    # ── training history ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📉 Training History (Proposed Model)")

    if history_df.empty:
        st.info("No training history found.")
    else:
        h_left, h_right = st.columns(2)
        with h_left:
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=history_df["epoch"], y=history_df["train_loss"],
                name="Train Loss", line=dict(color="#ef4444"),
            ))
            fig_loss.add_trace(go.Scatter(
                x=history_df["epoch"], y=history_df["val_loss"],
                name="Val Loss", line=dict(color="#f59e0b"),
            ))
            fig_loss.update_layout(
                title="Loss Curves", paper_bgcolor="#0a0e17",
                plot_bgcolor="#111827", font_color="#e0e0e0", height=300,
                xaxis_title="Epoch", yaxis_title="Loss",
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with h_right:
            fig_met = go.Figure()
            if "val_iou" in history_df.columns:
                fig_met.add_trace(go.Scatter(
                    x=history_df["epoch"], y=history_df["val_iou"],
                    name="Val IoU", line=dict(color="#00e5ff"),
                ))
            if "val_dice" in history_df.columns:
                fig_met.add_trace(go.Scatter(
                    x=history_df["epoch"], y=history_df["val_dice"],
                    name="Val Dice", line=dict(color="#10b981"),
                ))
            fig_met.update_layout(
                title="Validation Metrics", paper_bgcolor="#0a0e17",
                plot_bgcolor="#111827", font_color="#e0e0e0", height=300,
                xaxis_title="Epoch", yaxis_title="Score",
            )
            st.plotly_chart(fig_met, use_container_width=True)

    # ── failure-case gallery ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Failure-Case Gallery")
    st.caption("Visual overlays from evaluation — focusing on Lookalike & challenging patches.")

    vis_images = sorted(VIS_DIR.glob("*.png"))
    if not vis_images:
        st.info("No visualisation images found. Run `python src/evaluate.py` to generate overlays.")
    else:
        # Group by model
        from collections import defaultdict
        model_vis = defaultdict(list)
        for img_path in vis_images:
            # Filename: modelname_patchid.png
            parts = img_path.stem.split("_", 1)
            if len(parts) >= 2:
                model_vis[parts[0]].append(img_path)
            else:
                model_vis["unknown"].append(img_path)

        vis_model = st.selectbox(
            "Filter by model",
            list(model_vis.keys()),
            key="vis_model_filter",
        )

        gallery_images = model_vis[vis_model]
        n_cols = 2
        for row_start in range(0, len(gallery_images), n_cols):
            cols = st.columns(n_cols)
            for j, col in enumerate(cols):
                idx = row_start + j
                if idx < len(gallery_images):
                    with col:
                        st.image(
                            str(gallery_images[idx]),
                            caption=gallery_images[idx].stem,
                            use_container_width=True,
                        )

    # ── geographic breakdown ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🌐 Patch Distribution by Region & Class (Test Set)")

    test_df = patches_df3[patches_df3["split"] == "test"]
    if not test_df.empty:
        geo_counts = test_df.groupby(["region", "class"]).size().reset_index(name="patches")
        fig_geo = px.bar(
            geo_counts, x="region", y="patches", color="class",
            barmode="stack",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#10b981"],
            title="Test-Set Patch Distribution",
        )
        fig_geo.update_layout(
            paper_bgcolor="#0a0e17", plot_bgcolor="#111827",
            font_color="#e0e0e0", height=350,
        )
        st.plotly_chart(fig_geo, use_container_width=True)
    else:
        st.info("No test-set patches available.")
