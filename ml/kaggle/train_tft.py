#!/usr/bin/env python3
"""
train_tft.py - Autonomous Kaggle GPU training script for Vessel Performance TFT.

Trains the Temporal Fusion Transformer (TFT) surrogate model on MARIS-Forecast NOAA Track A.
Predicts expected speed over ground (SOG in knots) given:
1. Past 30-step AIS trajectory (speed, course, heading, displacement)
2. Vessel operational profile (operational speed mean, min, max, std)
3. Dynamic marine environmental conditions (wind, waves, swell)

Key features:
- Automatically validates CUDA GPU availability and hardware specs (Tesla T4)
- Prevents CPU / Mac MPS runs unless explicitly overridden with --allow-cpu-debug
- Leakage-free feature schema: eliminates constant dummy curation flags and direct displacement shortcuts
- Strict train-only normalization statistics
- Evaluates on strictly held-out, vessel-disjoint test split
- Exports tft_vessel_performance.pt, tft_metadata.json, and tft_kaggle_artifacts.zip
"""

import os
import sys
import gzip
import csv
import json
import time
import math
import shutil
import zipfile
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_tft")

CANDIDATE_DATASET_DIRS = [
    "/kaggle/input/noaa-track-v1",
    "/kaggle/input/maris-forecast-noaa-track-a",
    "/kaggle/input/noaa-track-v1/noaa_track_v1",
    "/kaggle/input/maris-forecast-noaa-track-a/noaa_track_v1",
    "/kaggle/working/data/training/noaa_track_v1",
    "data/training/noaa_track_v1",
    "../data/training/noaa_track_v1",
    "../../data/training/noaa_track_v1",
]


# =====================================================================
# 1. TFT Neural Network Architecture
# =====================================================================

class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) with sigmoid gating."""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc(x).chunk(2, dim=-1)
        return val * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) as specified in TFT architecture."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.glu = GatedLinearUnit(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.fc1(x)
        if self.context_fc is not None and context is not None:
            if context.dim() == 2 and h.dim() == 3:
                c = self.context_fc(context).unsqueeze(1)
            else:
                c = self.context_fc(context)
            h = h + c
        h = F.elu(h)
        h = self.fc2(h)
        h = self.dropout(h)
        gated = self.glu(h)
        return self.layer_norm(self.skip(x) + gated)


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network (VSN) for weighting individual feature contributions."""
    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        context_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.var_transforms = nn.ModuleList([
            nn.Linear(1, hidden_dim) for _ in range(num_features)
        ])
        self.gating = GatedResidualNetwork(
            input_dim=num_features * hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=num_features,
            context_dim=context_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        transformed = [self.var_transforms[i](x[..., i:i+1]) for i in range(self.num_features)]
        stacked = torch.stack(transformed, dim=-2)
        flat = stacked.flatten(start_dim=-2)
        sparse_weights = F.softmax(self.gating(flat, context), dim=-1)
        weights_expanded = sparse_weights.unsqueeze(-1)
        selected = (stacked * weights_expanded).sum(dim=-2)
        return selected, sparse_weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention mechanism."""
    def __init__(self, hidden_dim: int, num_heads: int = 2):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.hidden_dim = hidden_dim
        
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = q.shape
        Q = self.q_linear(q).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(context)
        return output, attn_weights.mean(dim=1)


class VesselPerformanceTFT(nn.Module):
    """
    Temporal Fusion Transformer specialized for maritime vessel speed prediction.
    """
    def __init__(
        self,
        hist_num_features: int = 6,
        env_num_features: int = 10,
        num_ship_types: int = 16,
        ship_embed_dim: int = 8,
        hidden_dim: int = 48,
        num_heads: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Vessel Static Embedding & GRN (4 static features: [mean_sog, min_sog, max_sog, std_sog])
        self.ship_type_embed = nn.Embedding(num_ship_types, ship_embed_dim)
        self.static_vessel_fc = nn.Linear(ship_embed_dim + 4, hidden_dim)
        self.static_context_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)

        # 2. Historical Sequence VSN + LSTM
        self.hist_vsn = VariableSelectionNetwork(hist_num_features, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=1)
        self.lstm_post_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)

        # 3. Self-Attention over Historical Sequence
        self.temporal_attn = InterpretableMultiHeadAttention(hidden_dim, num_heads=num_heads)
        self.attn_post_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)

        # 4. Environmental Context VSN
        self.env_vsn = VariableSelectionNetwork(env_num_features, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        self.env_post_grn = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)

        # 5. Fusion & Output Head
        self.fusion_grn = GatedResidualNetwork(hidden_dim * 2, hidden_dim, hidden_dim, context_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        hist_seq: torch.Tensor,
        ship_type_id: torch.Tensor,
        vessel_stats: torch.Tensor,
        env_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Step 1: Compute Static Context Representation
        type_emb = self.ship_type_embed(ship_type_id)
        vessel_raw = torch.cat([type_emb, vessel_stats], dim=-1)
        vessel_emb = F.relu(self.static_vessel_fc(vessel_raw))
        static_context = self.static_context_grn(vessel_emb)

        # Step 2: Temporal Historical Processing
        hist_selected, hist_weights = self.hist_vsn(hist_seq, context=static_context)
        lstm_out, _ = self.lstm(hist_selected)
        lstm_post = self.lstm_post_grn(lstm_out, context=static_context)

        # Step 3: Self-Attention over Historical Time Steps
        attn_out, attn_weights = self.temporal_attn(lstm_post, lstm_post, lstm_post)
        attn_post = self.attn_post_grn(attn_out + lstm_post, context=static_context)
        hist_summary = attn_post[:, -1, :]

        # Step 4: Environmental Context Processing
        env_selected, env_weights = self.env_vsn(env_features, context=static_context)
        env_post = self.env_post_grn(env_selected, context=static_context)

        # Step 5: Fusion of Trajectory State and Marine Conditions
        fused_repr = torch.cat([hist_summary, env_post], dim=-1)
        fused = self.fusion_grn(fused_repr, context=static_context)

        # Step 6: Prediction Head (speed in knots >= 0)
        raw_pred = self.head(fused).squeeze(-1)
        pred_speed = F.softplus(raw_pred)

        return {
            "predicted_sog_kn": pred_speed,
            "temporal_weights": attn_weights,
            "hist_feature_weights": hist_weights,
            "env_feature_weights": env_weights,
        }


# =====================================================================
# 2. Data Loading & Leakage-Free Feature Extraction
# =====================================================================

class NOAATrackDataset(Dataset):
    """PyTorch Dataset for NOAA Track A AIS + Marine Weather sequences."""
    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        return {
            "hist_seq": torch.tensor(item["hist_seq"], dtype=torch.float32),
            "ship_type_id": torch.tensor(item["ship_type_id"], dtype=torch.long),
            "vessel_stats": torch.tensor(item["vessel_stats"], dtype=torch.float32),
            "env_features": torch.tensor(item["env_features"], dtype=torch.float32),
            "target_sog": torch.tensor(item["target_sog"], dtype=torch.float32),
        }


def extract_row_features(row: dict) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, float]:
    """
    Parse one raw CSV row into structured feature matrices.
    Audited to prevent target leakage and constant flag bias.
    """
    # 1. Historical sequence (length 30)
    hist_sog = json.loads(row["hist_sog_json"])
    hist_cog_sin = json.loads(row["hist_cog_sin_json"])
    hist_cog_cos = json.loads(row["hist_cog_cos_json"])
    hist_hdg_sin = json.loads(row["hist_heading_sin_json"])
    hist_hdg_cos = json.loads(row["hist_heading_cos_json"])
    hist_x = json.loads(row["hist_x_json"])
    hist_y = json.loads(row["hist_y_json"])

    seq_len = min(len(hist_sog), 30)
    hist_features = np.zeros((30, 6), dtype=np.float32)

    for t in range(seq_len):
        step_disp = 0.0
        if t > 0:
            step_disp = math.sqrt((hist_x[t] - hist_x[t-1])**2 + (hist_y[t] - hist_y[t-1])**2)
        hist_features[t, 0] = float(hist_sog[t])
        hist_features[t, 1] = float(hist_cog_sin[t])
        hist_features[t, 2] = float(hist_cog_cos[t])
        hist_features[t, 3] = float(hist_hdg_sin[t])
        hist_features[t, 4] = float(hist_hdg_cos[t])
        hist_features[t, 5] = step_disp / 100.0

    # 2. Ship Type & Vessel Context
    try:
        ship_type_id = int(row["ship_type_id"])
    except (ValueError, TypeError):
        ship_type_id = 1
    ship_type_id = min(max(ship_type_id, 0), 15)

    # Clean operational vessel speed dynamics: [mean, min, max, std]
    valid_sog = hist_sog[:seq_len] if seq_len > 0 else [12.0]
    mean_sog = float(np.mean(valid_sog))
    min_sog = float(np.min(valid_sog))
    max_sog = float(np.max(valid_sog))
    std_sog = float(np.std(valid_sog))

    vessel_stats = np.array([
        mean_sog,
        min_sog,
        max_sog,
        std_sog,
    ], dtype=np.float32)

    # 3. Environmental features (10 dims)
    wind_spd = float(row["met_wind_speed_mps"]) if row["met_wind_speed_mps"] != "" else 0.0
    wind_dir = float(row["met_wind_dir_deg"]) if row["met_wind_dir_deg"] != "" else 0.0
    wind_rel = float(row["met_wind_rel_heading_deg"]) if row["met_wind_rel_heading_deg"] != "" else 0.0
    wave_h = float(row["sea_wave_height_m"]) if row["sea_wave_height_m"] != "" else 0.0
    wave_dir = float(row["sea_wave_dir_deg"]) if row["sea_wave_dir_deg"] != "" else 0.0
    wave_period = float(row["sea_wave_period_s"]) if row["sea_wave_period_s"] != "" else 0.0
    swell_h = float(row["sea_swell_wave_height_m"]) if row["sea_swell_wave_height_m"] != "" else 0.0

    wind_rad = math.radians(wind_dir)
    wave_rad = math.radians(wave_dir)

    env_features = np.array([
        wind_spd,
        math.sin(wind_rad),
        math.cos(wind_rad),
        math.sin(math.radians(wind_rel)),
        math.cos(math.radians(wind_rel)),
        wave_h,
        math.sin(wave_rad),
        math.cos(wave_rad),
        wave_period,
        swell_h,
    ], dtype=np.float32)

    # 4. Target: Expected future SOG in knots over the 10-minute forecast window
    fut_disp_m = float(row["fut_displacement_m"])
    target_sog = (fut_disp_m / 1852.0) * 6.0

    return hist_features, ship_type_id, vessel_stats, env_features, target_sog


def load_split_data(csv_gz_path: str, max_samples: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
    """Load samples from compressed CSV."""
    samples = []
    with gzip.open(csv_gz_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_samples and i >= max_samples:
                break
            hist_feat, ship_id, vessel_st, env_feat, target = extract_row_features(row)
            samples.append({
                "hist_seq": hist_feat,
                "ship_type_id": ship_id,
                "vessel_stats": vessel_st,
                "env_features": env_feat,
                "target_sog": target,
            })
    return samples


def compute_normalization_stats(train_samples: List[Dict[str, np.ndarray]]) -> Dict[str, Dict[str, List[float]]]:
    """Compute mean and std on TRAINING SPLIT ONLY to strictly prevent data leakage."""
    hist_all = np.stack([s["hist_seq"] for s in train_samples])
    vessel_all = np.stack([s["vessel_stats"] for s in train_samples])
    env_all = np.stack([s["env_features"] for s in train_samples])

    hist_flat = hist_all.reshape(-1, 6)

    def get_stats(arr: np.ndarray) -> Dict[str, List[float]]:
        mean = np.mean(arr, axis=0).tolist()
        std = np.std(arr, axis=0)
        std = np.where(std < 1e-6, 1.0, std).tolist()
        return {"mean": mean, "std": std}

    return {
        "hist": get_stats(hist_flat),
        "vessel": get_stats(vessel_all),
        "env": get_stats(env_all),
    }


def apply_normalization(samples: List[Dict[str, np.ndarray]], stats: dict) -> None:
    """In-place Z-score normalization using training statistics."""
    h_mean = np.array(stats["hist"]["mean"], dtype=np.float32)
    h_std = np.array(stats["hist"]["std"], dtype=np.float32)
    v_mean = np.array(stats["vessel"]["mean"], dtype=np.float32)
    v_std = np.array(stats["vessel"]["std"], dtype=np.float32)
    e_mean = np.array(stats["env"]["mean"], dtype=np.float32)
    e_std = np.array(stats["env"]["std"], dtype=np.float32)

    for s in samples:
        s["hist_seq"] = (s["hist_seq"] - h_mean) / h_std
        s["vessel_stats"] = (s["vessel_stats"] - v_mean) / v_std
        s["env_features"] = (s["env_features"] - e_mean) / e_std


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, float]:
    """Compute MAE, RMSE, and R² metrics on unseen split."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            hist = batch["hist_seq"].to(device)
            ship_id = batch["ship_type_id"].to(device)
            vstats = batch["vessel_stats"].to(device)
            env = batch["env_features"].to(device)
            target = batch["target_sog"].to(device)

            out = model(hist, ship_id, vstats, env)
            preds.extend(out["predicted_sog_kn"].cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())

    preds = np.array(preds)
    targets = np.array(targets)

    mae = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    ss_res = np.sum((targets - preds) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return mae, rmse, r2


# =====================================================================
# 3. Environment & Hardware Verification
# =====================================================================

def verify_gpu_environment(allow_cpu_override: bool = False) -> torch.device:
    """Verify that a genuine CUDA GPU (e.g. Tesla T4) is active."""
    if not torch.cuda.is_available():
        error_msg = (
            "\n" + "=" * 70 + "\n"
            "CRITICAL ERROR: CUDA GPU is NOT available!\n"
            "Project policy strictly requires training on a free Kaggle GPU (e.g. Tesla T4).\n"
            "Mac MPS or CPU training is NOT permitted for production TFT training.\n\n"
            "To fix on Kaggle:\n"
            "  1. Go to Notebook Settings in the right sidebar\n"
            "  2. Click 'Accelerator'\n"
            "  3. Select 'GPU T4 x1' (or 'GPU T4 x2')\n"
            "  4. Restart session and run again.\n"
            + "=" * 70
        )
        if not allow_cpu_override:
            logger.error(error_msg)
            raise RuntimeError("CUDA GPU not detected. Halting execution per project policy.")
        else:
            logger.warning("Overriding GPU requirement for testing/debugging. Running on CPU.")
            return torch.device("cpu")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    logger.info("=" * 60)
    logger.info("GPU ENVIRONMENT VERIFIED:")
    logger.info("  Device:        CUDA:0")
    logger.info("  GPU Model:     %s", gpu_name)
    logger.info("  Total VRAM:    %.2f GB", vram_gb)
    logger.info("  PyTorch:       %s", torch.__version__)
    logger.info("=" * 60)

    return device


def find_dataset_dir(explicit_dir: Optional[str] = None) -> str:
    """Locate the NOAA Track A dataset directory."""
    if explicit_dir and os.path.isdir(explicit_dir):
        if os.path.exists(os.path.join(explicit_dir, "train", "part-000.csv.gz")):
            return os.path.abspath(explicit_dir)

    for path in CANDIDATE_DATASET_DIRS:
        if os.path.exists(os.path.join(path, "train", "part-000.csv.gz")):
            logger.info("Found NOAA Track A dataset at: %s", os.path.abspath(path))
            return os.path.abspath(path)

    raise FileNotFoundError(
        "Could not locate NOAA Track A dataset directory. Looked in:\n  - "
        + "\n  - ".join(CANDIDATE_DATASET_DIRS)
        + "\n\nPlease supply --data-dir pointing to the folder containing train/part-000.csv.gz"
    )


# =====================================================================
# 4. Main Training Pipeline
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Vessel Performance TFT on Kaggle GPU.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to noaa_track_v1 directory")
    parser.add_argument("--epochs", type=int, default=35, help="Maximum epochs to train (default: 35)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience (default: 6)")
    parser.add_argument("--export-dir", type=str, default="export_tft", help="Directory to export artifacts")
    parser.add_argument("--allow-cpu-debug", action="store_true", help="Allow CPU only for quick syntax checking")
    args = parser.parse_args()

    # 1. Enforce CUDA GPU verification
    device = verify_gpu_environment(allow_cpu_override=args.allow_cpu_debug)

    # 2. Locate and load dataset
    dataset_dir = find_dataset_dir(args.data_dir)
    os.makedirs(args.export_dir, exist_ok=True)

    logger.info("Loading NOAA Track A dataset splits from %s...", dataset_dir)
    t0 = time.time()
    train_samples = load_split_data(os.path.join(dataset_dir, "train", "part-000.csv.gz"))
    val_samples = load_split_data(os.path.join(dataset_dir, "val", "part-000.csv.gz"))
    test_samples = load_split_data(os.path.join(dataset_dir, "test", "part-000.csv.gz"))
    logger.info("Loaded dataset in %.2fs:", time.time() - t0)
    logger.info("  Train samples: %d", len(train_samples))
    logger.info("  Val samples:   %d", len(val_samples))
    logger.info("  Test samples:  %d", len(test_samples))

    # 3. Fit normalization strictly on training split
    logger.info("Fitting normalization statistics on training data only...")
    norm_stats = compute_normalization_stats(train_samples)
    apply_normalization(train_samples, norm_stats)
    apply_normalization(val_samples, norm_stats)
    apply_normalization(test_samples, norm_stats)

    # 4. Create DataLoaders
    train_loader = DataLoader(NOAATrackDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(NOAATrackDataset(val_samples), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(NOAATrackDataset(test_samples), batch_size=args.batch_size, shuffle=False)

    # 5. Initialize TFT Model
    model = VesselPerformanceTFT(
        hist_num_features=6,
        env_num_features=10,
        num_ship_types=16,
        ship_embed_dim=8,
        hidden_dim=48,
        num_heads=2,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("TFT Model initialized with %d trainable parameters.", total_params)

    # 6. Optimizer, Scheduler & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss(beta=0.5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = os.path.join(args.export_dir, "tft_vessel_performance.pt")

    logger.info("Commencing training for up to %d epochs (early stopping patience=%d)...", args.epochs, args.patience)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        start_epoch = time.time()

        for batch in train_loader:
            hist = batch["hist_seq"].to(device)
            ship_id = batch["ship_type_id"].to(device)
            vstats = batch["vessel_stats"].to(device)
            env = batch["env_features"].to(device)
            target = batch["target_sog"].to(device)

            optimizer.zero_grad()
            out = model(hist, ship_id, vstats, env)
            loss = criterion(out["predicted_sog_kn"], target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * hist.size(0)

        train_loss /= len(train_samples)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                hist = batch["hist_seq"].to(device)
                ship_id = batch["ship_type_id"].to(device)
                vstats = batch["vessel_stats"].to(device)
                env = batch["env_features"].to(device)
                target = batch["target_sog"].to(device)

                out = model(hist, ship_id, vstats, env)
                v_loss = criterion(out["predicted_sog_kn"], target)
                val_loss += v_loss.item() * hist.size(0)

        val_loss /= len(val_samples)
        scheduler.step(val_loss)
        epoch_sec = time.time() - start_epoch

        val_mae, val_rmse, val_r2 = evaluate_model(model, val_loader, device)
        logger.info(
            "Epoch %2d/%d | Train Loss: %.4f | Val Loss: %.4f | Val MAE: %.2f kn | Val R²: %.3f | %.1fs",
            epoch, args.epochs, train_loss, val_loss, val_mae, val_r2, epoch_sec
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            logger.info("  --> Saved new best checkpoint to %s", best_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # 7. Final Evaluation on Held-Out Test Set
    logger.info("=" * 60)
    logger.info("Evaluating Best Checkpoint on Unseen Held-Out Test Set...")
    logger.info("=" * 60)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_mae, test_rmse, test_r2 = evaluate_model(model, test_loader, device)

    # Benchmark single sample latency on CPU
    cpu_model = VesselPerformanceTFT(
        hist_num_features=6,
        env_num_features=10,
        num_ship_types=16,
        ship_embed_dim=8,
        hidden_dim=48,
        num_heads=2,
        dropout=0.0,
    )
    cpu_model.load_state_dict(torch.load(best_weights_path, map_location=torch.device("cpu")))
    cpu_model.eval()

    sample = test_samples[0]
    hist_t = torch.tensor(sample["hist_seq"], dtype=torch.float32).unsqueeze(0)
    ship_t = torch.tensor([sample["ship_type_id"]], dtype=torch.long)
    vstat_t = torch.tensor(sample["vessel_stats"], dtype=torch.float32).unsqueeze(0)
    env_t = torch.tensor(sample["env_features"], dtype=torch.float32).unsqueeze(0)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = cpu_model(hist_t, ship_t, vstat_t, env_t)

    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = cpu_model(hist_t, ship_t, vstat_t, env_t)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))

    metadata = {
        "model_architecture": "TemporalFusionTransformer",
        "model_version": "1.1.0-leakage-free",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "MARIS-Forecast NOAA Track A",
        "dataset_doi": "10.5281/zenodo.21224009",
        "feature_schema": {
            "historical_features": [
                "hist_sog_kn", "hist_cog_sin", "hist_cog_cos",
                "hist_heading_sin", "hist_heading_cos", "hist_step_displacement_m",
            ],
            "vessel_static_features": ["mean_sog_kn", "min_sog_kn", "max_sog_kn", "std_sog_kn"],
            "environmental_features": [
                "wind_speed_mps", "wind_dir_sin", "wind_dir_cos",
                "wind_rel_heading_sin", "wind_rel_heading_cos",
                "sea_wave_height_m", "sea_wave_dir_sin", "sea_wave_dir_cos",
                "sea_wave_period_s", "sea_swell_wave_height_m",
            ],
        },
        "normalization_statistics": norm_stats,
        "evaluation_metrics": {
            "test_mae_kn": round(test_mae, 3),
            "test_rmse_kn": round(test_rmse, 3),
            "test_r2": round(test_r2, 4),
            "single_sample_cpu_latency_ms": round(mean_latency_ms, 3),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "vessel_disjoint": True,
        },
    }

    meta_path = os.path.join(args.export_dir, "tft_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Package into downloadable zip
    zip_path = os.path.join(os.path.dirname(os.path.abspath(args.export_dir)), "tft_kaggle_artifacts.zip")
    logger.info("Packaging artifacts into: %s", zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(args.export_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, args.export_dir)
                zipf.write(full_path, arcname=rel_path)

    logger.info("=" * 60)
    logger.info("TFT KAGGLE TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("HELD-OUT TEST SET METRICS:")
    logger.info("  MAE:     %.3f knots", test_mae)
    logger.info("  RMSE:    %.3f knots", test_rmse)
    logger.info("  R²:      %.4f", test_r2)
    logger.info("  Latency: %.3f ms (CPU single-sample)", mean_latency_ms)
    logger.info("Artifacts bundled at: %s", zip_path)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
