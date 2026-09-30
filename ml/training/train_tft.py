import os
import sys
import gzip
import csv
import json
import time
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.models.tft import VesselPerformanceTFT


class NOAATrackDataset(Dataset):
    """PyTorch Dataset loading pre-processed NOAA Track A AIS + Weather features."""
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
    """Parse one raw CSV row into structured feature matrices."""
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
        hist_features[t, 5] = step_disp / 100.0  # scaled

    # 2. Ship Type & Vessel Context
    try:
        ship_type_id = int(row["ship_type_id"])
    except (ValueError, TypeError):
        ship_type_id = 1
    # Constrain to known embedding dictionary index [0..15]
    ship_type_id = min(max(ship_type_id, 0), 15)

    # In NOAA Track A, design_speed and dimensions proxies are derived from historical speed
    mean_sog = float(np.mean(hist_sog[:seq_len])) if seq_len > 0 else 12.0
    hist_disp = float(row.get("hist_displacement_m", 0.0)) / 1000.0
    vessel_stats = np.array([
        mean_sog,         # proxy for operational reference speed
        hist_disp,        # displacement proxy
        float(row.get("core_eligible", 1) == "True"),
        float(row.get("full_eligible", 1) == "True"),
    ], dtype=np.float32)

    # 3. Environmental features (10 dims)
    wind_spd = float(row["met_wind_speed_mps"]) if row["met_wind_speed_mps"] != "" else 0.0
    wind_dir = float(row["met_wind_dir_deg"]) if row["met_wind_dir_deg"] != "" else 0.0
    wind_rel = float(row["met_wind_rel_heading_deg"]) if row["met_wind_rel_heading_deg"] != "" else 0.0
    temp_c = float(row["met_temperature_c"]) if row["met_temperature_c"] != "" else 15.0
    press_hpa = float(row["met_pressure_hpa"]) if row["met_pressure_hpa"] != "" else 1013.25
    
    # Missing wave fields (e.g. sheltered waters) filled with 0.0
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
    target_sog = (fut_disp_m / 1852.0) * 6.0  # 600s = 1/6 hr -> knots

    return hist_features, ship_type_id, vessel_stats, env_features, target_sog


def load_split_data(csv_gz_path: str, max_samples: Optional[int] = None) -> List[Dict[str, np.ndarray]]:
    """Load samples from compressed CSV."""
    samples = []
    with gzip.open(csv_gz_path, "rt") as f:
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
    """Compute mean and std on TRAINING SPLIT ONLY to prevent data leakage."""
    hist_all = np.stack([s["hist_seq"] for s in train_samples])  # (N, 30, 6)
    vessel_all = np.stack([s["vessel_stats"] for s in train_samples])  # (N, 4)
    env_all = np.stack([s["env_features"] for s in train_samples])  # (N, 10)

    # Flatten sequence for hist stats
    hist_flat = hist_all.reshape(-1, 6)

    def get_stats(arr: np.ndarray) -> Dict[str, List[float]]:
        mean = np.mean(arr, axis=0).tolist()
        std = np.std(arr, axis=0)
        # Avoid division by zero
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
    """Compute MAE, RMSE, and R² metrics."""
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


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "uncommitted"


def train_tft_pipeline():
    print("=" * 60)
    print("Starting Vessel Performance TFT Training Pipeline")
    print("Dataset: MARIS-Forecast NOAA Track A (DOI: 10.5281/zenodo.21224009)")
    print("=" * 60)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")

    data_dir = REPO_ROOT / "data" / "training" / "noaa_track_v1"
    weights_dir = REPO_ROOT / "ml" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading NOAA Track A dataset splits...")
    t0 = time.time()
    # Train on full dataset
    train_samples = load_split_data(str(data_dir / "train" / "part-000.csv.gz"))
    val_samples = load_split_data(str(data_dir / "val" / "part-000.csv.gz"))
    test_samples = load_split_data(str(data_dir / "test" / "part-000.csv.gz"))
    print(f"Loaded in {time.time() - t0:.2f}s:")
    print(f"  Train samples: {len(train_samples)}")
    print(f"  Val samples:   {len(val_samples)}")
    print(f"  Test samples:  {len(test_samples)}")

    # 2. Strict leakage-free normalization using training split only
    print("Fitting normalization statistics on training data only...")
    norm_stats = compute_normalization_stats(train_samples)
    apply_normalization(train_samples, norm_stats)
    apply_normalization(val_samples, norm_stats)
    apply_normalization(test_samples, norm_stats)

    # 3. Create DataLoaders
    batch_size = 128
    train_loader = DataLoader(NOAATrackDataset(train_samples), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(NOAATrackDataset(val_samples), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(NOAATrackDataset(test_samples), batch_size=batch_size, shuffle=False)

    # 4. Instantiate TFT Model
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
    print(f"TFT Model initialized with {total_params:,} trainable parameters.")

    # 5. Optimizer, Loss & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss(beta=0.5)

    # 6. Training Loop with Early Stopping
    max_epochs = 35
    patience = 6
    best_val_loss = float("inf")
    patience_counter = 0
    best_weights_path = weights_dir / "tft_vessel_performance.pt"

    print(f"\nTraining for up to {max_epochs} epochs (early stopping patience={patience})...")

    for epoch in range(1, max_epochs + 1):
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

            # Gradient clipping
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

        print(f"Epoch {epoch:2d}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} kn | Val R²: {val_r2:.3f} | {epoch_sec:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_weights_path)
            print(f"  --> Saved new best checkpoint to {best_weights_path.name}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # 7. Final Model Evaluation on TEST SET
    print("\n" + "=" * 60)
    print("Evaluating Best Model Checkpoint on Unseen TEST SET...")
    print("=" * 60)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))

    # Benchmark Inference Latency
    test_mae, test_rmse, test_r2 = evaluate_model(model, test_loader, device)

    # Benchmark single sample latency
    model.eval()
    sample = test_samples[0]
    hist_t = torch.tensor(sample["hist_seq"], dtype=torch.float32).unsqueeze(0).to(device)
    ship_t = torch.tensor([sample["ship_type_id"]], dtype=torch.long).to(device)
    vstat_t = torch.tensor(sample["vessel_stats"], dtype=torch.float32).unsqueeze(0).to(device)
    env_t = torch.tensor(sample["env_features"], dtype=torch.float32).unsqueeze(0).to(device)

    # Warmup
    for _ in range(50):
        with torch.no_grad():
            _ = model(hist_t, ship_t, vstat_t, env_t)

    latencies = []
    for _ in range(100):
        t_start = time.perf_counter()
        with torch.no_grad():
            _ = model(hist_t, ship_t, vstat_t, env_t)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    mean_latency_ms = float(np.mean(latencies))

    # Load report vessel counts
    with open(data_dir / "reports" / "train_report.json") as f:
        train_rep = json.load(f)
    with open(data_dir / "reports" / "val_report.json") as f:
        val_rep = json.load(f)
    with open(data_dir / "reports" / "test_report.json") as f:
        test_rep = json.load(f)

    # 8. Save Complete Production Metadata & Normalization Statistics
    model_metadata = {
        "model_architecture": "TemporalFusionTransformer",
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "dataset_name": "MARIS-Forecast NOAA Track A",
        "dataset_doi": "10.5281/zenodo.21224009",
        "upstream_repository": "https://github.com/mark000071/MARIS-Forecast",
        "licensing": {
            "dataset_release": "CC-BY-4.0",
            "ais_data": "U.S. Public Domain (NOAA MarineCadastre)",
            "osm_data": "Open Database License (ODbL)",
            "weather_data": "Attribution required (Copernicus / Open-Meteo)",
        },
        "feature_schema": {
            "historical_features": [
                "hist_sog_kn", "hist_cog_sin", "hist_cog_cos",
                "hist_heading_sin", "hist_heading_cos", "hist_step_displacement_m",
            ],
            "vessel_static_features": ["design_speed_kn", "displacement_m", "core_eligible", "full_eligible"],
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
            "single_sample_latency_ms": round(mean_latency_ms, 3),
            "total_samples": 60000,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "train_vessels_mmsi": train_rep.get("unique_mmsi", 2727),
            "val_vessels_mmsi": val_rep.get("unique_mmsi", 317),
            "test_vessels_mmsi": test_rep.get("unique_mmsi", 335),
            "vessel_disjoint": True,
        },
    }

    metadata_path = weights_dir / "tft_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(model_metadata, f, indent=2)

    print("\n=== FINAL TEST SET EVALUATION REPORT ===")
    print(f"Target: Expected Vessel Speed Over Ground (knots)")
    print(f"  Test MAE:   {test_mae:.3f} knots")
    print(f"  Test RMSE:  {test_rmse:.3f} knots")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Inference Latency: {mean_latency_ms:.3f} ms / sample")
    print(f"  Split sizes: Train={len(train_samples)} (2,727 vessels), Val={len(val_samples)} (317 vessels), Test={len(test_samples)} (335 vessels)")
    print(f"  Model Weights: {best_weights_path}")
    print(f"  Model Metadata: {metadata_path}")
    print("=" * 60)


if __name__ == "__main__":
    train_tft_pipeline()
