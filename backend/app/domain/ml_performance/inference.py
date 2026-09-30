import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from app.core.logging import logger
from app.domain.ml_performance.models import PerformancePrediction
from app.domain.ml_performance.tft import VesselPerformanceTFT
from app.domain.weather_routing.models import (
    MarineWeatherConditions,
    VesselCharacteristics,
)

# Locate weights path (supports both host and docker mount paths)
WEIGHT_CANDIDATE_PATHS = [
    Path(__file__).resolve().parents[4] / "ml" / "weights",
    Path("/app/ml/weights"),
    Path("ml/weights"),
]


class TFTPerformancePredictor:
    """
    Production inference engine for vessel speed loss and performance prediction.
    Uses the trained Temporal Fusion Transformer on MARIS-Forecast NOAA Track A.
    """
    _instance: Optional["TFTPerformancePredictor"] = None

    def __init__(self):
        self.device = torch.device("cpu")  # CPU inference for API stability & low latency
        self.model: VesselPerformanceTFT | None = None
        self.metadata: dict[str, Any] | None = None
        self.norm_stats: dict[str, Any] | None = None
        self._load_model_artifacts()

    @classmethod
    def get_instance(cls) -> "TFTPerformancePredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model_artifacts(self) -> None:
        weights_dir = None
        for p in WEIGHT_CANDIDATE_PATHS:
            if p.exists() and (p / "tft_metadata.json").exists():
                weights_dir = p
                break

        if not weights_dir:
            logger.warning("TFT model weights directory not found. Using naval architecture fallback.")
            return

        meta_file = weights_dir / "tft_metadata.json"
        weights_file = weights_dir / "tft_vessel_performance.pt"

        try:
            with open(meta_file, "r") as f:
                self.metadata = json.load(f)
            self.norm_stats = self.metadata.get("normalization_statistics", {})

            # Instantiate model
            model = VesselPerformanceTFT(
                hist_num_features=6,
                env_num_features=10,
                num_ship_types=16,
                ship_embed_dim=8,
                hidden_dim=48,
                num_heads=2,
                dropout=0.0,
            )
            model.load_state_dict(torch.load(weights_file, map_location=self.device))
            model.eval()
            self.model = model
            logger.info(f"Loaded TFT vessel performance model from {weights_file}")
        except Exception as exc:
            logger.error(f"Failed loading TFT model: {exc}")
            self.model = None

    def predict_performance(
        self,
        vessel: VesselCharacteristics,
        weather: MarineWeatherConditions,
        bearing_deg: float,
        distance_nm: float,
    ) -> PerformancePrediction:
        """
        Predict operational vessel speed under marine environmental conditions.
        """
        calm_speed = vessel.design_speed_kn

        # Calculate relative angles
        rel_wind_deg = (weather.wind_direction_deg - bearing_deg) % 360.0
        if rel_wind_deg > 180.0:
            rel_wind_deg -= 360.0

        rel_wave_deg = (weather.wave_direction_deg - bearing_deg) % 360.0
        if rel_wave_deg > 180.0:
            rel_wave_deg -= 360.0

        # If model is loaded, execute TFT inference
        if self.model is not None and self.norm_stats:
            try:
                pred_sog = self._infer_tft(vessel, weather, bearing_deg, rel_wind_deg, rel_wave_deg)
                # Ensure plausible physics boundaries (cannot exceed 110% calm speed or drop below 2 kn)
                pred_sog = min(max(pred_sog, 2.0), calm_speed * 1.05)
                speed_loss = max(calm_speed - pred_sog, 0.0)
                retention = pred_sog / calm_speed

                return PerformancePrediction(
                    calm_reference_speed_kn=round(calm_speed, 2),
                    predicted_sog_kn=round(pred_sog, 2),
                    speed_loss_kn=round(speed_loss, 2),
                    speed_retention_ratio=round(retention, 3),
                    feature_attributions={
                        "wind_speed_mps": weather.wind_speed_mps,
                        "relative_wind_deg": round(rel_wind_deg, 1),
                        "wave_height_m": weather.wave_height_m,
                        "relative_wave_deg": round(rel_wave_deg, 1),
                    },
                )
            except Exception as e:
                logger.warning(f"TFT inference failed, using analytical fallback: {e}")

        # Analytical Kwon baseline fallback if model is uninitialized
        return self._analytical_kwon_baseline(vessel, weather, bearing_deg, rel_wind_deg, rel_wave_deg)

    def _infer_tft(
        self,
        vessel: VesselCharacteristics,
        weather: MarineWeatherConditions,
        bearing_deg: float,
        rel_wind_deg: float,
        rel_wave_deg: float,
    ) -> float:
        """Run forward pass through TFT model combined with naval architecture speed degradation."""
        type_map = {"cargo": 1, "tanker": 2, "passenger": 5, "fishing": 7, "tug": 8, "service": 9}
        type_id = type_map.get(vessel.vessel_type, 1)

        # 1. Historical sequence (consistent with NOAA Track A training representation)
        hist_sog = vessel.design_speed_kn
        bearing_rad = math.radians(bearing_deg)
        hist_seq = np.zeros((30, 6), dtype=np.float32)
        step_disp = (hist_sog * 0.514444 * 20.0) / 100.0  # scaled step displacement
        for t in range(30):
            hist_seq[t, 0] = hist_sog
            hist_seq[t, 1] = math.sin(bearing_rad)
            hist_seq[t, 2] = math.cos(bearing_rad)
            hist_seq[t, 3] = math.sin(bearing_rad)
            hist_seq[t, 4] = math.cos(bearing_rad)
            hist_seq[t, 5] = step_disp

        # 2. Vessel stats (scaled consistent with training data)
        disp_km = (hist_sog * 0.514444 * 600.0) / 1000.0
        vessel_stats = np.array([
            hist_sog,
            disp_km,
            1.0,
            1.0,
        ], dtype=np.float32)

        # 3. Environmental features
        wind_rad = math.radians(weather.wind_direction_deg)
        wave_rad = math.radians(weather.wave_direction_deg)
        env_features = np.array([
            weather.wind_speed_mps,
            math.sin(wind_rad),
            math.cos(wind_rad),
            math.sin(math.radians(rel_wind_deg)),
            math.cos(math.radians(rel_wind_deg)),
            weather.wave_height_m,
            math.sin(wave_rad),
            math.cos(wave_rad),
            weather.wave_period_s,
            weather.swell_wave_height_m,
        ], dtype=np.float32)

        # 4. Apply training normalization statistics
        h_mean = np.array(self.norm_stats["hist"]["mean"], dtype=np.float32)
        h_std = np.array(self.norm_stats["hist"]["std"], dtype=np.float32)
        v_mean = np.array(self.norm_stats["vessel"]["mean"], dtype=np.float32)
        v_std = np.array(self.norm_stats["vessel"]["std"], dtype=np.float32)
        e_mean = np.array(self.norm_stats["env"]["mean"], dtype=np.float32)
        e_std = np.array(self.norm_stats["env"]["std"], dtype=np.float32)

        hist_norm = (hist_seq - h_mean) / h_std
        vessel_norm = (vessel_stats - v_mean) / v_std
        env_norm = (env_features - e_mean) / e_std

        # 5. Model forward pass
        with torch.no_grad():
            hist_t = torch.tensor(hist_norm, dtype=torch.float32).unsqueeze(0)
            ship_t = torch.tensor([type_id], dtype=torch.long)
            vstat_t = torch.tensor(vessel_norm, dtype=torch.float32).unsqueeze(0)
            env_t = torch.tensor(env_norm, dtype=torch.float32).unsqueeze(0)

            out = self.model(hist_t, ship_t, vstat_t, env_t)
            tft_pred = float(out["predicted_sog_kn"].item())

        # Baseline reference from TFT
        calm_speed = vessel.design_speed_kn
        ref_speed = min(max(tft_pred, 2.0), calm_speed)

        # 6. Environmental speed degradation (Kwon / Holtrop & Mennen formulation)
        cos_wave = math.cos(math.radians(abs(rel_wave_deg)))
        dir_factor = 0.5 * (1.0 + cos_wave)  # 1.0 head sea, 0.0 following sea
        size_factor = max(1.0 - (vessel.length_m / 450.0), 0.15)
        wave_loss_ratio = 0.025 * size_factor * (weather.wave_height_m ** 1.6) * dir_factor

        cos_wind = math.cos(math.radians(abs(rel_wind_deg)))
        wind_dir_factor = 0.5 * (1.0 + cos_wind)
        wind_loss_ratio = (weather.wind_speed_mps / 30.0) * 0.04 * wind_dir_factor

        total_loss_ratio = min(wave_loss_ratio + wind_loss_ratio, 0.65)
        pred_kn = max(ref_speed * (1.0 - total_loss_ratio), 2.0)
        return pred_kn

    def _analytical_kwon_baseline(
        self,
        vessel: VesselCharacteristics,
        weather: MarineWeatherConditions,
        bearing_deg: float,
        rel_wind_deg: float,
        rel_wave_deg: float,
    ) -> PerformancePrediction:
        """Kwon (2008) analytical formula for speed loss."""
        v_calm = vessel.design_speed_kn
        cos_mu = math.cos(math.radians(abs(rel_wave_deg)))
        # Directional factor: 1.0 for head sea, 0.5 beam sea, 0.0 following sea
        dir_factor = 0.5 * (1.0 + cos_mu)
        # Kwon approximate speed loss percentage: dV/V approx alpha * Hs^2
        # Larger ships suffer less relative speed loss than smaller vessels
        size_factor = max(1.0 - (vessel.length_m / 400.0), 0.15)
        speed_loss_pct = 0.035 * size_factor * (weather.wave_height_m ** 1.8) * dir_factor
        
        # Wind effect
        wind_speed_loss = (weather.wind_speed_mps / 25.0) * 0.05 * dir_factor
        total_loss_ratio = min(speed_loss_pct + wind_speed_loss, 0.65)

        pred_speed = max(v_calm * (1.0 - total_loss_ratio), 2.0)
        speed_loss = v_calm - pred_speed

        return PerformancePrediction(
            calm_reference_speed_kn=round(v_calm, 2),
            predicted_sog_kn=round(pred_speed, 2),
            speed_loss_kn=round(speed_loss, 2),
            speed_retention_ratio=round(pred_speed / v_calm, 3),
            feature_attributions={"wave_height_m": weather.wave_height_m, "wind_speed_mps": weather.wind_speed_mps},
        )
