#!/usr/bin/env python3
"""
export_model.py - Evaluates, packages, and exports YOLO26s SAR vessel detection artifacts.

Produces:
- best.pt and last.pt checkpoints
- Held-out test set evaluation (Precision, Recall, mAP50, mAP50-95)
- Ground-truth vs prediction overlay visualizations on held-out test images
- Comprehensive yolo26s_metadata.json
- Self-contained zip archive for 1-click download from Kaggle
"""

import os
import sys
import glob
import json
import shutil
import zipfile
import argparse
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import torch
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_model")


def get_git_info() -> str:
    """Safely get Git commit or repository version information."""
    try:
        import subprocess
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return "maritime-operations-ai-kaggle-v1"


def get_hardware_info(device: Any) -> Dict[str, Any]:
    """Retrieve detailed GPU / CPU hardware metadata."""
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda
        info["device_count"] = torch.cuda.device_count()
        info["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    return info


def export_artifacts(
    weights_path: str,
    dataset_yaml: str,
    output_dir: str = "export",
    imgsz: int = 640,
    device: Any = 0,
    epochs_completed: int = 0,
    best_epoch: Optional[int] = None,
    batch_size: int = 16,
    optimizer_name: str = "auto",
    lr: float = 0.001,
    last_weights_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run held-out test set evaluation, save predictions, and package metadata."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info("=== EVALUATING HELD-OUT TEST SPLIT ON %s ===", weights_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Trained weights not found at: {weights_path}")

    # Copy checkpoints into export directory
    exported_best = os.path.join(output_dir, "best.pt")
    shutil.copy2(weights_path, exported_best)
    logger.info("Copied best weights to: %s", exported_best)

    if last_weights_path and os.path.exists(last_weights_path):
        exported_last = os.path.join(output_dir, "last.pt")
        shutil.copy2(last_weights_path, exported_last)
        logger.info("Copied last weights to: %s", exported_last)

    # Evaluate on held-out test set
    model = YOLO(exported_best)
    test_results = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
    )

    box_metrics = getattr(test_results, "box", None)
    precision = float(box_metrics.mp) if box_metrics else 0.0
    recall = float(box_metrics.mr) if box_metrics else 0.0
    map50 = float(box_metrics.map50) if box_metrics else 0.0
    map50_95 = float(box_metrics.map) if box_metrics else 0.0

    logger.info("==========================================")
    logger.info("HELD-OUT TEST SET METRICS:")
    logger.info("  Precision: %.4f (%.2f%%)", precision, precision * 100)
    logger.info("  Recall:    %.4f (%.2f%%)", recall, recall * 100)
    logger.info("  mAP50:     %.4f (%.2f%%)", map50, map50 * 100)
    logger.info("  mAP50-95:  %.4f (%.2f%%)", map50_95, map50_95 * 100)
    logger.info("==========================================")

    # Generate prediction visualizations on held-out test set
    pred_dir = os.path.join(output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    # Find test images from dataset YAML or standard structure
    with open(dataset_yaml, "r", encoding="utf-8") as f:
        import yaml
        ds_cfg = yaml.safe_load(f)
    base_ds_path = ds_cfg.get("path", "")
    test_rel = ds_cfg.get("test", "images/test")
    test_dir = os.path.join(base_ds_path, test_rel) if not os.path.isabs(test_rel) else test_rel

    test_imgs = sorted(glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(os.path.join(test_dir, "*.png")))[:10]
    if test_imgs:
        logger.info("Generating prediction overlays for %d test images...", len(test_imgs))
        model.predict(
            source=test_imgs,
            save=True,
            project=output_dir,
            name="predictions",
            exist_ok=True,
            conf=0.25,
            device=device,
        )

    # Read dataset split counts
    train_dir = os.path.join(base_ds_path, ds_cfg.get("train", "images/train"))
    val_dir = os.path.join(base_ds_path, ds_cfg.get("val", "images/val"))
    train_count = len(glob.glob(os.path.join(train_dir, "*.jpg")))
    val_count = len(glob.glob(os.path.join(val_dir, "*.jpg")))
    test_count = len(glob.glob(os.path.join(test_dir, "*.jpg")))

    metadata = {
        "model_name": "YOLO26s-SAR-Vessel",
        "model_architecture": "yolo26s",
        "task": "detect",
        "dataset_name": "OpenSAR Insight Sentinel-1 GRD",
        "dataset_splits": {
            "train": train_count,
            "val": val_count,
            "test": test_count,
        },
        "classes": {
            0: "vessel",
        },
        "image_size": imgsz,
        "epochs_completed": epochs_completed,
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "optimizer": optimizer_name,
        "learning_rate": lr,
        "mixed_precision": True,
        "test_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "map50": round(map50, 4),
            "map50_95": round(map50_95, 4),
        },
        "training_date": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_info(),
        "hardware_info": get_hardware_info(device),
    }

    meta_path = os.path.join(output_dir, "yolo26s_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved metadata to: %s", meta_path)

    # Package everything into a single zip bundle for 1-click download from Kaggle
    zip_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "yolo26s_kaggle_artifacts.zip")
    logger.info("Creating downloadable zip archive: %s", zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, output_dir)
                zipf.write(full_path, arcname=rel_path)

    logger.info("=== EXPORT COMPLETE! ===")
    logger.info("Artifacts packaged at: %s", zip_path)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO26s SAR vessel model artifacts.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt weights")
    parser.add_argument("--dataset-yaml", type=str, default="dataset_run.yaml", help="Path to dataset YAML")
    parser.add_argument("--output-dir", type=str, default="export", help="Output directory")
    parser.add_argument("--last-weights", type=str, default=None, help="Path to last.pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", type=str, default="0", help="Device (0 for CUDA)")
    parser.add_argument("--epochs", type=int, default=0, help="Epochs completed")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    export_artifacts(
        weights_path=args.weights,
        dataset_yaml=args.dataset_yaml,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        device=args.device if args.device != "cpu" else "cpu",
        epochs_completed=args.epochs,
        batch_size=args.batch_size,
        last_weights_path=args.last_weights,
    )
