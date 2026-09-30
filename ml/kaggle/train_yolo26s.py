#!/usr/bin/env python3
"""
train_yolo26s.py - Autonomous Kaggle GPU training script for YOLO26s SAR vessel detection.

Key features:
- Automatically validates CUDA GPU availability and hardware specs (Tesla T4)
- Fails with a clear message if CUDA is not available (strictly prevents CPU/MPS runs)
- Runs end-to-end dataset audit and generates execution-ready YAML
- Trains YOLO26s with mixed precision (AMP) and radar-specific augmentations
- Evaluates held-out test set (split='test')
- Saves ground-truth and prediction visualizations
- Exports best.pt, last.pt, yolo26s_metadata.json, and yolo26s_kaggle_artifacts.zip
"""

import os
import sys
import time
import argparse
import logging
import torch
from ultralytics import YOLO

# Import local helper modules
from dataset_check import run_dataset_check, find_dataset_dir
from export_model import export_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_yolo26s")


def verify_gpu_environment(allow_cpu_override: bool = False) -> int:
    """Verify that a genuine CUDA GPU (e.g. Tesla T4) is active."""
    if not torch.cuda.is_available():
        error_msg = (
            "\n" + "=" * 70 + "\n"
            "CRITICAL ERROR: CUDA GPU is NOT available!\n"
            "Project policy strictly requires training on a free Kaggle GPU (e.g. Tesla T4).\n"
            "Mac MPS or CPU training is NOT permitted for production YOLO26s training.\n\n"
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
            return "cpu"

    device_id = 0
    gpu_name = torch.cuda.get_device_name(device_id)
    vram_gb = torch.cuda.get_device_properties(device_id).total_memory / (1024**3)
    cuda_ver = torch.version.cuda

    logger.info("=" * 60)
    logger.info("GPU ENVIRONMENT VERIFIED:")
    logger.info("  Device:        CUDA:%d", device_id)
    logger.info("  GPU Model:     %s", gpu_name)
    logger.info("  Total VRAM:    %.2f GB", vram_gb)
    logger.info("  CUDA Version:  %s", cuda_ver)
    logger.info("  Mixed Prec.:   Enabled (FP16 AMP)")
    logger.info("=" * 60)

    return device_id


def main():
    parser = argparse.ArgumentParser(description="Train YOLO26s on Kaggle GPU for SAR vessel detection.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to opensar_filtered dataset root")
    parser.add_argument("--epochs", type=int, default=150, help="Maximum epochs to train (default: 150)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--patience", type=int, default=25, help="Early stopping patience (default: 25)")
    parser.add_argument("--pretrained", type=str, default="yolo26s.pt", help="Pretrained model weights")
    parser.add_argument("--project-dir", type=str, default="kaggle_runs", help="Project output directory")
    parser.add_argument("--export-dir", type=str, default="export", help="Artifact export directory")
    parser.add_argument("--allow-cpu-debug", action="store_true", help="Allow CPU only for fast syntax checks")
    args = parser.parse_args()

    # 1. Enforce CUDA GPU verification
    device = verify_gpu_environment(allow_cpu_override=args.allow_cpu_debug)

    # 2. Audit dataset and generate clean execution YAML
    logger.info("Step 1: Auditing dataset structure and label integrity...")
    output_yaml = os.path.join(args.project_dir, "dataset_run.yaml")
    vis_dir = os.path.join(args.project_dir, "ground_truth_samples")

    success, dataset_yaml_path, split_reports = run_dataset_check(
        data_dir=args.data_dir,
        output_yaml=output_yaml,
        vis_dir=vis_dir,
    )
    if not success:
        raise ValueError("OpenSAR dataset audit failed. Please review error logs above.")

    # 3. Instantiate YOLO26s standard detection model
    logger.info("Step 2: Initializing YOLO26s standard detection model from %s...", args.pretrained)
    model = YOLO(args.pretrained)

    # 4. Train with mixed precision, early stopping, and radar augmentations
    logger.info("Step 3: Commencing training on CUDA:%s for up to %d epochs...", device, args.epochs)
    start_time = time.time()

    train_results = model.train(
        data=dataset_yaml_path,
        epochs=min(args.epochs, 150),
        patience=args.patience,
        batch=args.batch_size,
        imgsz=args.imgsz,
        device=device,
        amp=True,                  # Mixed precision FP16
        project=args.project_dir,
        name="yolo26s_sar_vessel",
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
        seed=42,
        deterministic=True,
        # SAR Radar-specific augmentations (no color jitter, radar intensity invariant)
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        lr0=0.001,
        lrf=0.01,
    )

    elapsed_time_s = time.time() - start_time
    logger.info("Training concluded in %.2f minutes (%.2f seconds)", elapsed_time_s / 60.0, elapsed_time_s)

    # 5. Locate best and last weights
    trainer = getattr(model, "trainer", None)
    best_weights = str(trainer.best) if trainer and hasattr(trainer, "best") else os.path.join(args.project_dir, "yolo26s_sar_vessel", "weights", "best.pt")
    last_weights = str(trainer.last) if trainer and hasattr(trainer, "last") else os.path.join(args.project_dir, "yolo26s_sar_vessel", "weights", "last.pt")

    epochs_completed = int(trainer.epoch + 1) if trainer and hasattr(trainer, "epoch") else args.epochs
    best_epoch = int(trainer.best_epoch) if trainer and hasattr(trainer, "best_epoch") else None

    # 6. Evaluate on held-out test set, generate prediction visualizations, and package metadata
    logger.info("Step 4: Evaluating held-out test split and exporting production artifacts...")
    metadata = export_artifacts(
        weights_path=best_weights,
        dataset_yaml=dataset_yaml_path,
        output_dir=args.export_dir,
        imgsz=args.imgsz,
        device=device,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        batch_size=args.batch_size,
        optimizer_name="auto",
        lr=0.001,
        last_weights_path=last_weights,
    )

    logger.info("=" * 60)
    logger.info("KAGGLE TRAINING SUCCESSFUL!")
    logger.info("Artifacts ready for download:")
    logger.info("  1. %s/best.pt", args.export_dir)
    logger.info("  2. %s/last.pt", args.export_dir)
    logger.info("  3. %s/yolo26s_metadata.json", args.export_dir)
    logger.info("  4. %s/yolo26s_kaggle_artifacts.zip", os.path.dirname(os.path.abspath(args.export_dir)))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
