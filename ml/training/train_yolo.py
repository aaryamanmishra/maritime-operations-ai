import os
import time
import json
import logging
from typing import Dict, Any
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_yolo26s_vessel_detector(
    dataset_yaml: str = "data/processed/opensar_filtered/dataset.yaml",
    pretrained_weights: str = "yolo26s.pt",
    output_dir: str = "ml/weights",
    epochs: int = 100,
    batch_size: int = 16,
    imgsz: int = 640,
    patience: int = 20,
) -> Dict[str, Any]:
    """
    Train and validate the production YOLO26s vessel detector on OpenSAR Insight GRD data.
    Configured explicitly for standard object detection (single class: 0=vessel).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine device: MPS (Apple Silicon), CUDA, or CPU
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Training YOLO26s on device: %s", device)
    logger.info("Dataset YAML: %s", dataset_yaml)
    logger.info("Pretrained base: %s", pretrained_weights)

    # Load standard detection model explicitly
    model = YOLO(pretrained_weights)
    logger.info("Instantiated YOLO26s task: %s, nc: %s", getattr(model, "task", "detect"), getattr(model.model, "nc", 80))

    # Train model with SAR-appropriate augmentations
    start_train_time = time.time()
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project="ml/runs",
        name="yolo26s_sar_vessel",
        exist_ok=True,
        save=True,
        plots=True,
        val=True,
        seed=42,
        deterministic=True,
        # SAR radar augmentation policy
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
    train_duration_s = time.time() - start_train_time
    logger.info("Training completed in %.2f seconds", train_duration_s)

    # Evaluate on held-out test split
    logger.info("Evaluating on held-out test split...")
    test_metrics = model.val(
        data=dataset_yaml,
        split="test",
        imgsz=imgsz,
        device=device,
        plots=True,
    )

    # Save visual test predictions on representative test images
    test_pred_dir = "data/diagnostics/predictions"
    os.makedirs(test_pred_dir, exist_ok=True)
    import glob
    test_imgs = sorted(glob.glob("data/processed/opensar_filtered/images/test/*.jpg"))[:6]
    best_weights_path = str(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else pretrained_weights
    best_model = YOLO(best_weights_path)
    if test_imgs:
        logger.info("Saving prediction visualizations for %d test images to %s...", len(test_imgs), test_pred_dir)
        best_model.predict(
            source=test_imgs,
            save=True,
            project="data/diagnostics",
            name="predictions",
            exist_ok=True,
            conf=0.25,
            device=device,
        )

    # Measure CPU single-sample inference latency
    model_cpu = YOLO(model.trainer.best if hasattr(model, "trainer") and hasattr(model.trainer, "best") else pretrained_weights)
    dummy_input = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32)
    # Warmup
    for _ in range(5):
        _ = model_cpu.predict(dummy_input, device="cpu", verbose=False)
    
    latencies = []
    for _ in range(20):
        t0 = time.time()
        _ = model_cpu.predict(dummy_input, device="cpu", verbose=False)
        latencies.append((time.time() - t0) * 1000.0)
    avg_latency_ms = sum(latencies) / len(latencies)

    # Extract metrics safely from Ultralytics metric object
    box_metrics = getattr(test_metrics, "box", None)
    precision = float(box_metrics.mp) if box_metrics else 0.0
    recall = float(box_metrics.mr) if box_metrics else 0.0
    map50 = float(box_metrics.map50) if box_metrics else 0.0
    map50_95 = float(box_metrics.map) if box_metrics else 0.0

    best_weights_path = str(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else pretrained_weights
    final_target_path = os.path.join(output_dir, "yolo26s_sar_vessel.pt")
    
    # Copy best weights to ml/weights/yolo26s_sar_vessel.pt and backend/ml/weights/
    import shutil
    shutil.copy2(best_weights_path, final_target_path)
    backend_weights_dir = "backend/ml/weights"
    os.makedirs(backend_weights_dir, exist_ok=True)
    shutil.copy2(final_target_path, os.path.join(backend_weights_dir, "yolo26s_sar_vessel.pt"))

    model_size_mb = os.path.getsize(final_target_path) / (1024 * 1024)

    metadata = {
        "model_name": "YOLO26s-SAR-Vessel",
        "model_version": "yolo26s-opensar-v1",
        "architecture": "YOLO26s (Ultralytics / OpenSAR Insight)",
        "task": "vessel-detection",
        "classes": ["VESSEL"],
        "dataset": {
            "name": "OpenSAR Insight vessel-detection-dataset",
            "repository": "https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset",
            "modality": "Sentinel-1 GRD VV/VH",
            "resolution": "512x512 patches",
            "license": "Composite (MIT OpenSARInsight / AGPL-3.0 Ultralytics / Copernicus Open Access)",
        },
        "training": {
            "epochs_run": epochs,
            "imgsz": imgsz,
            "batch_size": batch_size,
            "device": str(device),
            "duration_seconds": round(train_duration_s, 2),
        },
        "evaluation_held_out_test": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "model_size_mb": round(model_size_mb, 2),
            "cpu_inference_latency_ms": round(avg_latency_ms, 2),
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    meta_path = os.path.join(output_dir, "yolo26s_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    with open(os.path.join(backend_weights_dir, "yolo26s_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Evaluation results saved to %s:", meta_path)
    logger.info(json.dumps(metadata["evaluation_held_out_test"], indent=2))
    return metadata


if __name__ == "__main__":
    train_yolo26s_vessel_detector()
