# ==============================================================================
# Colab Export Helper: Evaluate Paper Checkpoint & Download All Artifacts
# Paste and run this cell in your Google Colab session!
# ==============================================================================
import json
import shutil
import hashlib
import zipfile
from pathlib import Path
import torch
from ultralytics import YOLO

try:
    from google.colab import files
except ImportError:
    files = None

BASE_DIR = Path('/content')
WORK_DIR = BASE_DIR / 'opensar_yolo11s'
PAPER_RUNS_DIR = WORK_DIR / 'paper_runs'
PAPER_EXPORT_DIR = BASE_DIR / 'yolo11s_opensar_paper_export'
PAPER_EXPORT_DIR.mkdir(parents=True, exist_ok=True)

paper_best_weight = PAPER_RUNS_DIR / 'yolo11s_opensar_slc_paper' / 'weights' / 'best.pt'
paper_last_weight = PAPER_RUNS_DIR / 'yolo11s_opensar_slc_paper' / 'weights' / 'last.pt'
paper_yaml_path = WORK_DIR / 'paper_dataset.yaml'

if not paper_best_weight.is_file():
    raise FileNotFoundError(f"Paper best weight not found at {paper_best_weight}!")

print("1. Evaluating paper-specific model on held-out test split...")
model = YOLO(str(paper_best_weight))
device = 0 if torch.cuda.is_available() else 'cpu'
test_res = model.val(
    data=str(paper_yaml_path),
    split='test',
    imgsz=512,
    device=device,
    plots=True,
    project=str(PAPER_EXPORT_DIR),
    name='test_evaluation',
    exist_ok=True,
    verbose=True,
)

box = test_res.box
best_export = PAPER_EXPORT_DIR / 'best.pt'
shutil.copy2(paper_best_weight, best_export)
if paper_last_weight.is_file():
    shutil.copy2(paper_last_weight, PAPER_EXPORT_DIR / 'last.pt')

paper_meta = {
    "model_name": "YOLO11s-SAR-Vessel-Paper-Compliant",
    "model_version": "yolo11s-opensar-slc-paper-v1",
    "dataset": "OpenSAR Insight Sentinel-1 SLC (512x512)",
    "dataset_url": "https://huggingface.co/datasets/opensar-insight/vessel-detection-dataset",
    "official_splits_preserved": True,
    "test_used_for_selection": False,
    "training": {
        "pretrained": "yolo11s.pt",
        "epochs_requested": 100,
        "epochs_completed": 37,
        "best_epoch": 17,
        "imgsz": 512,
        "batch_size": 16,
        "optimizer": "SGD",
        "lr0": 0.01,
        "seed": 42
    },
    "held_out_test_metrics": {
        "precision": float(box.mp),
        "recall": float(box.mr),
        "ap50": float(box.map50),
        "ap50_95": float(box.map),
    },
    "best_sha256": hashlib.sha256(best_export.read_bytes()).hexdigest(),
}
(PAPER_EXPORT_DIR / 'yolo11s_opensar_paper_metadata.json').write_text(json.dumps(paper_meta, indent=2))

# Create paper zip bundle
paper_bundle = BASE_DIR / 'yolo11s_opensar_paper_artifacts.zip'
with zipfile.ZipFile(paper_bundle, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in ('best.pt', 'last.pt', 'yolo11s_opensar_paper_metadata.json'):
        f = PAPER_EXPORT_DIR / name
        if f.is_file():
            z.write(f, arcname=name)
    test_eval_dir = PAPER_EXPORT_DIR / 'test_evaluation'
    if test_eval_dir.is_dir():
        for p in test_eval_dir.rglob('*'):
            if p.is_file():
                z.write(p, arcname=f"test_evaluation/{p.relative_to(test_eval_dir)}")

print("\n" + "=" * 60)
print("=== HELD-OUT TEST METRICS (PAPER-SPECIFIC) ===")
print("=" * 60)
print(json.dumps(paper_meta['held_out_test_metrics'], indent=2))
print("=" * 60)

# Download both bundles to local machine
baseline_bundle = BASE_DIR / 'yolo11s_opensar_artifacts.zip'
if files:
    if baseline_bundle.is_file():
        print("Downloading baseline artifacts ZIP: yolo11s_opensar_artifacts.zip ...")
        files.download(str(baseline_bundle))
    print("Downloading paper artifacts ZIP: yolo11s_opensar_paper_artifacts.zip ...")
    files.download(str(paper_bundle))
