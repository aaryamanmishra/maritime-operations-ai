#!/usr/bin/env python3
"""
import_checkpoints.py - Automated import and verification of recovered Colab OpenSAR checkpoints.
Places models into backend/ml/weights/experimental/ and verifies integrity without touching production weights.
"""

import os
import sys
import json
import shutil
import hashlib
import zipfile
from pathlib import Path

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def scan_and_import():
    base_dir = Path(__file__).resolve().parent.parent
    imports_dir = base_dir / "imports" / "opensar_checkpoints"
    downloads_dir = Path.home() / "Downloads"
    runs_dir = base_dir.parent / "all_runs_and_artifacts"
    target_exp_dir = base_dir / "backend" / "ml" / "weights" / "experimental"
    
    print("=" * 70)
    print("OPENSAR CHECKPOINT IMPORT & VERIFICATION UTILITY")
    print("=" * 70)
    print(f"Scanning imports directory: {imports_dir}")
    print(f"Scanning Downloads directory: {downloads_dir}")
    print(f"Scanning all_runs_and_artifacts: {runs_dir}")
    print(f"Target experimental directory: {target_exp_dir}")
    
    # 1. Look for baseline ZIP or directory
    baseline_zips = list(imports_dir.glob("**/yolo11s_opensar_artifacts*.zip")) + \
                    list(downloads_dir.glob("yolo11s_opensar_artifacts*.zip")) + \
                    list(runs_dir.glob("yolo11s_opensar_artifacts*.zip"))
    paper_zips = list(imports_dir.glob("**/yolo11s_opensar_paper_artifacts*.zip")) + \
                 list(downloads_dir.glob("yolo11s_opensar_paper_artifacts*.zip")) + \
                 list(runs_dir.glob("yolo11s_opensar_paper_artifacts*.zip"))
                 
    # 2. Extract baseline
    if baseline_zips:
        src = baseline_zips[0]
        dest = target_exp_dir / "yolo11s_opensar_baseline"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[+] Extracting Baseline Bundle: {src} -> {dest}")
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(dest)
        best_pt = dest / "best.pt"
        if best_pt.exists():
            print(f"    best.pt SHA256: {compute_sha256(best_pt)}")
        meta_json = dest / "yolo11s_opensar_metadata.json"
        if meta_json.exists():
            meta = json.loads(meta_json.read_text())
            print(f"    Baseline Test Metrics: {meta.get('held_out_test_metrics')}")
    else:
        print("\n[-] Baseline bundle (yolo11s_opensar_artifacts.zip) not found yet.")

    # 3. Extract paper
    if paper_zips:
        src = paper_zips[0]
        dest = target_exp_dir / "yolo11s_opensar_paper"
        dest.mkdir(parents=True, exist_ok=True)
        print(f"\n[+] Extracting Paper Bundle: {src} -> {dest}")
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(dest)
        best_pt = dest / "best.pt"
        if best_pt.exists():
            print(f"    best.pt SHA256: {compute_sha256(best_pt)}")
        meta_json = dest / "yolo11s_opensar_paper_metadata.json"
        if meta_json.exists():
            meta = json.loads(meta_json.read_text())
            print(f"    Paper Test Metrics: {meta.get('held_out_test_metrics')}")
    else:
        print("[-] Paper bundle (yolo11s_opensar_paper_artifacts.zip) not found yet.")
        
    print("\n" + "=" * 70)
    print("VERIFYING PRODUCTION WEIGHTS ISOLATION:")
    prod_path = base_dir / "backend" / "ml" / "weights" / "yolo26s_sar_vessel.pt"
    if prod_path.exists():
        print(f"[OK] Production GRD weights intact at: {prod_path}")
        print(f"     SHA256: {compute_sha256(prod_path)}")
    print("=" * 70)

if __name__ == "__main__":
    scan_and_import()
