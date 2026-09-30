# ADR-004: Strict Separation of ML Training from Production Inference

## Status
Accepted

## Context
Machine learning workflows for satellite radar vessel detection (SAR) and vessel fuel/performance surrogate models require heavyweight frameworks (PyTorch, torchvision, ultralytics), CUDA-enabled GPUs, large training datasets (multi-gigabyte SAR image chips), and long experimentation cycles.

Running training pipelines inside the application backend or coupling training dependencies to the web container causes bloated Docker images (exceeding 8–10 GB), slow deployment cycles, GPU resource competition with web serving, and environment instability.

## Decision
We enforce a strict separation between ML Training and Production Inference:
1. **Repository Structure**:
   - `ml/training/`: Houses standalone, reproducible PyTorch training scripts, Kaggle/Colab notebooks, dataset download utilities, data augmentation pipelines, and hyperparameter tuning configs.
   - `ml/inference/` and `backend/app/domain/ml/`: Houses lightweight inference engines, export scripts, and evaluation metrics.
2. **Standardized Artifact Format**: All trained models are exported to **ONNX (Open Neural Network Exchange)** format or optimized TorchScript weights, accompanied by an explicit `model_card.json` detailing training dataset provenance, validation IoU/mAP, resolution limitations, and input normalization requirements.
3. **Production Inference Runtime**: The production backend uses `onnxruntime` (or a dedicated lightweight PyTorch CPU/CUDA runner) with zero dependencies on training frameworks, Jupyter, or training datasets.
4. **Zero-Cost Training Infrastructure**: Model training is structured to execute on free cloud GPU environments (Kaggle Kernels with free 30h/week T4/P100 GPUs or Google Colab) with automated model artifact export to Hugging Face Hub or GitHub Releases.

## Consequences
### Positive
- Production backend Docker container remains lean, fast-starting, and reproducible.
- Model training can be run independently by any student on Kaggle without requiring expensive local GPUs.
- Models are versioned, audited, and benchmarked before deployment.

### Negative / Trade-offs
- Model updates require an explicit export and testing step before integration into the backend service.
