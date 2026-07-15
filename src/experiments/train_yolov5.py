#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 YOLOv5s Training Pipeline
This script trains YOLOv5 for pothole detection as a comparison baseline.
No monkey-patching needed for standard YOLO architectures.
"""

import os
import sys
import time
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable WandB to prevent login blocks
os.environ["WANDB_DISABLED"] = "true"

def print_system_info():
    """Prints hardware and environment information."""
    print("=" * 60)
    print("🖥️  SYSTEM ENVIRONMENT INFORMATION")
    print("=" * 60)
    print(f"🐍 Python Version: {sys.version}")
    print(f"🔥 PyTorch Version: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"🤖 CUDA GPU Available: {cuda_available}")
    if cuda_available:
        print(f"✅ Active GPU: {torch.cuda.get_device_name(0)}")
        print(f"⚙️ CUDA Version (PyTorch): {torch.version.cuda}")
    else:
        print("❌ WARNING: CUDA is not available. Training will run on CPU!")
    print("=" * 60 + "\n")

def get_and_update_dataset_yaml(project_root):
    """Finds and dynamically updates dataset.yaml path on this machine."""
    yaml_candidates = [
        os.path.join(project_root, "data", "processed", "combined_pothole", "dataset.yaml"),
        os.path.join(project_root, "data", "processed", "trial_dataset", "dataset.yaml"),
    ]
    
    dataset_yaml = next((p for p in yaml_candidates if os.path.exists(p)), None)
            
    if not dataset_yaml:
        raise FileNotFoundError("❌ Could not find dataset.yaml. Please check your data directories!")
        
    print(f"✅ Found dataset configuration at: {os.path.abspath(dataset_yaml)}")
    
    # Dynamically update the root path in dataset.yaml to be absolute
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
        
    abs_dataset_dir = os.path.abspath(os.path.dirname(dataset_yaml))
    yaml_content['path'] = abs_dataset_dir.replace('\\', '/')
    
    with open(dataset_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_content, f, allow_unicode=True)
        
    print(f"🔄 Automatically updated 'path' in dataset.yaml to: {yaml_content['path']}")
    return dataset_yaml

def main():
    # 0. Print System Details
    print_system_info()
    
    print("=" * 60)
    print("🚀 INITIALIZING YOLOv5s TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Resolve Project Root and Data Configuration
    dataset_yaml = get_and_update_dataset_yaml(str(PROJECT_ROOT))
    
    # ------------------------------------------------------------------
    # OUTPUT PATHS
    # ------------------------------------------------------------------
    RUN_PROJECT = os.path.join(str(PROJECT_ROOT), "runs", "detect")
    RUN_NAME = "yolov5s_baseline"
    RUN_DIR = os.path.join(RUN_PROJECT, RUN_NAME)
    
    print(f"\n📁 Output directory : {RUN_DIR}")
    
    # 2. Load Model
    print("\n📦 Loading Pre-trained YOLOv5s...")
    model = YOLO('yolov5su.pt')
    
    # ------------------------------------------------------------------
    # CHECK FOR EXISTING CHECKPOINT → AUTO-RESUME
    # ------------------------------------------------------------------
    resume_weights = os.path.join(RUN_DIR, "weights", "last.pt")
    if os.path.exists(resume_weights):
        print(f"\n🔄 Found existing checkpoint: {resume_weights}")
        print("   Resuming training from last saved state...")
        model = YOLO(resume_weights)
        results = model.train(resume=True)
    else:
        print("🔥 Starting Training (Hardware configs synced)...")
        results = model.train(
            data=dataset_yaml,
            project=RUN_PROJECT,
            name=RUN_NAME,
            exist_ok=True,
            
            # --- Hardware Optimizations ---
            cache='ram',
            workers=0,          # Must be 0 on Windows to avoid CUDA launch failure
            batch=16,           # Safe for 12GB VRAM
            imgsz=640,
            device=0,
            amp=False,           # Disabled AMP to avoid CUDA unspecified launch failure on Windows

            # --- Safety: Checkpoint & Early Stopping ---
            save=True,          # Save best.pt & last.pt
            save_period=10,     # Save checkpoint every 10 epochs
            epochs=100,
            patience=20,        # Early stopping: stop if no improvement for 20 epochs

            # --- Training Hyperparams ---
            optimizer="auto",
            lr0=0.01,           # Initial learning rate
            lrf=0.01,           # Final LR = lr0 × lrf (cosine schedule)
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,

            # --- Augmentation for Domain Generalization ---
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            flipud=0.0,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
            erasing=0.2,
            close_mosaic=15,
        )

    print("\n🎉 Training run completed successfully!")

    # ==================================================================
    # 3. POST-TRAINING EVALUATION — F1-SCORE & METRICS
    # ==================================================================
    print("\n" + "=" * 60)
    print("📊 RUNNING POST-TRAINING EVALUATION (F1-Score & Metrics)")
    print("=" * 60)

    # Load the best checkpoint from the training run
    best_weights = os.path.join(RUN_DIR, "weights", "best.pt")
    if os.path.exists(best_weights):
        eval_model = YOLO(best_weights)
        metrics = eval_model.val(
            data=dataset_yaml,
            imgsz=640,
            batch=16,
            device=0,
            plots=True,
            save_json=False,
        )

        # Copy validation curves and results to reports/
        import shutil
        reports_fig_dir = os.path.join(str(PROJECT_ROOT), "reports", "figures")
        reports_metric_dir = os.path.join(str(PROJECT_ROOT), "reports", "metrics")
        os.makedirs(reports_fig_dir, exist_ok=True)
        os.makedirs(reports_metric_dir, exist_ok=True)
        
        src_dst_mappings = [
            ("BoxF1_curve.png", os.path.join(reports_fig_dir, "yolov5s_F1_curve.png")),
            ("BoxPR_curve.png", os.path.join(reports_fig_dir, "yolov5s_PR_curve.png")),
            ("confusion_matrix.png", os.path.join(reports_fig_dir, "yolov5s_confusion_matrix.png")),
            ("results.png", os.path.join(reports_fig_dir, "yolov5s_results_curve.png")),
            ("results.csv", os.path.join(reports_metric_dir, "yolov5s_results.csv")),
        ]
        for src_name, dst_path in src_dst_mappings:
            src_path = os.path.join(RUN_DIR, src_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"📋 Copied yolov5s {src_name} -> {dst_path}")

if __name__ == "__main__":
    main()
