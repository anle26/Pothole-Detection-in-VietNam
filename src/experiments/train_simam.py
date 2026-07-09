#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 YOLOv8s + SimAM Training Pipeline
This script integrates the Parameter-Free Spatial Attention Module (SimAM) into YOLOv8.
It uses dynamic monkey-patching so you DO NOT need to modify ultralytics source code.
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

# Import SimAM modules and monkey patch
from src.models.simam_module import apply_simam_patch, SimAM, C2f_SimAM

# 💥 MONKEY-PATCHING: Replace standard C2f with our C2f_SimAM dynamically
apply_simam_patch()

# Ensure they are available in __main__ for torch.load compatibility
sys.modules['__main__'].C2f_SimAM = C2f_SimAM
sys.modules['__main__'].SimAM = SimAM

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
    print("🚀 INITIALIZING YOLOv8s + SimAM TRAINING PIPELINE")
    print("=" * 60)
    
    # 1. Resolve Project Root and Data Configuration
    dataset_yaml = get_and_update_dataset_yaml(str(PROJECT_ROOT))
    
    # ------------------------------------------------------------------
    # OUTPUT PATHS
    # Baseline saves to: runs/detect/yolov8s_baseline/
    # SimAM   saves to: runs/detect/yolov8s_simam/
    # ------------------------------------------------------------------
    RUN_PROJECT = os.path.join(str(PROJECT_ROOT), "runs", "detect")
    RUN_NAME = "yolov8s_simam"
    RUN_DIR = os.path.join(RUN_PROJECT, RUN_NAME)
    
    print(f"\n📁 Output directory : {RUN_DIR}")
    print(f"   (Baseline lives in : {os.path.join(RUN_PROJECT, 'yolov8s_baseline')} — NOT overwritten)")
    
    # 2. Load Model
    print("\n📦 Loading Pre-trained YOLOv8s and applying SimAM patch...")
    pretrained_weights = os.path.join(str(PROJECT_ROOT), "models", "pre_trained", "yolov8s.pt")
    if not os.path.exists(pretrained_weights):
        print("⚠️ No weights in models/pre_trained/, fallback to yolov8s.pt")
        pretrained_weights = "yolov8s.pt"
        
    model = YOLO(pretrained_weights)
    
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
            amp=True,           # Mixed precision → faster + less VRAM

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
            hsv_h=0.015,        # Hue shift — lighting variation
            hsv_s=0.7,          # Saturation — wet vs dry road
            hsv_v=0.4,          # Brightness — day vs night
            degrees=5.0,        # Slight rotation — tilted camera
            translate=0.1,      # Position shift
            scale=0.5,          # Multi-scale potholes
            fliplr=0.5,         # Horizontal flip
            flipud=0.0,         # No vertical flip — potholes are always below
            mosaic=1.0,         # Full mosaic for dense scenes
            mixup=0.1,          # Light mixup for regularization
            copy_paste=0.1,     # Paste potholes onto other road surfaces
            erasing=0.2,        # Random erasing — occlusion robustness
            close_mosaic=15,    # Disable mosaic for last 15 epochs (fine-tune)
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
            plots=True,        # Generates P-R curve, F1-curve, confusion matrix
            save_json=False,
        )

        # Copy validation curves and results to reports/
        import shutil
        reports_fig_dir = os.path.join(str(PROJECT_ROOT), "reports", "figures")
        reports_metric_dir = os.path.join(str(PROJECT_ROOT), "reports", "metrics")
        os.makedirs(reports_fig_dir, exist_ok=True)
        os.makedirs(reports_metric_dir, exist_ok=True)
        
        src_dst_mappings = [
            ("BoxF1_curve.png", os.path.join(reports_fig_dir, "simam_F1_curve.png")),
            ("BoxPR_curve.png", os.path.join(reports_fig_dir, "simam_PR_curve.png")),
            ("confusion_matrix.png", os.path.join(reports_fig_dir, "simam_confusion_matrix.png")),
            ("results.png", os.path.join(reports_fig_dir, "simam_results_curve.png")),
            ("results.csv", os.path.join(reports_metric_dir, "simam_results.csv")),
        ]
        for src_name, dst_path in src_dst_mappings:
            src_path = os.path.join(RUN_DIR, src_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
                print(f"📋 Copied simam {src_name} -> {dst_path}")

        # Extract key metrics
        precision = metrics.box.mp       # Mean Precision
        recall = metrics.box.mr          # Mean Recall
        map50 = metrics.box.map50        # mAP@0.5
        map50_95 = metrics.box.map       # mAP@0.5:0.95

        # Compute F1-Score from Precision & Recall
        if (precision + recall) > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # ==============================================================
        # 4. MODEL EFFICIENCY METRICS — Parameters, GFLOPs, Latency
        # ==============================================================
        print("\n" + "=" * 60)
        print("⚙️  PROFILING MODEL EFFICIENCY")
        print("=" * 60)

        # --- Parameter Count ---
        total_params = sum(p.numel() for p in eval_model.model.parameters())
        trainable_params = sum(p.numel() for p in eval_model.model.parameters() if p.requires_grad)

        # --- GFLOPs ---
        try:
            model_info = eval_model.info(detailed=False, verbose=False)
            if isinstance(model_info, (list, tuple)) and len(model_info) >= 4:
                gflops = model_info[3]
            else:
                gflops = None
        except Exception:
            gflops = None

        # Fallback: compute GFLOPs manually via thop if available
        if gflops is None:
            try:
                from thop import profile as thop_profile
                dummy_input = torch.randn(1, 3, 640, 640).to(next(eval_model.model.parameters()).device)
                flops, _ = thop_profile(eval_model.model, inputs=(dummy_input,), verbose=False)
                gflops = flops / 1e9
            except ImportError:
                gflops = -1  # thop not installed

        # --- Inference Latency Benchmark ---
        print("  ⏱️  Benchmarking inference speed (GPU)...")
        device = next(eval_model.model.parameters()).device
        eval_model.model.eval()
        dummy = torch.randn(1, 3, 640, 640, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(100):
                _ = eval_model.model(dummy)

        # Timed runs
        num_runs = 300
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = eval_model.model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.perf_counter()

        avg_ms = (t_end - t_start) / num_runs * 1000  # ms per image
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0

        # ==============================================================
        # PRINT ALL RESULTS
        # ==============================================================
        print("\n" + "=" * 60)
        print("📈 FINAL EVALUATION RESULTS — YOLOv8s + SimAM")
        print("=" * 60)

        print("\n  --- Accuracy Metrics ---")
        print(f"  Precision       : {precision:.4f}")
        print(f"  Recall          : {recall:.4f}")
        print(f"  ⭐ F1-Score      : {f1_score:.4f}")
        print(f"  mAP@0.5         : {map50:.4f}")
        print(f"  mAP@0.5:0.95    : {map50_95:.4f}")

        print("\n  --- Efficiency Metrics ---")
        print(f"  Parameters      : {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  Trainable       : {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        if gflops and gflops > 0:
            print(f"  GFLOPs          : {gflops:.2f}")
        else:
            print(f"  GFLOPs          : N/A (install 'thop' for manual calc)")
        print(f"  Inference Time  : {avg_ms:.2f} ms/image")
        print(f"  FPS             : {fps:.1f}")

        print("=" * 60)
    else:
        print(f"⚠️ Best weights not found at {best_weights}. Skipping evaluation.")

if __name__ == "__main__":
    main()
