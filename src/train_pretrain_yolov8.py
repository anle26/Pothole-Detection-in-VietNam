#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 YOLOv8s BASELINE Training Pipeline
This script trains a standard YOLOv8s model WITHOUT any attention module.
It serves as the BASELINE for fair comparison against YOLOv8s + SimAM.

ALL hyperparameters (augmentation, LR, batch, epochs, etc.) are IDENTICAL
to train_yolov8_simam.py to ensure a fair comparison.

Hardware profile: RTX 3060 (12GB VRAM) & Ryzen 7 5000 series (Windows)
"""

import os
import sys
import time
import yaml
import torch
from ultralytics import YOLO

# Disable WandB to prevent login blocks
os.environ["WANDB_DISABLED"] = "true"


# ==============================================================================
# 1. UTILITY FUNCTIONS
# ==============================================================================

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


# ==============================================================================
# 2. MAIN TRAINING LOOP
# ==============================================================================

def main():
    # 0. Print System Details
    print_system_info()
    
    print("=" * 60)
    print("🚀 INITIALIZING YOLOv8s BASELINE TRAINING PIPELINE")
    print("   (No attention module — for fair comparison with SimAM)")
    print("=" * 60)
    
    # 1. Resolve Project Root and Data Configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) in ("src", "notebooks") else current_dir
    dataset_yaml = get_and_update_dataset_yaml(project_root)
    
    # ------------------------------------------------------------------
    # OUTPUT PATHS — separate from SimAM results
    # Baseline saves to: runs/detect/yolov8s_baseline/
    # SimAM   saves to: runs/detect/yolov8s_simam/
    # ------------------------------------------------------------------
    RUN_PROJECT = os.path.join(project_root, "runs", "detect")
    RUN_NAME = "yolov8s_baseline"
    RUN_DIR = os.path.join(RUN_PROJECT, RUN_NAME)
    
    print(f"\n📁 Output directory : {RUN_DIR}")
    print(f"   (SimAM lives in : {os.path.join(RUN_PROJECT, 'yolov8s_simam')} — NOT overwritten)")
    
    # 2. Load Model
    print("\n📦 Loading Pre-trained YOLOv8s (standard, no attention)...")
    model = YOLO("yolov8s.pt")
    
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
        print("🔥 Starting Training (Hardware configs synced for RTX 3060 & Windows)...")
        results = model.train(
            data=dataset_yaml,
            project=RUN_PROJECT,
            name=RUN_NAME,
            exist_ok=True,
            
            # --- Hardware Optimizations ---
            # ⚠️ IDENTICAL to SimAM version for fair comparison
            cache='ram',
            workers=0,          # Must be 0 on Windows to avoid CUDA launch failure with multiprocessing
            batch=16,           # Safe for 12GB VRAM
            imgsz=640,
            device=0,
            amp=True,           # Mixed precision → faster + less VRAM; disable if NaN loss

            # --- Safety: Checkpoint & Early Stopping ---
            save=True,          # Save best.pt & last.pt
            save_period=10,     # Save checkpoint every 10 epochs
            epochs=100,
            patience=20,        # Early stopping: stop if no improvement for 20 epochs

            # --- Training Hyperparams ---
            # ⚠️ IDENTICAL to SimAM version for fair comparison
            optimizer="auto",
            lr0=0.01,           # Initial learning rate
            lrf=0.01,           # Final LR = lr0 × lrf (cosine schedule)
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,

            # --- Augmentation for Domain Generalization ---
            # ⚠️ IDENTICAL to SimAM version for fair comparison
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

        # --- GFLOPs (via ultralytics built-in) ---
        try:
            model_info = eval_model.info(detailed=False, verbose=False)
            # model.info() returns (layers, params, gradients, gflops)
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

        # Warmup (discard first runs for GPU JIT compilation)
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
        print("📈 FINAL EVALUATION RESULTS — YOLOv8s BASELINE")
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
