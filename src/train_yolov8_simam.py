#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 YOLOv8s + SimAM Training Pipeline
This script integrates the Parameter-Free Spatial Attention Module (SimAM) into YOLOv8.
It uses dynamic monkey-patching so you DO NOT need to modify ultralytics source code
or deal with complex custom YAML index shifting. Pretrained weights load perfectly!

Hardware profile: RTX 3060 (12GB VRAM) & Ryzen 7 5000 series (Windows)
"""

import os
import sys
import time
import yaml
import torch
import torch.nn as nn

# Disable WandB to prevent login blocks
os.environ["WANDB_DISABLED"] = "true"

from ultralytics import YOLO
import ultralytics.nn.modules.block as block
import ultralytics.nn.tasks as tasks

# ==============================================================================
# 1. DEFINE SimAM AND CUSTOM C2f BLOCK
# ==============================================================================

class SimAM(nn.Module):
    """
    SimAM: Simple Parameter-Free Attention Module
    https://proceedings.mlr.press/v139/yang21o.html
    """
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        
        # Spatial energy calculation
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * self.activation(y)

# Keep a reference to the original C2f just in case
OriginalC2f = block.C2f

class C2f_SimAM(OriginalC2f):
    """
    Custom C2f block that passes its output through the SimAM module.
    Because SimAM has no parameters, this block is perfectly compatible
    with standard YOLOv8 pretrained weights!
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.simam = SimAM()
        
    def forward(self, x):
        # Run standard C2f extraction
        out = super().forward(x)
        # Apply SimAM attention
        return self.simam(out)

# 💥 MONKEY-PATCHING: Replace standard C2f with our C2f_SimAM dynamically
block.C2f = C2f_SimAM
tasks.C2f = C2f_SimAM

# ==============================================================================
# 2. UTILITY FUNCTIONS
# ==============================================================================

def get_and_update_dataset_yaml(project_root):
    yaml_candidates = [
        os.path.join(project_root, "data", "processed", "combined_pothole", "dataset.yaml"),
        os.path.join(project_root, "data", "processed", "trial_dataset", "dataset.yaml"),
    ]
    dataset_yaml = next((p for p in yaml_candidates if os.path.exists(p)), None)
            
    if not dataset_yaml:
        raise FileNotFoundError("❌ Could not find dataset.yaml!")
        
    # Dynamically update the root path in dataset.yaml to be absolute
    with open(dataset_yaml, 'r', encoding='utf-8') as f:
        yaml_content = yaml.safe_load(f)
        
    abs_dataset_dir = os.path.abspath(os.path.dirname(dataset_yaml))
    yaml_content['path'] = abs_dataset_dir.replace('\\', '/')
    
    with open(dataset_yaml, 'w', encoding='utf-8') as f:
        yaml.safe_dump(yaml_content, f, allow_unicode=True)
        
    return dataset_yaml

# ==============================================================================
# 3. MAIN TRAINING LOOP
# ==============================================================================

def main():
    print("=" * 60)
    print("🚀 INITIALIZING YOLOv8s + SimAM TRAINING PIPELINE")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir) if os.path.basename(current_dir) in ("src", "notebooks") else current_dir
    dataset_yaml = get_and_update_dataset_yaml(project_root)
    
    # ------------------------------------------------------------------
    # OUTPUT PATHS — separate from baseline YOLOv8 (base_model)
    # Baseline saves to: runs/detect/base_model/
    # SimAM   saves to: runs/detect/yolov8s_simam/
    # ------------------------------------------------------------------
    RUN_PROJECT = os.path.join(project_root, "runs", "detect")
    RUN_NAME = "yolov8s_simam"
    RUN_DIR = os.path.join(RUN_PROJECT, RUN_NAME)
    
    print(f"\n📁 Output directory : {RUN_DIR}")
    print(f"   (Baseline lives in: {os.path.join(RUN_PROJECT, 'base_model')} — NOT overwritten)")
    
    # 💡 When YOLO("yolov8s.pt") runs, it uses tasks.parse_model().
    # Because we patched tasks.C2f, it builds C2f_SimAM blocks automatically!
    # Then it safely maps all Conv weights since SimAM has 0 parameters.
    print("\n📦 Loading Base Model & Injecting SimAM into C2f blocks...")
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
            cache=False,
            workers=4,          # __main__ guard present → safe on Windows; lower to 2 if errors
            batch=16,           # Safe for 12GB VRAM
            imgsz=640,
            device=0,
            amp=True,           # Mixed precision → faster + less VRAM; disable if NaN loss

            # --- Safety: Checkpoint & Early Stopping ---
            save=True,          # Save best.pt & last.pt (default, explicit for clarity)
            save_period=10,     # Save checkpoint every 10 epochs (epoch10.pt, epoch20.pt, ...)
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
    # 4. POST-TRAINING EVALUATION — F1-SCORE & METRICS
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
        # 5. MODEL EFFICIENCY METRICS — Parameters, GFLOPs, Latency
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
        print("📈 FINAL EVALUATION RESULTS")
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
