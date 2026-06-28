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
    project_root = current_dir if os.path.basename(current_dir) != "notebooks" else os.path.dirname(current_dir)
    dataset_yaml = get_and_update_dataset_yaml(project_root)
    
    # 💡 When YOLO("yolov8s.pt") runs, it uses tasks.parse_model().
    # Because we patched tasks.C2f, it builds C2f_SimAM blocks automatically!
    # Then it safely maps all Conv weights since SimAM has 0 parameters.
    print("\n📦 Loading Base Model & Injecting SimAM into C2f blocks...")
    model = YOLO("yolov8s.pt")
    
    print("🔥 Starting Training (Hardware configs synced for RTX 3060 & Windows)...")
    results = model.train(
        data=dataset_yaml,
        project="runs/detect",
        name="simam_pretrain",
        exist_ok=True,
        
        # --- Hardware Optimizations (From your configs) ---
        cache=False,
        workers=0,      # Avoid CUDA launch failure on Windows
        batch=16,       # Safe for 12GB VRAM
        imgsz=640,
        device=0,
        amp=False,      # Avoid CUDA crashes during make_anchors on some Windows setups
        
        # --- Training Hyperparams ---
        epochs=100,
        patience=20,
        optimizer="auto",
        
        # --- Augmentation for Domain Generalization ---
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=15,
    )
    
    print("\n🎉 Training run completed successfully!")

if __name__ == "__main__":
    main()
