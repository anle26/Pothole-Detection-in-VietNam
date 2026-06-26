#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 YOLOv8s Training Pipeline - Domain Generalization (Pre-training)
This script is optimized for RTX 3060 (12GB VRAM) and 64GB RAM on Windows.
It handles automatic path updates, model training, validation, and ONNX export.
"""

import os
import sys
import torch
import yaml
from ultralytics import YOLO

# Disable Weights & Biases (WandB) to avoid interactive login prompts
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
    
    dataset_yaml = None
    for path in yaml_candidates:
        if os.path.exists(path):
            dataset_yaml = path
            break
            
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
    
    # 1. Resolve Project Root and Data Configuration
    # Assumes script is run from project root, fallback to parent of notebooks folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    if os.path.basename(project_root) == "notebooks":
        project_root = os.path.dirname(project_root)
        
    dataset_yaml = get_and_update_dataset_yaml(project_root)
    
    # 2. Check for Training Checkpoint (Auto-Resume Capability)
    # Using relative paths since settings.json handles runs_dir relative or absolute
    checkpoint_candidates = [
        os.path.join(project_root, "runs", "detect", "runs", "detect", "base_model", "weights", "last.pt"),
        os.path.join(project_root, "runs", "detect", "base_model", "weights", "last.pt"),
        os.path.join(project_root, "runs", "detect", "train", "weights", "last.pt"),
    ]
    
    last_checkpoint = None
    for cp_path in checkpoint_candidates:
        if os.path.exists(cp_path):
            last_checkpoint = cp_path
            break
            
    # 3. Model Initialization & Training
    if last_checkpoint:
        print(f"\n🔄 [RESUME] Found last checkpoint at: {last_checkpoint}")
        print("⏳ Resuming training from last saved epoch...")
        model = YOLO(last_checkpoint)
        results = model.train(resume=True, workers=0, batch=8, amp=False)
    else:
        print("\n🆕 [NEW RUN] No checkpoint found. Loading pre-trained yolov8s.pt...")
        model = YOLO("yolov8s.pt")
        
        print("🔥 Launching hardware-optimized training...")
        results = model.train(
            # --- Paths and Output ---
            data=dataset_yaml,
            project="runs/detect",
            name="base_model",
            exist_ok=True,
            
            # --- Hardware Optimization (RTX 3060 12GB + 64GB RAM) ---
            cache=False,            # Disable RAM caching when using workers > 0 to avoid Windows multiprocessing pickle MemoryError (since SSD is fast enough)
            workers=0,              # FIXED: Must be 0 on Windows to avoid "CUDA error: unspecified launch failure"
            batch=16,               # TĂNG LẠI LÊN 16: GPU của bạn rất mát (44 độ), Batch 16 sẽ giúp model học mượt hơn (gradient ổn định hơn).
            imgsz=640,              # Standard 640x640 input resolution
            device=0,               # Run on GPU index 0
            amp=False,              # FIXED: Disabled AMP (Mixed Precision) to prevent fatal CUDA crash on Windows during make_anchors
            
            # --- Convergence Parameters ---
            epochs=100,             # Train for 100 epochs
            patience=20,            # Early stopping if mAP@50-95 doesn't improve for 20 epochs
            optimizer="auto",       # YOLO auto-selects optimizer (typically AdamW)
            
            # --- Augmentation (Domain Generalization) ---
            mosaic=0.5,             # GIẢM XUỐNG 0.5: Ổ gà là vật thể nhỏ, cắt ghép quá nhiều (1.0) sẽ làm mất bối cảnh mặt đường.
            mixup=0.0,              # TẮT (0.0): Chồng ảnh (Mixup) làm ổ gà bị mờ, model rất khó nhận diện.
            close_mosaic=15,        # Turn off Mosaic in the last 15 epochs to fine-tune bounding boxes
        )
    print("\n🎉 Training run completed successfully!")
    
    # 4. Objective Metrics Evaluation (Validation)
    print("\n" + "=" * 60)
    print("📊 STEP 2: RUNNING VALIDATION ON OBJECTIVE METRICS")
    print("=" * 60)
    
    # Locate the best weights
    best_weight_candidates = [
        os.path.join(project_root, "runs", "detect", "runs", "detect", "base_model", "weights", "best.pt"),
        os.path.join(project_root, "runs", "detect", "base_model", "weights", "best.pt"),
    ]
    best_model_path = None
    for path in best_weight_candidates:
        if os.path.exists(path):
            best_model_path = path
            break
            
    if not best_model_path:
        print("❌ Could not find best.pt file. Skipping validation step.")
        return
        
    print(f"🎯 Loading best model from: {best_model_path}")
    model_eval = YOLO(best_model_path)
    
    val_results = model_eval.val(
        data=dataset_yaml,
        device=0,
        batch=32,
        imgsz=640
    )
    
    metrics = val_results.results_dict
    map50 = metrics.get('metrics/mAP50(B)', 0.0)
    map50_95 = metrics.get('metrics/mAP50-95(B)', 0.0)
    
    print("\n" + "=" * 60)
    print("🏆 VALIDATION RESULTS (VAL METRICS COCO):")
    print(f"📈 mAP@50 (IoU=0.50)      : {map50:.4f} ({map50 * 100:.2f}%)")
    print(f"📈 mAP@50-95 (IoU=0.50:0.95): {map50_95:.4f} ({map50_95 * 100:.2f}%)")
    print("=" * 60)
    
    # 5. Export to ONNX format
    print("\n" + "=" * 60)
    print("📦 STEP 3: EXPORTING MODEL TO ONNX FORMAT")
    print("=" * 60)
    
    print("📦 Converting model to ONNX...")
    onnx_path = model_eval.export(
        format="onnx",
        dynamic=True,       # Dynamic batch & image sizes
        simplify=True       # Simplify ONNX execution graph
    )
    print("\n" + "=" * 60)
    print("🎉 MODEL EXPORTED SUCCESSFULLY!")
    print(f"💾 ONNX File Path: {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
