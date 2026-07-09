#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 Fine-Tuning Pipeline: YOLOv8 + SimAM on Vietnam Dataset
This script evaluates and fine-tunes the SIMAM proposed model on the Vietnam Dataset.
It runs Zero-Shot, Few-Shot, and 5-Fold LP-FT.
Uses Subprocess to guarantee 100% clean CUDA memory for every single training run.
"""

import os
import sys
import argparse
import subprocess
import torch
import json
from ultralytics import YOLO
from pathlib import Path

# Fix Windows console emoji printing error
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Disable WandB to prevent login blocks
os.environ["WANDB_DISABLED"] = "true"
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Import SimAM modules and monkey patch
from src.models.simam_module import apply_simam_patch, SimAM, C2f_SimAM

# Ensure they are available in __main__ for torch.load compatibility
sys.modules['__main__'].C2f_SimAM = C2f_SimAM
sys.modules['__main__'].SimAM = SimAM

# ==============================================================================
# 1. WORKER FUNCTIONS (Run in isolated processes)
# ==============================================================================
def worker_zero_shot(model_path, data_yaml, run_dir):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        project=os.path.dirname(run_dir),
        name=os.path.basename(run_dir),
        exist_ok=True,
        split='val',
        imgsz=640,
        device=DEVICE
    )
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({
            "map": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr)
        }, f)

def worker_lp_ft(model_path, data_yaml, run_dir, epochs_lp, epochs_ft):
    model = YOLO(model_path)
    
    # STAGE 1: Linear Probing
    lp_dir = os.path.join(run_dir, "lp")
    model.train(
        data=data_yaml,
        project=os.path.dirname(lp_dir),
        name=os.path.basename(lp_dir),
        exist_ok=True,
        epochs=epochs_lp,
        freeze=10,
        optimizer="SGD",
        lr0=0.005,
        imgsz=640,
        workers=0,
        batch=8,
        device=DEVICE,
        amp=False,
        save=True,
        patience=10
    )
    
    lp_best = os.path.join(lp_dir, "weights", "best.pt")
    if not os.path.exists(lp_best):
        print("⚠️ LP failed. Exiting worker.")
        return
        
    # Free memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # STAGE 2: Fine-Tuning
    model_ft = YOLO(lp_best)
    ft_dir = os.path.join(run_dir, "ft")
    model_ft.train(
        data=data_yaml,
        project=os.path.dirname(ft_dir),
        name=os.path.basename(ft_dir),
        exist_ok=True,
        epochs=epochs_ft,
        freeze=0,
        optimizer="SGD",
        lr0=0.001,
        imgsz=640,
        workers=0,
        batch=8,
        device=DEVICE,
        amp=False,
        save=True,
        patience=15
    )
    
    # Evaluate
    ft_best = os.path.join(ft_dir, "weights", "best.pt")
    if os.path.exists(ft_best):
        eval_model = YOLO(ft_best)
        metrics = eval_model.val(data=data_yaml, exist_ok=True, split='val', imgsz=640, device=DEVICE)
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump({
                "map": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr)
            }, f)

# ==============================================================================
# 2. MASTER FUNCTIONS (Orchestrates Subprocesses)
# ==============================================================================
def spawn_worker(worker_task, model_path, data_yaml, run_dir, epochs_lp=0, epochs_ft=0):
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--mode", "worker",
        "--worker_task", worker_task,
        "--model_path", model_path,
        "--data_yaml", data_yaml,
        "--run_dir", run_dir,
        "--epochs_lp", str(epochs_lp),
        "--epochs_ft", str(epochs_ft)
    ]
    # Check if this task was already completed in a previous run
    score_file = os.path.join(run_dir, "metrics.json")
    if os.path.exists(score_file):
        print(f"✅ Skipping completed task (Found existing metrics): {run_dir}")
        with open(score_file, "r") as f:
            return json.load(f)

    print(f"\n🚀 Spawning isolated process for: {run_dir}")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            subprocess.run(cmd, check=True)
            break # Success
        except subprocess.CalledProcessError as e:
            print(f"❌ [Attempt {attempt}/{max_retries}] Subprocess crashed with error: {e}")
            if attempt == max_retries:
                print("💥 Maximum retries reached. Moving to next task (Score will be 0.0)")
            else:
                print("🔄 Retrying in 5 seconds...")
                import time
                time.sleep(5)
    
    # Read the score that the worker saved
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            return json.load(f)
    return {"map": 0.0, "precision": 0.0, "recall": 0.0}

def run_kfold(model_path, kfold_yaml_prefix, out_dir):
    print("\n--- 5-Fold Cross Validation LP-FT ---")
    map_scores, p_scores, r_scores = [], [], []
    
    for k in range(1, 6):
        print(f"\n   -> FOLD {k}")
        data_yaml = f"{kfold_yaml_prefix}{k}.yaml"
        run_dir = os.path.join(out_dir, "full_lpft", "yolo_simam", f"fold_{k}") # IMPORTANT: Keeps EXACT same path
        
        metrics = spawn_worker("lpft", model_path, data_yaml, run_dir, epochs_lp=20, epochs_ft=50)
        map_scores.append(metrics["map"])
        p_scores.append(metrics["precision"])
        r_scores.append(metrics["recall"])
        print(f"   Fold {k} - mAP50-95: {metrics['map']:.4f} | P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")
        
    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0
    avg_p = sum(p_scores) / len(p_scores) if p_scores else 0
    avg_r = sum(r_scores) / len(r_scores) if r_scores else 0
    print(f"\n[SimAM] K-Fold Avg - mAP50-95: {avg_map:.4f} | P: {avg_p:.4f} | R: {avg_r:.4f}")
    return {"map": avg_map, "precision": avg_p, "recall": avg_r}

# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8+SimAM on Vietnam Dataset")
    parser.add_argument("--mode", type=str, default="master", choices=["master", "worker"], help="Internal use only")
    parser.add_argument("--task", type=str, default="all", choices=["all", "zero_shot", "few_shot", "kfold"], help="Task to run (master mode)")
    
    # Worker arguments
    parser.add_argument("--worker_task", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_yaml", type=str, default="")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--epochs_lp", type=int, default=0)
    parser.add_argument("--epochs_ft", type=int, default=0)
    args = parser.parse_args()

    # --- WORKER MODE (Internal isolated run) ---
    if args.mode == "worker":
        apply_simam_patch() # Must apply patch before loading model
        os.makedirs(args.run_dir, exist_ok=True)
        if args.worker_task == "zero_shot":
            worker_zero_shot(args.model_path, args.data_yaml, args.run_dir)
        elif args.worker_task == "lpft":
            worker_lp_ft(args.model_path, args.data_yaml, args.run_dir, args.epochs_lp, args.epochs_ft)
        return

    # --- MASTER MODE (User interaction) ---
    base_dir = str(PROJECT_ROOT)
    out_dir = os.path.join(base_dir, "runs", "vietnam_evaluation")
    
    # Datasets
    few_shot_yaml = os.path.join(base_dir, "data", "processed", "vietnam_kfold", "dataset_fewshot_50_images.yaml")
    kfold_yaml_prefix = os.path.join(base_dir, "data", "processed", "vietnam_kfold", "dataset_fold_")
    
    # Pretrained simam model
    simam_model_path = os.path.join(base_dir, "runs", "detect", "yolov8s_simam", "weights", "best.pt")
    if not os.path.exists(simam_model_path):
        print(f"❌ Cannot find pretrained simam model at: {simam_model_path}")
        print("   Please run train_simam.py first.")
        sys.exit(1)

    print("=" * 60)
    print("🚀 EVALUATING/FINE-TUNING: YOLOv8 + SimAM")
    print(f"   Task selected: {args.task}")
    print("=" * 60)
    
    # 1. Zero-Shot
    if args.task in ["all", "zero_shot"]:
        print("\n[Task 1] ZERO-SHOT EVALUATION")
        simam_zero_dir = os.path.join(out_dir, "zero_shot", "yolo_simam") # IMPORTANT: Path matches perfectly
        spawn_worker("zero_shot", simam_model_path, few_shot_yaml, simam_zero_dir)
        
    # 2. Few-Shot
    if args.task in ["all", "few_shot"]:
        print("\n[Task 2] FEW-SHOT (50 IMAGES) LP-FT")
        simam_few_dir = os.path.join(out_dir, "few_shot", "yolo_simam") # IMPORTANT: Path matches perfectly
        spawn_worker("lpft", simam_model_path, few_shot_yaml, simam_few_dir, epochs_lp=30, epochs_ft=50)
        
    # 3. Full K-Fold
    if args.task in ["all", "kfold"]:
        print("\n[Task 3] 5-FOLD CROSS VALIDATION LP-FT")
        run_kfold(simam_model_path, kfold_yaml_prefix, out_dir)

    print("\n✅ Script completed successfully.")

if __name__ == "__main__":
    main()
