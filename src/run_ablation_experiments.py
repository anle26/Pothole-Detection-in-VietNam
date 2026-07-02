#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🚀 Ablation Study: Vietnam Pothole Evaluation
Uses Subprocess to guarantee 100% clean CUDA memory for every single training run!
"""

import os
import sys
import argparse
import subprocess
import torch
import torch.nn as nn
from ultralytics import YOLO
import ultralytics.nn.modules.block as block
import ultralytics.nn.tasks as tasks

# Disable WandB to prevent login blocks
os.environ["WANDB_DISABLED"] = "true"

# ==============================================================================
# 1. SimAM DYNAMIC PATCHING
# ==============================================================================
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)

OriginalC2f = block.C2f

class C2f_SimAM(OriginalC2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.simam = SimAM()
        
    def forward(self, x):
        out = super().forward(x)
        return self.simam(out)

def apply_simam_patch():
    print("🔧 Applying SimAM monkey-patch...")
    block.C2f = C2f_SimAM
    tasks.C2f = C2f_SimAM

def remove_simam_patch():
    print("🔧 Restoring Base C2f...")
    block.C2f = OriginalC2f
    tasks.C2f = OriginalC2f


# ==============================================================================
# 2. WORKER FUNCTIONS (Run in isolated processes)
# ==============================================================================
def worker_zero_shot(model_path, data_yaml, run_dir):
    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        project=os.path.dirname(run_dir),
        name=os.path.basename(run_dir),
        exist_ok=True,
        split='val',
    )
    # Save the mAP score to a file so the master process can read it
    with open(os.path.join(run_dir, "map_score.txt"), "w") as f:
        f.write(str(metrics.box.map))

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
        lr0=0.01,
        workers=0,
        batch=16,
        device=0,
        amp=False, # Must be False on Windows to avoid crash
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
        lr0=0.001,
        workers=0,
        batch=16,
        device=0,
        amp=False, # Must be False on Windows to avoid crash
        save=True,
        patience=15
    )
    
    # Evaluate
    ft_best = os.path.join(ft_dir, "weights", "best.pt")
    if os.path.exists(ft_best):
        eval_model = YOLO(ft_best)
        metrics = eval_model.val(data=data_yaml, exist_ok=True, split='val')
        with open(os.path.join(run_dir, "map_score.txt"), "w") as f:
            f.write(str(metrics.box.map))

# ==============================================================================
# 3. MASTER FUNCTIONS (Orchestrates Subprocesses)
# ==============================================================================
def spawn_worker(task, model_type, model_path, data_yaml, run_dir, epochs_lp=0, epochs_ft=0):
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--mode", "worker",
        "--task", task,
        "--model_type", model_type,
        "--model_path", model_path,
        "--data_yaml", data_yaml,
        "--run_dir", run_dir,
        "--epochs_lp", str(epochs_lp),
        "--epochs_ft", str(epochs_ft)
    ]
    # Check if this task was already completed in a previous run
    score_file = os.path.join(run_dir, "map_score.txt")
    if os.path.exists(score_file):
        print(f"\n✅ Skipping completed task: [{model_type}] {task}")
        with open(score_file, "r") as f:
            return float(f.read().strip())

    print(f"\n🚀 Spawning isolated process for: [{model_type}] {task}")
    
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
    score_file = os.path.join(run_dir, "map_score.txt")
    if os.path.exists(score_file):
        with open(score_file, "r") as f:
            return float(f.read().strip())
    return 0.0

def run_kfold(model_path, model_type, kfold_yaml_prefix, out_dir):
    print(f"\n[{model_type}] Kịch Bản 3: Full K-Fold LP-FT")
    map_scores = []
    
    for k in range(1, 6):
        print(f"\n   --- FOLD {k} ---")
        data_yaml = f"{kfold_yaml_prefix}{k}.yaml"
        run_dir = os.path.join(out_dir, "full_lpft", model_type, f"fold_{k}")
        
        score = spawn_worker("lpft", model_type, model_path, data_yaml, run_dir, 20, 50)
        map_scores.append(score)
        print(f"   Fold {k} mAP50-95: {score:.4f}")
        
    avg_map = sum(map_scores) / len(map_scores) if map_scores else 0
    print(f"\n[{model_type}] K-Fold Average mAP50-95: {avg_map:.4f}")
    return avg_map


# ==============================================================================
# ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="master", choices=["master", "worker"])
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--model_type", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_yaml", type=str, default="")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--epochs_lp", type=int, default=0)
    parser.add_argument("--epochs_ft", type=int, default=0)
    args = parser.parse_args()

    # --- WORKER MODE ---
    if args.mode == "worker":
        if args.model_type == "yolo_simam":
            apply_simam_patch()
        else:
            remove_simam_patch()
            
        os.makedirs(args.run_dir, exist_ok=True)
        if args.task == "zero_shot":
            worker_zero_shot(args.model_path, args.data_yaml, args.run_dir)
        elif args.task == "lpft":
            worker_lp_ft(args.model_path, args.data_yaml, args.run_dir, args.epochs_lp, args.epochs_ft)
        return

    # --- MASTER MODE ---
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(base_dir, "runs", "vietnam_evaluation")
    
    # Datasets
    few_shot_yaml = os.path.join(base_dir, "data", "processed", "vietnam_kfold", "dataset_fewshot_50_images.yaml")
    kfold_yaml_prefix = os.path.join(base_dir, "data", "processed", "vietnam_kfold", "dataset_fold_")
    
    # Pretrained models
    base_model_path = os.path.join(base_dir, "runs", "detect", "yolov8s_baseline", "weights", "best.pt")
    simam_model_path = os.path.join(base_dir, "runs", "detect", "yolov8s_simam", "weights", "best.pt")

    print("=" * 60)
    print("🚀 STARTING ABLATION STUDY EXPERIMENTS (MULTIPROCESS SAFE)")
    print("=" * 60)
    
    # --- YOLOv8 BASE ---
    print("\n" + "=" * 40)
    print("EVALUATING YOLOv8 BASE")
    print("=" * 40)
    base_zero_dir = os.path.join(out_dir, "zero_shot", "yolo_base")
    base_few_dir = os.path.join(out_dir, "few_shot", "yolo_base")
    
    base_zero_map = spawn_worker("zero_shot", "yolo_base", base_model_path, few_shot_yaml, base_zero_dir)
    base_few_map = spawn_worker("lpft", "yolo_base", base_model_path, few_shot_yaml, base_few_dir, 30, 50)
    base_full_map = run_kfold(base_model_path, "yolo_base", kfold_yaml_prefix, out_dir)
    
    # --- YOLOv8 + SimAM ---
    print("\n" + "=" * 40)
    print("EVALUATING YOLOv8 + SimAM")
    print("=" * 40)
    simam_zero_dir = os.path.join(out_dir, "zero_shot", "yolo_simam")
    simam_few_dir = os.path.join(out_dir, "few_shot", "yolo_simam")
    
    simam_zero_map = spawn_worker("zero_shot", "yolo_simam", simam_model_path, few_shot_yaml, simam_zero_dir)
    simam_few_map = spawn_worker("lpft", "yolo_simam", simam_model_path, few_shot_yaml, simam_few_dir, 30, 50)
    simam_full_map = run_kfold(simam_model_path, "yolo_simam", kfold_yaml_prefix, out_dir)
    
    # --- SUMMARY ---
    print("\n" + "=" * 60)
    print("🏆 ABLATION STUDY RESULTS SUMMARY (mAP@0.5:0.95)")
    print("=" * 60)
    print(f"{'Method':<25} | {'YOLOv8 Base':<15} | {'YOLOv8 + SimAM':<15}")
    print("-" * 60)
    print(f"{'Zero-Shot Transfer':<25} | {base_zero_map:<15.4f} | {simam_zero_map:<15.4f}")
    print(f"{'Few-Shot (50 images)':<25} | {base_few_map:<15.4f} | {simam_few_map:<15.4f}")
    print(f"{'Full LP-FT (5-Fold)':<25} | {base_full_map:<15.4f} | {simam_full_map:<15.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
