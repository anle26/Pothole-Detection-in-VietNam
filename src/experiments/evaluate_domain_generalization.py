#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🌍 Domain Generalization Evaluation
Evaluates catastrophic forgetting by running Source Pre-trained models and 
Vietnam Fine-tuned (LP-FT) models on the Original Source Validation Set.
"""

import os
import sys
import argparse
import subprocess
import torch
import json
from pathlib import Path

# Fix Windows console emoji printing error
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def worker_eval(model_path, data_yaml, run_dir, use_simam):
    from ultralytics import YOLO
    if use_simam:
        from src.models.simam_module import apply_simam_patch, SimAM, C2f_SimAM
        apply_simam_patch()
        import sys
        sys.modules['__main__'].C2f_SimAM = C2f_SimAM
        sys.modules['__main__'].SimAM = SimAM
    else:
        # Guarantee pure baseline
        import ultralytics.nn.modules.block as block
        import ultralytics.nn.tasks as tasks
        from src.models.simam_module import OriginalC2f
        block.C2f = OriginalC2f
        tasks.C2f = OriginalC2f

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        project=os.path.dirname(run_dir),
        name=os.path.basename(run_dir),
        exist_ok=True,
        split='val',
        imgsz=640,
        device=0 if torch.cuda.is_available() else "cpu"
    )
    
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump({
            "map": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr)
        }, f)

def run_eval_subprocess(model_path, data_yaml, run_dir, use_simam):
    if os.path.exists(os.path.join(run_dir, "metrics.json")):
        print(f"✅ Skipping completed evaluation: {run_dir}")
        with open(os.path.join(run_dir, "metrics.json"), "r") as f:
            return json.load(f)

    print(f"\n🚀 Evaluating model: {model_path}")
    print(f"   Output dir: {run_dir}")
    
    script_path = os.path.abspath(__file__)
    cmd = [
        sys.executable, script_path,
        "--mode", "worker",
        "--model_path", model_path,
        "--data_yaml", data_yaml,
        "--run_dir", run_dir
    ]
    if use_simam:
        cmd.append("--use_simam")
        
    subprocess.run(cmd, check=True)
    
    with open(os.path.join(run_dir, "metrics.json"), "r") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="master")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_yaml", type=str, default="")
    parser.add_argument("--run_dir", type=str, default="")
    parser.add_argument("--use_simam", action="store_true")
    args = parser.parse_args()

    if args.mode == "worker":
        worker_eval(args.model_path, args.data_yaml, args.run_dir, args.use_simam)
        return

    base_dir = str(PROJECT_ROOT)
    source_yaml = os.path.join(base_dir, "data", "processed", "combined_pothole", "dataset.yaml")
    out_dir = os.path.join(base_dir, "runs", "domain_generalization")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Base Models
    base_pretrained = os.path.join(base_dir, "runs", "detect", "yolov8s_baseline", "weights", "best.pt")
    # Using Fold 1 as the representative fine-tuned model
    base_finetuned = os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_base", "fold_1", "ft", "weights", "best.pt")
    
    # 2. SimAM Models
    simam_pretrained = os.path.join(base_dir, "runs", "detect", "yolov8s_simam", "weights", "best.pt")
    simam_finetuned = os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_simam", "fold_1", "ft", "weights", "best.pt")

    if not os.path.exists(source_yaml):
        print(f"❌ Cannot find source dataset at: {source_yaml}")
        sys.exit(1)

    print("=" * 80)
    print("🌍 DOMAIN GENERALIZATION EVALUATION (Catastrophic Forgetting Test)")
    print("   Dataset: Source Domain (RDD2022 + BharatPothole)")
    print("=" * 80)

    # Evaluate
    metrics = {}
    
    if os.path.exists(base_pretrained) and os.path.exists(base_finetuned):
        metrics["base_pt"] = run_eval_subprocess(base_pretrained, source_yaml, os.path.join(out_dir, "base_pretrained"), False)
        metrics["base_ft"] = run_eval_subprocess(base_finetuned, source_yaml, os.path.join(out_dir, "base_finetuned"), False)
    else:
        print("⚠️ Missing Baseline weights (Pre-trained or Fine-tuned). Skipping Baseline.")
        
    if os.path.exists(simam_pretrained) and os.path.exists(simam_finetuned):
        metrics["simam_pt"] = run_eval_subprocess(simam_pretrained, source_yaml, os.path.join(out_dir, "simam_pretrained"), True)
        metrics["simam_ft"] = run_eval_subprocess(simam_finetuned, source_yaml, os.path.join(out_dir, "simam_finetuned"), True)
    else:
        print("⚠️ Missing SimAM weights (Pre-trained or Fine-tuned). Skipping SimAM.")

    # Print Report
    print("\n" + "=" * 90)
    print(f"{'Model Architecture':<20} | {'Source Pre-trained (mAP)':<25} | {'Vietnam Fine-tuned (mAP)':<25} | {'Relative Drop'}")
    print("-" * 90)
    
    if "base_pt" in metrics and "base_ft" in metrics:
        b_pt = metrics["base_pt"]["map"]
        b_ft = metrics["base_ft"]["map"]
        b_drop = (b_pt - b_ft) / b_pt * 100 if b_pt > 0 else 0
        print(f"{'YOLOv8 Baseline':<20} | {b_pt:<25.4f} | {b_ft:<25.4f} | -{b_drop:.1f}%")
        
    if "simam_pt" in metrics and "simam_ft" in metrics:
        s_pt = metrics["simam_pt"]["map"]
        s_ft = metrics["simam_ft"]["map"]
        s_drop = (s_pt - s_ft) / s_pt * 100 if s_pt > 0 else 0
        print(f"{'YOLOv8 + SimAM':<20} | {s_pt:<25.4f} | {s_ft:<25.4f} | -{s_drop:.1f}%")
        
    print("=" * 90)
    print("📌 INTERPRETATION FOR PAPER:")
    print("- Drop < 15%: LP-FT preserved domain generalization well.")
    print("- Drop 15-30%: Moderate forgetting, trade-off for better target adaptation.")
    print("- Drop > 30%: Catastrophic forgetting occurred. Model overfitted to Vietnam.")

if __name__ == "__main__":
    main()
