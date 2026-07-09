#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
🏆 Vietnam Dataset Evaluation Summary
This script does NOT run any training.
It simply scans the 'runs/vietnam_evaluation' folder, reads the 'metrics.json' 
produced by finetune_vietnam_baseline.py and finetune_vietnam_simam.py, 
and prints a clean comparison table for your research paper.
"""

import os
import json
import sys
from pathlib import Path

# Fix Windows console emoji printing error
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def load_score(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception:
            return {"map": 0.0, "precision": 0.0, "recall": 0.0}
    return {"map": 0.0, "precision": 0.0, "recall": 0.0}

def get_kfold_average(model_type, out_dir):
    map_scores, p_scores, r_scores = [], [], []
    for k in range(1, 6):
        score_file = os.path.join(out_dir, "full_lpft", model_type, f"fold_{k}", "metrics.json")
        score = load_score(score_file)
        if score["map"] > 0:
            map_scores.append(score["map"])
            p_scores.append(score["precision"])
            r_scores.append(score["recall"])
            
    if not map_scores:
        return {"map": 0.0, "precision": 0.0, "recall": 0.0}
        
    avg_map = sum(map_scores) / len(map_scores)
    avg_p = sum(p_scores) / len(p_scores)
    avg_r = sum(r_scores) / len(r_scores)
    return {"map": avg_map, "precision": avg_p, "recall": avg_r}

def main():
    base_dir = str(PROJECT_ROOT)
    out_dir = os.path.join(base_dir, "runs", "vietnam_evaluation")
    
    if not os.path.exists(out_dir):
        print(f"❌ Evaluation directory not found at {out_dir}")
        print("   Make sure you have run the finetuning scripts first!")
        sys.exit(1)
        
    print("=" * 85)
    print("🏆 VIETNAM DATASET: FINAL EVALUATION COMPARISON")
    print("=" * 85)
    
    # Read Baseline Metrics
    bz = load_score(os.path.join(out_dir, "zero_shot", "yolo_base", "metrics.json"))
    bf = load_score(os.path.join(out_dir, "few_shot", "yolo_base", "metrics.json"))
    bk = get_kfold_average("yolo_base", out_dir)
    
    # Read SimAM Metrics
    sz = load_score(os.path.join(out_dir, "zero_shot", "yolo_simam", "metrics.json"))
    sf = load_score(os.path.join(out_dir, "few_shot", "yolo_simam", "metrics.json"))
    sk = get_kfold_average("yolo_simam", out_dir)

    print(f"{'Method':<25} | {'YOLOv8 Base (mAP / P / R)':<30} | {'YOLOv8 + SimAM (mAP / P / R)':<30}")
    print("-" * 85)
    
    bz_s = f"{bz['map']:.4f} / {bz['precision']:.4f} / {bz['recall']:.4f}"
    sz_s = f"{sz['map']:.4f} / {sz['precision']:.4f} / {sz['recall']:.4f}"
    print(f"{'1. Zero-Shot Transfer':<25} | {bz_s:<30} | {sz_s:<30}")
    
    bf_s = f"{bf['map']:.4f} / {bf['precision']:.4f} / {bf['recall']:.4f}"
    sf_s = f"{sf['map']:.4f} / {sf['precision']:.4f} / {sf['recall']:.4f}"
    print(f"{'2. Few-Shot (50 imgs)':<25} | {bf_s:<30} | {sf_s:<30}")
    
    bk_s = f"{bk['map']:.4f} / {bk['precision']:.4f} / {bk['recall']:.4f}"
    sk_s = f"{sk['map']:.4f} / {sk['precision']:.4f} / {sk['recall']:.4f}"
    print(f"{'3. Full LP-FT (5-Fold)':<25} | {bk_s:<30} | {sk_s:<30}")
    print("=" * 85)

if __name__ == "__main__":
    main()
