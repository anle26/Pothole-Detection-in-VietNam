#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
📊 Benchmark Tool for YOLO Pothole Detection Models
Calculates Parameters, GFLOPs, and FPS for all models in the comparison experiment.
This outputs a ready-to-copy Markdown table for your paper!
"""

import os
import sys
import time
import torch
import ultralytics
from ultralytics import YOLO
from pathlib import Path

# Fix Windows console emoji printing error
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import Patches
from src.models.simam_module import apply_simam_patch, SimAM, C2f_SimAM
from src.models.cbam_module import apply_cbam_patch, CBAM, C2f_CBAM

# Ensure they are available in __main__
sys.modules['__main__'].C2f_SimAM = C2f_SimAM
sys.modules['__main__'].SimAM = SimAM
sys.modules['__main__'].C2f_CBAM = C2f_CBAM
sys.modules['__main__'].CBAM = CBAM

def get_gflops_thop(model, imgsz=640, device="cuda"):
    """Calculate GFLOPs using thop if ultralytics built-in fails."""
    try:
        from thop import profile as thop_profile
        dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
        flops, _ = thop_profile(model.model, inputs=(dummy_input,), verbose=False)
        return flops / 1e9
    except ImportError:
        return -1

def benchmark_model(model_name, weight_path, patch_func=None):
    if not os.path.exists(weight_path):
        return {"name": model_name, "params": "N/A", "gflops": "N/A", "fps": "N/A", "status": "Weights not found"}

    print(f"\n⏳ Benchmarking {model_name}...")
    
    # Apply monkey patch before loading
    if patch_func:
        patch_func()
    elif "C2f" in str(ultralytics.nn.modules.block.C2f): 
        # Clean up any previous patch for baseline models
        import ultralytics.nn.modules.block as block
        import ultralytics.nn.tasks as tasks
        # SimAM OriginalC2f holds the clean copy
        from src.models.simam_module import OriginalC2f
        block.C2f = OriginalC2f
        tasks.C2f = OriginalC2f
        
    model = YOLO(weight_path)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # --- 1. Parameters ---
    total_params = sum(p.numel() for p in model.model.parameters())
    params_m = total_params / 1e6
    
    # --- 2. GFLOPs ---
    try:
        model_info = model.info(detailed=False, verbose=False)
        gflops = model_info[3] if isinstance(model_info, (list, tuple)) and len(model_info) >= 4 else None
    except:
        gflops = None
        
    if not gflops:
        gflops = get_gflops_thop(model, device=device)
        
    # --- 3. FPS (Latency) ---
    model.model.eval()
    dummy = torch.randn(1, 3, 640, 640, device=device)
    
    # Warmup
    print("   -> Warming up GPU...")
    with torch.no_grad():
        for _ in range(100):
            _ = model.model(dummy)
            
    # Timed runs
    print("   -> Running 500 inferences...")
    num_runs = 500
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    avg_latency_ms = (t_end - t_start) / num_runs * 1000
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
    
    # Free memory
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "name": model_name,
        "params": f"{params_m:.2f}",
        "gflops": f"{gflops:.2f}" if gflops != -1 else "N/A",
        "fps": f"{fps:.1f}",
        "status": "OK"
    }

def get_mAP_from_csv(csv_path):
    """Parses ultralytics results.csv to get the final mAP50 and mAP50-95."""
    if not os.path.exists(csv_path):
        return "N/A", "N/A"
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        # ultralytics results.csv column names often have leading spaces
        df.columns = df.columns.str.strip()
        last_row = df.iloc[-1]
        
        # Depending on ultralytics version, column might be 'metrics/mAP50(B)' or similar
        map50_col = next((c for c in df.columns if 'mAP50(B)' in c or c == 'metrics/mAP_0.5'), None)
        map50_95_col = next((c for c in df.columns if 'mAP50-95(B)' in c or c == 'metrics/mAP_0.5:0.95'), None)
        
        map50 = f"{last_row[map50_col]:.4f}" if map50_col else "N/A"
        map50_95 = f"{last_row[map50_95_col]:.4f}" if map50_95_col else "N/A"
        return map50, map50_95
    except Exception as e:
        return "Error", "Error"

def main():
    print("="*60)
    print("🔬 YOLO SPEED & EFFICIENCY BENCHMARK")
    print("="*60)
    
    import ultralytics
    base_dir = str(PROJECT_ROOT)
    
    # Models to test and their corresponding evaluation CSV
    models_to_test = [
        {
            "name": "YOLOv5s",
            "weight": os.path.join(base_dir, "runs", "detect", "yolov5s_baseline", "weights", "best.pt"),
            "csv_path": os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolov5s", "fold_1", "ft", "results.csv"),
            "patch": None
        },
        {
            "name": "YOLOv8s Base",
            "weight": os.path.join(base_dir, "runs", "detect", "yolov8s_baseline", "weights", "best.pt"),
            "csv_path": os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_base", "fold_1", "ft", "results.csv"),
            "patch": None
        },
        {
            "name": "YOLOv8s+CBAM",
            "weight": os.path.join(base_dir, "runs", "detect", "yolov8s_cbam", "weights", "best.pt"),
            "csv_path": os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_cbam", "fold_1", "ft", "results.csv"),
            "patch": apply_cbam_patch
        },
        {
            "name": "YOLOv8s+SimAM",
            "weight": os.path.join(base_dir, "runs", "detect", "yolov8s_simam", "weights", "best.pt"),
            "csv_path": os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_simam", "fold_1", "ft", "results.csv"),
            "patch": apply_simam_patch
        }
    ]
    
    results = []
    for m in models_to_test:
        res = benchmark_model(m["name"], m["weight"], m["patch"])
        map50, map50_95 = get_mAP_from_csv(m["csv_path"])
        res["map50"] = map50
        res["map50_95"] = map50_95
        results.append(res)
        
    print("\n\n" + "="*100)
    print("📊 COMPARISON EXPERIMENTS TABLE (Ready for Paper)")
    print("="*100)
    print(f"| {'Model':<15} | {'Params (M)':<10} | {'GFLOPs':<8} | {'FPS':<6} | {'mAP@0.5':<8} | {'mAP@0.5:0.95':<12} | {'Status':<15} |")
    print(f"|{'-'*17}|{'-'*12}|{'-'*10}|{'-'*8}|{'-'*10}|{'-'*14}|{'-'*17}|")
    for r in results:
        print(f"| {r['name']:<15} | {r['params']:<10} | {r['gflops']:<8} | {r['fps']:<6} | {r['map50']:<8} | {r['map50_95']:<12} | {r['status']:<15} |")
    print("="*100)
    
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"* Note: FPS and Latency were measured on {device_name} with batch-size 1, FP32 precision.")
    print("* Note: mAP metrics are extracted from Vietnam dataset Fine-Tuning (Fold 1).")

if __name__ == "__main__":
    main()
