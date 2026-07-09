import os
import sys
import pandas as pd
import torch
import json
from ultralytics import YOLO
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import SimAM modules and helper
import ultralytics.nn.modules.block as block
import ultralytics.nn.tasks as tasks
from src.models.simam_module import apply_simam_patch, OriginalC2f, C2f_SimAM

base_dir = str(PROJECT_ROOT)
out_dir = os.path.join(base_dir, 'runs', 'vietnam_evaluation')
few_shot_yaml = os.path.join(base_dir, 'data', 'processed', 'vietnam_kfold', 'dataset_fewshot_50_images.yaml')

# YOLOv8 Base paths
base_model_path = os.path.join(base_dir, 'runs', 'detect', 'yolov8s_baseline', 'weights', 'best.pt')
simam_model_path = os.path.join(base_dir, 'runs', 'detect', 'yolov8s_simam', 'weights', 'best.pt')

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

def get_best_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Best epoch based on mAP50-95
    best_row = df.loc[df['metrics/mAP50-95(B)'].idxmax()]
    return {
        'map': best_row['metrics/mAP50-95(B)'],
        'map50': best_row['metrics/mAP50(B)'],
        'precision': best_row['metrics/precision(B)'],
        'recall': best_row['metrics/recall(B)']
    }

def main():
    print('Evaluating Zero-Shot YOLO Base...')
    # Restore original C2f first to be sure
    block.C2f = OriginalC2f
    tasks.C2f = OriginalC2f
    
    model_base = YOLO(base_model_path)
    metrics_base = model_base.val(data=few_shot_yaml, split='val', imgsz=640, device=DEVICE, verbose=False)
    base_zero_res = {
        'map': float(metrics_base.box.map),
        'map50': float(metrics_base.box.map50),
        'precision': float(metrics_base.box.mp),
        'recall': float(metrics_base.box.mr)
    }

    print('Evaluating Zero-Shot YOLO SimAM...')
    # Apply SimAM patch
    apply_simam_patch()
    
    model_simam = YOLO(simam_model_path)
    metrics_simam = model_simam.val(data=few_shot_yaml, split='val', imgsz=640, device=DEVICE, verbose=False)
    simam_zero_res = {
        'map': float(metrics_simam.box.map),
        'map50': float(metrics_simam.box.map50),
        'precision': float(metrics_simam.box.mp),
        'recall': float(metrics_simam.box.mr)
    }

    # Restore original
    block.C2f = OriginalC2f
    tasks.C2f = OriginalC2f

    print('Parsing Few-Shot...')
    base_few_res = get_best_from_csv(os.path.join(out_dir, 'few_shot', 'yolo_base', 'ft', 'results.csv'))
    simam_few_res = get_best_from_csv(os.path.join(out_dir, 'few_shot', 'yolo_simam', 'ft', 'results.csv'))

    print('Parsing Full LP-FT...')
    base_k_res = []
    simam_k_res = []
    for k in range(1, 6):
        base_k_res.append(get_best_from_csv(os.path.join(out_dir, 'full_lpft', 'yolo_base', f'fold_{k}', 'ft', 'results.csv')))
        simam_k_res.append(get_best_from_csv(os.path.join(out_dir, 'full_lpft', 'yolo_simam', f'fold_{k}', 'ft', 'results.csv')))

    def avg_metrics(res_list):
        return {k: sum(d[k] for d in res_list)/len(res_list) for k in res_list[0]}

    base_full_res = avg_metrics(base_k_res)
    simam_full_res = avg_metrics(simam_k_res)

    print('\n' + '=' * 105)
    print('🏆 ABLATION STUDY RESULTS SUMMARY')
    print('=' * 105)
    print(f"{'Method':<20} | {'YOLOv8 Base (mAP50-95 / mAP50 / P / R)':<40} | {'YOLOv8 + SimAM (mAP50-95 / mAP50 / P / R)':<40}")
    print('-' * 105)

    bz = base_zero_res
    sz = simam_zero_res
    print(f"{'Zero-Shot Transfer':<20} | {bz['map']:.4f} / {bz['map50']:.4f} / {bz['precision']:.4f} / {bz['recall']:.4f} | {sz['map']:.4f} / {sz['map50']:.4f} / {sz['precision']:.4f} / {sz['recall']:.4f}")

    bf = base_few_res
    sf = simam_few_res
    print(f"{'Few-Shot (50 imgs)':<20} | {bf['map']:.4f} / {bf['map50']:.4f} / {bf['precision']:.4f} / {bf['recall']:.4f} | {sf['map']:.4f} / {sf['map50']:.4f} / {sf['precision']:.4f} / {sf['recall']:.4f}")

    bk = base_full_res
    sk = simam_full_res
    print(f"{'Full LP-FT (5-Fold)':<20} | {bk['map']:.4f} / {bk['map50']:.4f} / {bk['precision']:.4f} / {bk['recall']:.4f} | {sk['map']:.4f} / {sk['map50']:.4f} / {sk['precision']:.4f} / {sk['recall']:.4f}")
    print('=' * 105)

if __name__ == '__main__':
    main()
