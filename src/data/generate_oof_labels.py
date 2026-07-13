import os
import shutil
import cv2
import glob
from ultralytics import YOLO
import torch
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simam_module import apply_simam_patch

def main():
    base_dir = str(PROJECT_ROOT)
    
    # Output directories
    relabel_dir = os.path.join(base_dir, "data", "relabel")
    images_dir = os.path.join(relabel_dir, "images")
    labels_dir = os.path.join(relabel_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Apply monkey-patching for SimAM compatibility
    apply_simam_patch()
    
    conf_thresh = 0.15 
    total_processed = 0
    
    # 1. Process K-Fold OOF predictions (400 images)
    for fold in range(1, 6):
        print(f"\n🚀 Processing Fold {fold}...")
        weight_path = os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_simam", f"fold_{fold}", "ft", "weights", "best.pt")
        
        if not os.path.exists(weight_path):
            print(f"⚠️ Model weight not found for fold {fold}: {weight_path}")
            continue
            
        model = YOLO(weight_path)
        
        val_txt_path = os.path.join(base_dir, "data", "processed", "vietnam_kfold", f"fold_{fold}", "val.txt")
        with open(val_txt_path, "r") as f:
            image_paths = [line.strip() for line in f.readlines() if line.strip()]
            
        print(f"   Found {len(image_paths)} validation images in Fold {fold}.")
        
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
                
            basename = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(basename)
            
            results = model.predict(source=img_path, conf=conf_thresh, verbose=False, imgsz=640)
            for result in results:
                result.save(filename=os.path.join(images_dir, basename))
                with open(os.path.join(labels_dir, f"{name_no_ext}.txt"), "w") as txt_f:
                    for i in range(len(result.boxes)):
                        cls = int(result.boxes.cls[i].item())
                        x, y, w, h = result.boxes.xywhn[i].tolist()
                        txt_f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            total_processed += 1
            
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Process Test set predictions (100 images) using Fold 1 model (OOF for test set)
    print(f"\n🚀 Processing Test Set (100 images) using Fold 1 model...")
    weight_path_fold1 = os.path.join(base_dir, "runs", "vietnam_evaluation", "full_lpft", "yolo_simam", "fold_1", "ft", "weights", "best.pt")
    if os.path.exists(weight_path_fold1):
        model = YOLO(weight_path_fold1)
        dataset_dir = os.path.join(base_dir, "data", "processed", "VietNamPotholeDataset_External")
        test_images = glob.glob(os.path.join(dataset_dir, "test", "images", "*.jpg")) + \
                      glob.glob(os.path.join(dataset_dir, "test", "images", "*.png")) + \
                      glob.glob(os.path.join(dataset_dir, "test", "images", "*.jpeg"))
        
        print(f"   Found {len(test_images)} test images.")
        
        for img_path in test_images:
            basename = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(basename)
            
            results = model.predict(source=img_path, conf=conf_thresh, verbose=False, imgsz=640)
            for result in results:
                result.save(filename=os.path.join(images_dir, basename))
                with open(os.path.join(labels_dir, f"{name_no_ext}.txt"), "w") as txt_f:
                    for i in range(len(result.boxes)):
                        cls = int(result.boxes.cls[i].item())
                        x, y, w, h = result.boxes.xywhn[i].tolist()
                        txt_f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            total_processed += 1
            
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
    print(f"\n✅ All OOF predictions saved to: {relabel_dir}")
    print(f"Total processed images: {total_processed}")

if __name__ == "__main__":
    main()
