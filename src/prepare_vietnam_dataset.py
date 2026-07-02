import os
import glob
import random
import yaml
from pathlib import Path
from sklearn.model_selection import KFold

def get_image_paths(directory):
    return glob.glob(os.path.join(directory, "*.jpg")) + \
           glob.glob(os.path.join(directory, "*.png")) + \
           glob.glob(os.path.join(directory, "*.jpeg"))

def main():
    random.seed(42)
    
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.join(base_dir, "data", "processed", "VietNamPotholeDataset_External")
    out_dir = os.path.join(base_dir, "data", "processed", "vietnam_kfold")
    os.makedirs(out_dir, exist_ok=True)
    
    # Gather all train and val images
    train_images = get_image_paths(os.path.join(dataset_dir, "train", "images"))
    val_images = get_image_paths(os.path.join(dataset_dir, "valid", "images"))
    all_images = train_images + val_images
    
    # Sort and shuffle for reproducibility
    all_images = sorted(all_images)
    random.shuffle(all_images)
    
    print(f"Total images for K-Fold: {len(all_images)}")
    
    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_train_lists = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_images), 1):
        fold_dir = os.path.join(out_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        
        train_list = [all_images[i] for i in train_idx]
        val_list = [all_images[i] for i in val_idx]
        fold_train_lists.append(train_list)
        
        # Write txt files
        train_txt = os.path.join(fold_dir, "train.txt")
        val_txt = os.path.join(fold_dir, "val.txt")
        
        with open(train_txt, "w") as f:
            f.write("\n".join(train_list))
        with open(val_txt, "w") as f:
            f.write("\n".join(val_list))
            
        # Create yaml
        yaml_path = os.path.join(out_dir, f"dataset_fold_{fold}.yaml")
        yaml_content = {
            "path": out_dir,
            "train": f"fold_{fold}/train.txt",
            "val": f"fold_{fold}/val.txt",
            "names": {0: "Pothole"}
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
        print(f"Fold {fold} generated: {len(train_list)} train, {len(val_list)} val.")
        
    # Generate Few-Shot (50 images from Fold 1 Train)
    fewshot_dir = os.path.join(out_dir, "fewshot_50")
    os.makedirs(fewshot_dir, exist_ok=True)
    
    fold1_train = fold_train_lists[0]
    fewshot_train_list = fold1_train[:50] # Take first 50 images
    
    # For Few-Shot evaluation, we use the original test set of Vietnam dataset
    test_images = get_image_paths(os.path.join(dataset_dir, "test", "images"))
    if not test_images:
        print("Test set not found, using Fold 1 validation set for few-shot val.")
        test_images = [all_images[i] for i in list(kf.split(all_images))[0][1]]
    
    fs_train_txt = os.path.join(fewshot_dir, "train.txt")
    fs_val_txt = os.path.join(fewshot_dir, "val.txt")
    
    with open(fs_train_txt, "w") as f:
        f.write("\n".join(fewshot_train_list))
    with open(fs_val_txt, "w") as f:
        f.write("\n".join(test_images))
        
    fs_yaml_path = os.path.join(out_dir, "dataset_fewshot_50_images.yaml")
    fs_yaml_content = {
        "path": out_dir,
        "train": "fewshot_50/train.txt",
        "val": "fewshot_50/val.txt",
        "names": {0: "Pothole"}
    }
    with open(fs_yaml_path, "w") as f:
        yaml.dump(fs_yaml_content, f, default_flow_style=False)
        
    print(f"Few-Shot dataset generated with {len(fewshot_train_list)} train images and {len(test_images)} val images.")

if __name__ == "__main__":
    main()
