import os
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Tự động xác định PROJECT_ROOT (lùi 3 cấp từ src/data/convert_labels.py)
PROJECT_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
RAW_RDD_DIR = PROJECT_ROOT / 'data' / 'raw' / 'rdd2022' / 'RDD_SPLIT'
RAW_BHARAT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'bharatpothole' / 'BharatPotHole' / 'BharatPotHole'

PROCESSED_RDD_DIR = PROJECT_ROOT / 'data' / 'processed' / 'rdd2022_processed'
COMBINED_DIR = PROJECT_ROOT / 'data' / 'processed' / 'combined_pothole'

# Tạo các thư mục đầu ra nếu chưa tồn tại
PROCESSED_RDD_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_DIR.mkdir(parents=True, exist_ok=True)


def process_rdd_split(split_name, src_root, dest_split_dir):
    """
    Tiền xử lý dữ liệu RDD-2022:
    - Lọc bỏ ảnh China_Drone
    - Lọc giữ lại nhãn lớp 4 (Pothole) và chuyển thành lớp 0
    - Lấy ngẫu nhiên 7% ảnh nền (background)
    """
    src_img_dir = src_root / split_name / 'images'
    src_lbl_dir = src_root / split_name / 'labels'
    
    dest_img_dir = dest_split_dir / 'images'
    dest_lbl_dir = dest_split_dir / 'labels'
    
    dest_img_dir.mkdir(parents=True, exist_ok=True)
    dest_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Tìm tất cả ảnh và loại bỏ China_Drone
    img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    all_images = []
    for ext in img_extensions:
        all_images.extend(src_img_dir.glob(f'*{ext}'))
    
    filtered_images = [img for img in all_images if "China_Drone" not in img.name]
    
    pothole_images = []
    background_images = []
    
    # 2. Phân loại ảnh có chứa ổ gà và ảnh nền
    for img_path in filtered_images:
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        has_pothole = False
        
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5 and int(parts[0]) == 4:  # ID 4 là Pothole
                        has_pothole = True
                        break
                        
        if has_pothole:
            pothole_images.append(img_path)
        else:
            background_images.append(img_path)
            
    # 3. Lấy ngẫu nhiên 7% ảnh nền
    random.seed(42)
    num_bg_keep = int(len(background_images) * 0.07)
    selected_bg = random.sample(background_images, num_bg_keep) if num_bg_keep > 0 else []
    
    # 4. Sao chép hình ảnh và tạo file nhãn tương ứng
    final_images = pothole_images + selected_bg
    print(f"\n⚙️ Đang xử lý RDD-2022 split [{split_name}] -> [{dest_split_dir.name}]:")
    print(f"   • Số ảnh ban đầu (đã loại China_Drone): {len(filtered_images):,}")
    print(f"   • Số ảnh Pothole (giữ lại 100%): {len(pothole_images):,}")
    print(f"   • Số ảnh nền gốc: {len(background_images):,} -> Giữ lại 7%: {len(selected_bg):,}")
    print(f"   • Tổng số ảnh lưu sang thư mục đích: {len(final_images):,}")
    
    for img_path in tqdm(final_images, desc=f"Đang copy {split_name}"):
        # Sao chép ảnh
        shutil.copy2(img_path, dest_img_dir / img_path.name)
        
        # Xử lý nhãn
        dest_lbl_path = dest_lbl_dir / f"{img_path.stem}.txt"
        lbl_path = src_lbl_dir / f"{img_path.stem}.txt"
        
        if img_path in pothole_images and lbl_path.exists():
            new_lines = []
            with open(lbl_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5 and int(parts[0]) == 4:
                        parts[0] = '0'  # Chuyển đổi ID lớp 4 thành lớp 0
                        new_lines.append(" ".join(parts))
            with open(dest_lbl_path, 'w') as f:
                f.write("\n".join(new_lines) + "\n")
        else:
            # Tạo file nhãn trống cho ảnh nền
            with open(dest_lbl_path, 'w') as f:
                pass


def merge_dataset_split(src_dirs, dest_split_dir):
    """
    Gộp nhiều thư mục nguồn (chứa images/ và labels/) vào một thư mục đích chung
    """
    dest_images = dest_split_dir / 'images'
    dest_labels = dest_split_dir / 'labels'
    
    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    total_images_copied = 0
    total_labels_copied = 0
    
    for src in src_dirs:
        src_images = src / 'images'
        src_labels = src / 'labels'
        
        if not src_images.exists() or not src_labels.exists():
            print(f"⚠️ Cảnh báo: Thư mục nguồn {src} không tồn tại thư mục images hoặc labels!")
            continue
            
        # Sao chép ảnh
        img_files = list(src_images.glob('*'))
        for f in tqdm(img_files, desc=f"Gộp ảnh từ {src.parent.name}/{src.name}", leave=False):
            if f.is_file():
                shutil.copy2(f, dest_images / f.name)
                total_images_copied += 1
                
        # Sao chép nhãn
        lbl_files = list(src_labels.glob('*'))
        for f in tqdm(lbl_files, desc=f"Gộp nhãn từ {src.parent.name}/{src.name}", leave=False):
            if f.is_file():
                shutil.copy2(f, dest_labels / f.name)
                total_labels_copied += 1
                
    print(f"✅ Đã gộp thành công vào [{dest_split_dir.name}]: {total_images_copied:,} ảnh và {total_labels_copied:,} file nhãn.")


def analyze_combined_dataset(dataset_dir):
    stats = {}
    splits = ['train', 'val']
    
    for split in splits:
        img_dir = dataset_dir / split / 'images'
        lbl_dir = dataset_dir / split / 'labels'
        
        all_images = list(img_dir.glob('*'))
        all_labels = list(lbl_dir.glob('*.txt'))
        
        total_images = len(all_images)
        
        pothole_boxes = 0
        images_with_pothole = 0
        
        for lbl_path in all_labels:
            has_pothole = False
            if lbl_path.exists():
                with open(lbl_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 5 and int(parts[0]) == 0:  # Lớp 0 hiện tại là Pothole
                            pothole_boxes += 1
                            has_pothole = True
            
            if has_pothole:
                images_with_pothole += 1
                
        real_background_images = total_images - images_with_pothole
        
        stats[split] = {
            'Tổng số ảnh': total_images,
            'Số ảnh chứa Ổ gà (Lớp 0)': images_with_pothole,
            'Số ảnh nền (Background)': real_background_images,
            'Tỷ lệ ảnh nền (%)': f"{round(real_background_images / total_images * 100, 2)}%" if total_images > 0 else "0%",
            'Tổng số nhãn Ổ gà (BBox)': pothole_boxes
        }
        
    df_stats = pd.DataFrame.from_dict(stats, orient='index')
    return df_stats


def main():
    print("✅ Khởi tạo cấu hình và thư mục thành công.")
    
    # Tiến hành tiền xử lý và gộp các tập dữ liệu RDD-2022
    # 1. Gộp RDD Train -> PROCESSED_RDD_DIR / 'train'
    process_rdd_split('train', RAW_RDD_DIR, PROCESSED_RDD_DIR / 'train')

    # 2. Gộp RDD Test -> PROCESSED_RDD_DIR / 'train' (Gộp Train và Test)
    process_rdd_split('test', RAW_RDD_DIR, PROCESSED_RDD_DIR / 'train')

    # 3. RDD Val -> PROCESSED_RDD_DIR / 'val' (Giữ riêng)
    process_rdd_split('val', RAW_RDD_DIR, PROCESSED_RDD_DIR / 'val')

    # 1. Gộp tập Train chung:
    # - RDD-2022 processed train (gồm train + test)
    # - BharatPothole train
    # - BharatPothole test
    print("\n🔄 Bắt đầu gộp tập TRAIN chung...")
    merge_dataset_split([
        PROCESSED_RDD_DIR / 'train',
        RAW_BHARAT_DIR / 'train',
        RAW_BHARAT_DIR / 'test'
    ], COMBINED_DIR / 'train')

    # 2. Gộp tập Val chung:
    # - RDD-2022 processed val
    # - BharatPothole valid
    print("\n🔄 Bắt đầu gộp tập VAL chung...")
    merge_dataset_split([
        PROCESSED_RDD_DIR / 'val',
        RAW_BHARAT_DIR / 'valid'
    ], COMBINED_DIR / 'val')

    print("\n📊 BẢNG THỐNG KÊ CHI TIẾT TẬP DỮ LIỆU GỘP CUỐI CÙNG:")
    df_result = analyze_combined_dataset(COMBINED_DIR)
    print(df_result.to_string())


if __name__ == "__main__":
    main()
