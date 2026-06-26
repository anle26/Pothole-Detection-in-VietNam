"""
📊 01 — Khám phá dữ liệu thô (Raw Data Exploration)
=====================================================
Script thay thế phần khám phá trong notebook 01_explore_raw_data.ipynb

Chạy bằng: python explore_raw_data.py
Hoặc copy từng phần vào notebook để chạy từng cell.
"""

import os
import sys
import glob
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# ============================================================
# CẤU HÌNH
# ============================================================
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 100
sns.set_style('whitegrid')
sns.set_palette('husl')

import os
# Tự động xác định PROJECT_ROOT
PROJECT_ROOT = Path(os.getcwd())
if PROJECT_ROOT.name == 'notebooks':
    PROJECT_ROOT = PROJECT_ROOT.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'

# Đường dẫn dataset
RDD_ROOT = RAW_DATA_DIR / 'rdd2022' / 'RDD_SPLIT'
BHARAT_ROOT = RAW_DATA_DIR / 'bharatpothole' / 'BharatPotHole' / 'BharatPotHole'

# RDD-2022 class names (5 classes: index 0-4)
RDD_CLASSES = {
    0: 'Longitudinal Crack',
    1: 'Transverse Crack',
    2: 'Alligator Crack',
    3: 'Other Corruption',
    4: 'Pothole'
}

# BharatPotHole class names (1 class: index 0)
BHARAT_CLASSES = {
    0: 'Pothole'
}

OUTPUT_DIR = PROJECT_ROOT / 'notebooks' / 'eda_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("📊 KHÁM PHÁ DỮ LIỆU THÔ - POTHOLE DETECTION")
print("=" * 70)

# ============================================================
# 2. KHÁM PHÁ CẤU TRÚC THƯ MỤC
# ============================================================
print("\n" + "=" * 70)
print("📂 2. CẤU TRÚC THƯ MỤC CỦA DATASETS")
print("=" * 70)

def explore_directory(root_path, dataset_name, max_depth=3):
    """In cấu trúc thư mục và đếm file."""
    print(f"\n🗂️  {dataset_name}")
    print(f"   Root: {root_path}")
    print(f"   Exists: {root_path.exists()}")
    
    if not root_path.exists():
        print(f"   ⚠️  Thư mục không tồn tại!")
        return
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        depth = len(Path(dirpath).relative_to(root_path).parts)
        if depth >= max_depth:
            continue
        indent = "   " + "│  " * depth
        dirname = os.path.basename(dirpath)
        n_files = len(filenames)
        n_dirs = len(dirnames)
        print(f"{indent}├── 📁 {dirname}/ ({n_dirs} thư mục, {n_files} files)")

explore_directory(RDD_ROOT, "RDD-2022")
explore_directory(BHARAT_ROOT, "BharatPotHole")


# ============================================================
# 3. PHÂN TÍCH SỐ LƯỢNG ẢNH VÀ NHÃN
# ============================================================
print("\n" + "=" * 70)
print("📊 3. THỐNG KÊ SỐ LƯỢNG ẢNH VÀ NHÃN")
print("=" * 70)

def count_files(directory, extensions):
    """Đếm file theo extension trong thư mục."""
    count = 0
    if directory.exists():
        for ext in extensions:
            count += len(list(directory.glob(f'*{ext}')))
    return count

def get_dataset_stats(root_path, splits, dataset_name):
    """Thu thập thống kê cho mỗi dataset."""
    stats = {}
    for split in splits:
        img_dir = root_path / split / 'images'
        lbl_dir = root_path / split / 'labels'
        
        n_images = count_files(img_dir, ['.jpg', '.jpeg', '.png', '.bmp'])
        n_labels = count_files(lbl_dir, ['.txt'])
        
        stats[split] = {
            'images': n_images,
            'labels': n_labels,
            'img_dir': img_dir,
            'lbl_dir': lbl_dir
        }
        
        print(f"\n   📁 {dataset_name} / {split}:")
        print(f"      🖼️  Ảnh:  {n_images:,}")
        print(f"      🏷️  Nhãn: {n_labels:,}")
        
        if n_images != n_labels:
            diff = abs(n_images - n_labels)
            print(f"      ⚠️  Chênh lệch: {diff} file")
    
    return stats

# RDD-2022
print("\n🔵 RDD-2022:")
rdd_stats = get_dataset_stats(RDD_ROOT, ['train', 'val', 'test'], 'RDD-2022')

# BharatPotHole 
print("\n🟢 BharatPotHole:")
bharat_stats = get_dataset_stats(BHARAT_ROOT, ['train', 'valid', 'test'], 'BharatPotHole')


# ============================================================
# 4. BIỂU ĐỒ SO SÁNH SỐ LƯỢNG ẢNH GIỮA CÁC SPLIT
# ============================================================
print("\n" + "=" * 70)
print("📊 4. BIỂU ĐỒ SO SÁNH SỐ LƯỢNG")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RDD-2022
rdd_splits = list(rdd_stats.keys())
rdd_img_counts = [rdd_stats[s]['images'] for s in rdd_splits]
rdd_lbl_counts = [rdd_stats[s]['labels'] for s in rdd_splits]

x = np.arange(len(rdd_splits))
width = 0.35

bars1 = axes[0].bar(x - width/2, rdd_img_counts, width, label='Images', color='#3498db', edgecolor='white')
bars2 = axes[0].bar(x + width/2, rdd_lbl_counts, width, label='Labels', color='#e74c3c', edgecolor='white')
axes[0].set_title('🔵 RDD-2022 — Số lượng ảnh/nhãn theo split', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels([s.upper() for s in rdd_splits])
axes[0].legend()
axes[0].set_ylabel('Số lượng')
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 50,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)

# BharatPotHole
bharat_splits = list(bharat_stats.keys())
bharat_img_counts = [bharat_stats[s]['images'] for s in bharat_splits]
bharat_lbl_counts = [bharat_stats[s]['labels'] for s in bharat_splits]

x2 = np.arange(len(bharat_splits))
bars3 = axes[1].bar(x2 - width/2, bharat_img_counts, width, label='Images', color='#2ecc71', edgecolor='white')
bars4 = axes[1].bar(x2 + width/2, bharat_lbl_counts, width, label='Labels', color='#f39c12', edgecolor='white')
axes[1].set_title('🟢 BharatPotHole — Số lượng ảnh/nhãn theo split', fontsize=14, fontweight='bold')
axes[1].set_xticks(x2)
axes[1].set_xticklabels([s.upper() for s in bharat_splits])
axes[1].legend()
axes[1].set_ylabel('Số lượng')
for bar in bars3:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)
for bar in bars4:
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                 f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dataset_split_counts.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'dataset_split_counts.png'}")


# ============================================================
# 5. PHÂN TÍCH KÍCH THƯỚC ẢNH
# ============================================================
print("\n" + "=" * 70)
print("📐 5. PHÂN TÍCH KÍCH THƯỚC ẢNH")
print("=" * 70)

def analyze_image_sizes(img_dir, dataset_name, max_samples=2000):
    """Phân tích kích thước ảnh trong thư mục."""
    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    if len(img_files) > max_samples:
        img_files = random.sample(img_files, max_samples)
    
    widths, heights, aspects, file_sizes = [], [], [], []
    
    for f in tqdm(img_files, desc=f"   Đang đọc {dataset_name}", leave=False):
        try:
            with Image.open(f) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w / h)
                file_sizes.append(f.stat().st_size / 1024)  # KB
        except Exception as e:
            pass
    
    return pd.DataFrame({
        'width': widths,
        'height': heights,
        'aspect_ratio': aspects,
        'file_size_kb': file_sizes,
        'dataset': dataset_name
    })

# Lấy stats từ train split
rdd_img_stats = analyze_image_sizes(
    RDD_ROOT / 'train' / 'images', 'RDD-2022'
)
bharat_img_stats = analyze_image_sizes(
    BHARAT_ROOT / 'train' / 'images', 'BharatPotHole'
)

all_img_stats = pd.concat([rdd_img_stats, bharat_img_stats], ignore_index=True)

# In thống kê
for ds_name in ['RDD-2022', 'BharatPotHole']:
    subset = all_img_stats[all_img_stats['dataset'] == ds_name]
    print(f"\n   📐 {ds_name} (train split):")
    print(f"      Kích thước phổ biến:")
    sizes = subset.groupby(['width', 'height']).size().sort_values(ascending=False)
    for (w, h), count in sizes.head(5).items():
        pct = count / len(subset) * 100
        print(f"        {w}x{h}: {count:,} ảnh ({pct:.1f}%)")
    print(f"      File size: {subset['file_size_kb'].mean():.1f} ± {subset['file_size_kb'].std():.1f} KB")
    print(f"      Min: {subset['file_size_kb'].min():.1f} KB | Max: {subset['file_size_kb'].max():.1f} KB")

# Biểu đồ kích thước ảnh
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Width distribution
for ds_name, color in [('RDD-2022', '#3498db'), ('BharatPotHole', '#2ecc71')]:
    subset = all_img_stats[all_img_stats['dataset'] == ds_name]
    axes[0, 0].hist(subset['width'], bins=50, alpha=0.7, label=ds_name, color=color, edgecolor='white')
axes[0, 0].set_title('Phân bố Width', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Width (pixels)')
axes[0, 0].legend()

# Height distribution
for ds_name, color in [('RDD-2022', '#3498db'), ('BharatPotHole', '#2ecc71')]:
    subset = all_img_stats[all_img_stats['dataset'] == ds_name]
    axes[0, 1].hist(subset['height'], bins=50, alpha=0.7, label=ds_name, color=color, edgecolor='white')
axes[0, 1].set_title('Phân bố Height', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Height (pixels)')
axes[0, 1].legend()

# Aspect ratio
for ds_name, color in [('RDD-2022', '#3498db'), ('BharatPotHole', '#2ecc71')]:
    subset = all_img_stats[all_img_stats['dataset'] == ds_name]
    axes[1, 0].hist(subset['aspect_ratio'], bins=50, alpha=0.7, label=ds_name, color=color, edgecolor='white')
axes[1, 0].set_title('Phân bố Aspect Ratio (W/H)', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Aspect Ratio')
axes[1, 0].legend()

# Scatter Width vs Height
for ds_name, color, marker in [('RDD-2022', '#3498db', 'o'), ('BharatPotHole', '#2ecc71', 's')]:
    subset = all_img_stats[all_img_stats['dataset'] == ds_name]
    axes[1, 1].scatter(subset['width'], subset['height'], alpha=0.3, label=ds_name,
                        color=color, marker=marker, s=10)
axes[1, 1].set_title('Width vs Height', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Width (pixels)')
axes[1, 1].set_ylabel('Height (pixels)')
axes[1, 1].legend()

plt.suptitle('📐 Phân tích kích thước ảnh', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'image_size_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'image_size_analysis.png'}")


# ============================================================
# 6. PHÂN TÍCH NHÃN (LABELS) - YOLO FORMAT
# ============================================================
print("\n" + "=" * 70)
print("🏷️  6. PHÂN TÍCH NHÃN (YOLO FORMAT)")
print("=" * 70)

print("""
   📝 YOLO Label Format:
   ──────────────────────────────────────────
   Mỗi dòng trong file .txt đại diện 1 bounding box:
   
   <class_id> <x_center> <y_center> <width> <height>
   
   - class_id: Số nguyên (0, 1, 2, ...)
   - x_center, y_center: Tâm bbox (normalized 0-1)
   - width, height: Kích thước bbox (normalized 0-1)
   
   Ví dụ: "0 0.462891 0.164062 0.054688 0.324219"
   → class 0, tâm (46.3%, 16.4%), kích thước (5.5%, 32.4%)
""")

def parse_yolo_labels(label_dir, class_names):
    """Parse tất cả YOLO label files và trả về DataFrame."""
    all_annotations = []
    empty_files = 0
    total_files = 0
    
    label_files = list(label_dir.glob('*.txt'))
    
    for lbl_file in tqdm(label_files, desc=f"   Parsing labels", leave=False):
        total_files += 1
        with open(lbl_file, 'r') as f:
            lines = f.readlines()
        
        lines = [l.strip() for l in lines if l.strip()]
        
        if len(lines) == 0:
            empty_files += 1
            all_annotations.append({
                'file': lbl_file.stem,
                'class_id': -1,  # empty marker
                'class_name': 'NO_ANNOTATION',
                'x_center': 0, 'y_center': 0,
                'bbox_width': 0, 'bbox_height': 0,
                'n_objects': 0
            })
            continue
        
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_c = float(parts[1])
                y_c = float(parts[2])
                bw = float(parts[3])
                bh = float(parts[4])
                
                cls_name = class_names.get(cls_id, f'Unknown_{cls_id}')
                
                all_annotations.append({
                    'file': lbl_file.stem,
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'x_center': x_c,
                    'y_center': y_c,
                    'bbox_width': bw,
                    'bbox_height': bh,
                    'n_objects': len(lines)
                })
    
    df = pd.DataFrame(all_annotations)
    return df, empty_files, total_files

# Parse RDD-2022 train labels
print("\n🔵 RDD-2022 (train):")
rdd_labels, rdd_empty, rdd_total = parse_yolo_labels(
    RDD_ROOT / 'train' / 'labels', RDD_CLASSES
)
rdd_valid = rdd_labels[rdd_labels['class_id'] >= 0]
print(f"   📄 Tổng file nhãn: {rdd_total:,}")
print(f"   📭 File rỗng (không có object): {rdd_empty:,}")
print(f"   🏷️  Tổng annotations: {len(rdd_valid):,}")
print(f"   📊 Trung bình objects/ảnh: {rdd_valid.groupby('file').size().mean():.2f}")

# Parse BharatPotHole train labels
print("\n🟢 BharatPotHole (train):")
bharat_labels, bharat_empty, bharat_total = parse_yolo_labels(
    BHARAT_ROOT / 'train' / 'labels', BHARAT_CLASSES
)
bharat_valid = bharat_labels[bharat_labels['class_id'] >= 0]
print(f"   📄 Tổng file nhãn: {bharat_total:,}")
print(f"   📭 File rỗng (không có object): {bharat_empty:,}")
print(f"   🏷️  Tổng annotations: {len(bharat_valid):,}")
print(f"   📊 Trung bình objects/ảnh: {bharat_valid.groupby('file').size().mean():.2f}")


# ============================================================
# 7. PHÂN BỐ CLASSES (RDD-2022)
# ============================================================
print("\n" + "=" * 70)
print("📊 7. PHÂN BỐ CLASSES")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# RDD-2022 class distribution
rdd_class_counts = rdd_valid['class_name'].value_counts()
colors_rdd = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#DDA0DD']
wedges1, texts1, autotexts1 = axes[0].pie(
    rdd_class_counts.values,
    labels=rdd_class_counts.index,
    autopct='%1.1f%%',
    colors=colors_rdd[:len(rdd_class_counts)],
    pctdistance=0.85,
    startangle=90,
    wedgeprops={'linewidth': 2, 'edgecolor': 'white'}
)
axes[0].set_title('🔵 RDD-2022 — Phân bố classes\n(train split)', fontsize=14, fontweight='bold')

# In số liệu
print("\n   🔵 RDD-2022 class counts:")
for cls_name, count in rdd_class_counts.items():
    pct = count / len(rdd_valid) * 100
    print(f"      • {cls_name}: {count:,} ({pct:.1f}%)")

# BharatPotHole
bharat_class_counts = bharat_valid['class_name'].value_counts()
bars = axes[1].barh(bharat_class_counts.index, bharat_class_counts.values,
                     color='#2ecc71', edgecolor='white', height=0.5)
axes[1].set_title('🟢 BharatPotHole — Phân bố classes\n(train split)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Số lượng annotations')
for bar in bars:
    axes[1].text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2.,
                 f'{int(bar.get_width()):,}', ha='left', va='center', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'class_distribution.png'}")


# ============================================================
# 8. PHÂN BỐ SỐ OBJECTS TRÊN MỖI ẢNH
# ============================================================
print("\n" + "=" * 70)
print("📊 8. SỐ LƯỢNG OBJECTS TRÊN MỖI ẢNH")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RDD-2022
rdd_obj_per_img = rdd_valid.groupby('file').size()
axes[0].hist(rdd_obj_per_img, bins=range(1, rdd_obj_per_img.max() + 2),
             color='#3498db', edgecolor='white', alpha=0.8, rwidth=0.85)
axes[0].set_title('🔵 RDD-2022 — Objects/ảnh', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Số objects')
axes[0].set_ylabel('Số ảnh')
axes[0].axvline(rdd_obj_per_img.mean(), color='red', linestyle='--',
                label=f'Mean: {rdd_obj_per_img.mean():.2f}')
axes[0].legend()

print(f"\n   🔵 RDD-2022 objects/ảnh:")
print(f"      Min: {rdd_obj_per_img.min()}")
print(f"      Max: {rdd_obj_per_img.max()}")
print(f"      Mean: {rdd_obj_per_img.mean():.2f}")
print(f"      Median: {rdd_obj_per_img.median():.1f}")

# BharatPotHole
bharat_obj_per_img = bharat_valid.groupby('file').size()
axes[1].hist(bharat_obj_per_img, bins=range(1, bharat_obj_per_img.max() + 2),
             color='#2ecc71', edgecolor='white', alpha=0.8, rwidth=0.85)
axes[1].set_title('🟢 BharatPotHole — Objects/ảnh', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Số objects')
axes[1].set_ylabel('Số ảnh')
axes[1].axvline(bharat_obj_per_img.mean(), color='red', linestyle='--',
                label=f'Mean: {bharat_obj_per_img.mean():.2f}')
axes[1].legend()

print(f"\n   🟢 BharatPotHole objects/ảnh:")
print(f"      Min: {bharat_obj_per_img.min()}")
print(f"      Max: {bharat_obj_per_img.max()}")
print(f"      Mean: {bharat_obj_per_img.mean():.2f}")
print(f"      Median: {bharat_obj_per_img.median():.1f}")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'objects_per_image.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'objects_per_image.png'}")


# ============================================================
# 9. PHÂN TÍCH KÍCH THƯỚC BOUNDING BOX
# ============================================================
print("\n" + "=" * 70)
print("📊 9. PHÂN TÍCH KÍCH THƯỚC BOUNDING BOX")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# RDD-2022 bbox sizes
axes[0, 0].scatter(rdd_valid['bbox_width'], rdd_valid['bbox_height'],
                    c=rdd_valid['class_id'], cmap='Set1', alpha=0.3, s=8)
axes[0, 0].set_title('🔵 RDD-2022 — BBox Width vs Height (normalized)',
                      fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Width (normalized)')
axes[0, 0].set_ylabel('Height (normalized)')
axes[0, 0].set_xlim(0, 1)
axes[0, 0].set_ylim(0, 1)

# BharatPotHole bbox sizes
axes[0, 1].scatter(bharat_valid['bbox_width'], bharat_valid['bbox_height'],
                    color='#2ecc71', alpha=0.3, s=8)
axes[0, 1].set_title('🟢 BharatPotHole — BBox Width vs Height (normalized)',
                      fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Width (normalized)')
axes[0, 1].set_ylabel('Height (normalized)')
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)

# RDD-2022 bbox area distribution
rdd_valid_area = rdd_valid['bbox_width'] * rdd_valid['bbox_height']
axes[1, 0].hist(rdd_valid_area, bins=100, color='#3498db', edgecolor='white', alpha=0.8)
axes[1, 0].set_title('🔵 RDD-2022 — Phân bố diện tích BBox', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Area (normalized)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].axvline(rdd_valid_area.mean(), color='red', linestyle='--',
                    label=f'Mean: {rdd_valid_area.mean():.4f}')
axes[1, 0].legend()

# BharatPotHole bbox area distribution
bharat_valid_area = bharat_valid['bbox_width'] * bharat_valid['bbox_height']
axes[1, 1].hist(bharat_valid_area, bins=100, color='#2ecc71', edgecolor='white', alpha=0.8)
axes[1, 1].set_title('🟢 BharatPotHole — Phân bố diện tích BBox', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Area (normalized)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].axvline(bharat_valid_area.mean(), color='red', linestyle='--',
                    label=f'Mean: {bharat_valid_area.mean():.4f}')
axes[1, 1].legend()

plt.suptitle('📐 Phân tích kích thước Bounding Box', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bbox_size_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'bbox_size_analysis.png'}")

# Phân loại bbox theo kích thước
print("\n   📏 Phân loại BBox theo kích thước (RDD-2022):")
small = (rdd_valid_area < 0.01).sum()
medium = ((rdd_valid_area >= 0.01) & (rdd_valid_area < 0.05)).sum()
large = (rdd_valid_area >= 0.05).sum()
total = len(rdd_valid_area)
print(f"      🔸 Small (area < 1%):   {small:,} ({small/total*100:.1f}%)")
print(f"      🔹 Medium (1% - 5%):    {medium:,} ({medium/total*100:.1f}%)")
print(f"      🔷 Large (area > 5%):    {large:,} ({large/total*100:.1f}%)")


# ============================================================
# 10. VỊ TRÍ TÂM BOUNDING BOX (HEATMAP)
# ============================================================
print("\n" + "=" * 70)
print("🗺️  10. VỊ TRÍ TÂM BOUNDING BOX (HEATMAP)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# RDD-2022 heatmap
h1, xedges, yedges, im1 = axes[0].hist2d(
    rdd_valid['x_center'], rdd_valid['y_center'],
    bins=50, cmap='YlOrRd', range=[[0, 1], [0, 1]]
)
axes[0].set_title('🔵 RDD-2022 — Vị trí tâm BBox', fontsize=14, fontweight='bold')
axes[0].set_xlabel('X center (normalized)')
axes[0].set_ylabel('Y center (normalized)')
axes[0].invert_yaxis()
plt.colorbar(im1, ax=axes[0], label='Số annotations')

# BharatPotHole heatmap
h2, xedges2, yedges2, im2 = axes[1].hist2d(
    bharat_valid['x_center'], bharat_valid['y_center'],
    bins=50, cmap='YlGn', range=[[0, 1], [0, 1]]
)
axes[1].set_title('🟢 BharatPotHole — Vị trí tâm BBox', fontsize=14, fontweight='bold')
axes[1].set_xlabel('X center (normalized)')
axes[1].set_ylabel('Y center (normalized)')
axes[1].invert_yaxis()
plt.colorbar(im2, ax=axes[1], label='Số annotations')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bbox_center_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'bbox_center_heatmap.png'}")


# ============================================================
# 11. HIỂN THỊ ẢNH MẪU VỚI BOUNDING BOXES
# ============================================================
print("\n" + "=" * 70)
print("🖼️  11. HIỂN THỊ ẢNH MẪU VỚI BOUNDING BOXES")
print("=" * 70)

# Bảng màu cho các class
CLASS_COLORS = {
    0: '#FF6B6B',  # Longitudinal Crack - Đỏ
    1: '#4ECDC4',  # Transverse Crack - Xanh ngọc
    2: '#45B7D1',  # Alligator Crack - Xanh dương
    3: '#96CEB4',  # Other Corruption - Xanh lá nhạt
    4: '#DDA0DD',  # Pothole - Tím
}

def draw_yolo_bboxes(ax, img, label_path, class_names, title=""):
    """Vẽ ảnh với bounding boxes từ YOLO label file."""
    ax.imshow(img)
    
    h, w = img.shape[:2] if hasattr(img, 'shape') else (img.size[1], img.size[0])
    img_array = np.array(img) if not isinstance(img, np.ndarray) else img
    h, w = img_array.shape[:2]
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            
            cls_id = int(parts[0])
            x_c, y_c, bw, bh = [float(p) for p in parts[1:5]]
            
            # Convert YOLO normalized → pixel coords
            x1 = (x_c - bw / 2) * w
            y1 = (y_c - bh / 2) * h
            box_w = bw * w
            box_h = bh * h
            
            color = CLASS_COLORS.get(cls_id, '#FFFFFF')
            cls_name = class_names.get(cls_id, f'Class {cls_id}')
            
            # Vẽ rectangle
            rect = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth=2.5, edgecolor=color, facecolor=color,
                alpha=0.25
            )
            ax.add_patch(rect)
            
            # Vẽ viền
            rect_border = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth=2.5, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect_border)
            
            # Label text
            ax.text(x1, y1 - 5, cls_name, fontsize=8, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.2',
                    facecolor=color, alpha=0.85, edgecolor='none'))
    
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')


def show_sample_images(img_dir, lbl_dir, class_names, dataset_name,
                       n_samples=12, n_cols=4):
    """Hiển thị grid ảnh mẫu với bounding boxes."""
    img_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
    
    # Ưu tiên chọn ảnh có nhãn (không rỗng)
    labeled_files = []
    for img_f in img_files:
        lbl_f = lbl_dir / (img_f.stem + '.txt')
        if lbl_f.exists():
            with open(lbl_f) as f:
                content = f.read().strip()
            if content:
                labeled_files.append(img_f)
    
    if len(labeled_files) > n_samples:
        selected = random.sample(labeled_files, n_samples)
    else:
        selected = labeled_files[:n_samples]
    
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'🖼️  {dataset_name} — Ảnh mẫu với Bounding Boxes',
                 fontsize=18, fontweight='bold', y=1.01)
    
    for idx, img_file in enumerate(selected):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        img = Image.open(img_file)
        lbl_path = lbl_dir / (img_file.stem + '.txt')
        
        draw_yolo_bboxes(ax, np.array(img), lbl_path, class_names,
                        title=img_file.name[:40])
    
    # Ẩn axes thừa
    for idx in range(len(selected), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    fname = f'sample_images_{dataset_name.lower().replace("-", "").replace(" ", "_")}.png'
    plt.savefig(OUTPUT_DIR / fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"   💾 Đã lưu: {OUTPUT_DIR / fname}")


# Hiển thị ảnh mẫu RDD-2022
random.seed(42)
print("\n🔵 RDD-2022 — Ảnh mẫu:")
show_sample_images(
    RDD_ROOT / 'train' / 'images',
    RDD_ROOT / 'train' / 'labels',
    RDD_CLASSES, 'RDD-2022',
    n_samples=12, n_cols=4
)

# Hiển thị ảnh mẫu BharatPotHole
print("\n🟢 BharatPotHole — Ảnh mẫu:")
show_sample_images(
    BHARAT_ROOT / 'train' / 'images',
    BHARAT_ROOT / 'train' / 'labels',
    BHARAT_CLASSES, 'BharatPotHole',
    n_samples=12, n_cols=4
)


# ============================================================
# 12. PHÂN BỐ KÍCH THƯỚC BBOX THEO CLASS (RDD-2022)
# ============================================================
print("\n" + "=" * 70)
print("📊 12. KÍCH THƯỚC BBOX THEO CLASS (RDD-2022)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Box plot width theo class
class_order = [RDD_CLASSES[i] for i in sorted(RDD_CLASSES.keys())]
available_classes = [c for c in class_order if c in rdd_valid['class_name'].values]

sns.boxplot(data=rdd_valid[rdd_valid['class_name'].isin(available_classes)],
            x='class_name', y='bbox_width', order=available_classes,
            palette=list(CLASS_COLORS.values())[:len(available_classes)],
            ax=axes[0])
axes[0].set_title('Width theo Class', fontsize=13, fontweight='bold')
axes[0].set_xlabel('')
axes[0].tick_params(axis='x', rotation=25)

# Box plot height theo class
sns.boxplot(data=rdd_valid[rdd_valid['class_name'].isin(available_classes)],
            x='class_name', y='bbox_height', order=available_classes,
            palette=list(CLASS_COLORS.values())[:len(available_classes)],
            ax=axes[1])
axes[1].set_title('Height theo Class', fontsize=13, fontweight='bold')
axes[1].set_xlabel('')
axes[1].tick_params(axis='x', rotation=25)

plt.suptitle('📊 RDD-2022 — Kích thước BBox theo từng class', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'bbox_size_by_class.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"   💾 Đã lưu: {OUTPUT_DIR / 'bbox_size_by_class.png'}")


# ============================================================
# 13. PHÂN TÍCH NGUỒN GỐC ẢNH RDD-2022 (từ filename)
# ============================================================
print("\n" + "=" * 70)
print("🌍 13. PHÂN TÍCH NGUỒN GỐC ẢNH RDD-2022")
print("=" * 70)

# RDD-2022 filename pattern: Country_Source_XXXXXX.jpg
rdd_files = list((RDD_ROOT / 'train' / 'images').glob('*.jpg'))
sources = []
for f in rdd_files:
    parts = f.stem.split('_')
    if len(parts) >= 2:
        source = parts[0] + '_' + parts[1]
        sources.append(source)
    else:
        sources.append(f.stem)

source_counts = pd.Series(sources).value_counts()

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.barh(source_counts.index[:15], source_counts.values[:15],
               color=sns.color_palette('viridis', len(source_counts[:15])),
               edgecolor='white')
ax.set_title('🌍 RDD-2022 — Nguồn gốc ảnh (từ filename)', fontsize=16, fontweight='bold')
ax.set_xlabel('Số lượng ảnh')
for bar in bars:
    ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height()/2.,
            f'{int(bar.get_width()):,}', ha='left', va='center', fontsize=10)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'rdd_image_sources.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n   Top nguồn gốc ảnh:")
for src, count in source_counts.head(10).items():
    print(f"      • {src}: {count:,} ảnh")


# ============================================================
# 14. TỔNG KẾT
# ============================================================
print("\n" + "=" * 70)
print("📋 14. TỔNG KẾT KHÁM PHÁ DỮ LIỆU")
print("=" * 70)

total_rdd = sum(rdd_stats[s]['images'] for s in rdd_stats)
total_bharat = sum(bharat_stats[s]['images'] for s in bharat_stats)

print(f"""
   ╔══════════════════════════════════════════════════════════╗
   ║              TỔNG KẾT DATASETS                          ║
   ╠══════════════════════════════════════════════════════════╣
   ║                                                          ║
   ║  🔵 RDD-2022:                                           ║
   ║     • Tổng ảnh: {total_rdd:>8,}                              ║
   ║     • Classes:  5 (cracks + pothole)                     ║
   ║     • Format:   YOLO normalized                          ║
   ║     • Nguồn:    China, India, Japan, ...                 ║
   ║                                                          ║
   ║  🟢 BharatPotHole:                                      ║
   ║     • Tổng ảnh: {total_bharat:>8,}                              ║
   ║     • Classes:  1 (pothole only)                         ║
   ║     • Format:   YOLO normalized (Roboflow)               ║
   ║     • Nguồn:    Dashcam Ấn Độ                            ║
   ║                                                          ║
   ║  📊 Tổng cộng: {total_rdd + total_bharat:>8,} ảnh                        ║
   ║                                                          ║
   ╠══════════════════════════════════════════════════════════╣
   ║  📁 Outputs đã lưu tại:                                 ║
   ║     {str(OUTPUT_DIR):<50} ║
   ╚══════════════════════════════════════════════════════════╝
""")

print("\n   🔍 Các phát hiện chính:")
print("   " + "─" * 50)
print(f"   1. RDD-2022 có 5 class: cracks chiếm đa số, pothole ít hơn")
print(f"   2. BharatPotHole chỉ focus vào pothole (1 class)")
print(f"   3. Cả 2 dataset đều dùng YOLO format (normalized)")
print(f"   4. BBox sizes đa dạng: từ rất nhỏ đến lớn")
print(f"   5. Cần merge/filter phù hợp khi train unified model")

print("\n✅ Hoàn tất khám phá dữ liệu!")
