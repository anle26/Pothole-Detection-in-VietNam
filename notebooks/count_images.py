"""Count images per street folder and map train/valid/test images back to source folders."""
import os
import json

base = r"D:\Research\Yolo_Pothole_detection\Pothole-Detection-in-VietNam\data\VietNamDataset"

# 1. Collect all images from street-level (source) folders
street_folders = []
source_map = {}  # filename -> street_folder_name

for item in os.listdir(base):
    full = os.path.join(base, item)
    if os.path.isdir(full) and item not in ('train', 'valid', 'test', 'full_dataset'):
        street_folders.append(item)
        for f in os.listdir(full):
            if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                name = os.path.splitext(f)[0]
                source_map[name] = item

print(f"=== Source folders ({len(street_folders)} folders) ===")
for sf in sorted(street_folders):
    count = sum(1 for v in source_map.values() if v == sf)
    print(f"  {sf}: {count} images")

print(f"\nTotal source images: {len(source_map)}")

# 2. Map train/valid/test images back to source folders
def extract_base_id(filename):
    basename = os.path.splitext(os.path.basename(filename))[0]
    if "_jpg" in basename:
        return basename.split("_jpg")[0]
    return basename

results = {}
for split in ['train', 'valid', 'test']:
    img_path = os.path.join(base, split, 'images')
    if not os.path.exists(img_path):
        continue
    
    results[split] = {}
    for f in os.listdir(img_path):
        if not f.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        base_id = extract_base_id(f)
        # Find which source folder this image came from
        if base_id in source_map:
            folder = source_map[base_id]
        else:
            folder = "UNKNOWN"
        
        if folder not in results[split]:
            results[split][folder] = 0
        results[split][folder] += 1

print("\n=== Distribution per split ===")
total_all = 0
overall = {}
for split in ['train', 'valid', 'test']:
    if split not in results:
        continue
    print(f"\n--- {split} ---")
    total = 0
    for folder, count in sorted(results[split].items()):
        print(f"  {folder}: {count}")
        total += count
        if folder not in overall:
            overall[folder] = 0
        overall[folder] += count
    print(f"  TOTAL: {total}")
    total_all += total

print(f"\n=== OVERALL (Total: {total_all}) ===")
for folder, count in sorted(overall.items()):
    pct = count / total_all * 100
    print(f"  {folder}: {count} ({pct:.1f}%)")

# Save as JSON for notebook use
output = {
    "per_split": results,
    "overall": overall,
    "total": total_all,
    "street_folders": sorted(street_folders)
}
with open(os.path.join(os.path.dirname(__file__), "image_distribution.json"), "w") as fp:
    json.dump(output, fp, indent=2, ensure_ascii=False)
print("\nSaved distribution to image_distribution.json")
