import os
# pyright: ignore [reportMissingImports]
import fiftyone as fo
from pathlib import Path

# Đường dẫn tới thư mục data đã xử lý (chứa file dataset.yaml)
PROJECT_ROOT = Path(os.getcwd())
if PROJECT_ROOT.name == 'notebooks':
    PROJECT_ROOT = PROJECT_ROOT.parent

DATASET_DIR = PROJECT_ROOT / 'processed' / 'combined_pothole'

dataset_name = "Pothole-Processed-Dataset"
print(f"Đang tải dataset từ: {DATASET_DIR}")

if fo.dataset_exists(dataset_name):
    print("Dataset đã tồn tại, tiến hành tải trực tiếp (rất nhanh)...")
    dataset = fo.load_dataset(dataset_name)
else:
    print("Quá trình này có thể mất vài phút với 12.5k ảnh...")
    # Tạo dataset mới trong FiftyOne từ chuẩn YOLO
    dataset = fo.Dataset.from_dir(
        dataset_dir=str(DATASET_DIR),
        dataset_type=fo.types.YOLOv5Dataset,
        name=dataset_name,
    )

# In ra thống kê để kiểm tra
print(dataset)

# Khởi chạy giao diện web của FiftyOne
print("\n=> Đang mở FiftyOne. Vui lòng truy cập http://localhost:5151 trên trình duyệt!")
session = fo.launch_app(dataset)
session.wait()
