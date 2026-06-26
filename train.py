import os
import sys
import torch
from ultralytics import YOLO

# Cấu hình biến môi trường Debug CUDA tránh treo/lỗi bất đồng bộ
os.environ["WANDB_DISABLED"] = "true"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["TORCH_USE_CUDA_DSA"] = "1"

def main():
    # Định dạng lại đường dẫn file cấu hình dataset.yaml
    dataset_yaml = "data/processed/combined_pothole/dataset.yaml"
    if not os.path.exists(dataset_yaml):
        # Dự phòng trường hợp chạy từ thư mục con
        dataset_yaml = "../data/processed/combined_pothole/dataset.yaml"
        
    if not os.path.exists(dataset_yaml):
        print("❌ Không tìm thấy file dataset.yaml. Vui lòng kiểm tra lại thư mục!")
        sys.exit(1)

    print(f"✅ Đã tìm thấy file dataset.yaml tại: {os.path.abspath(dataset_yaml)}")
    
    # Khởi tạo mô hình
    model = YOLO("yolov8s.pt")
    
    print("🔥 Đang chạy tiến trình huấn luyện (Cấu hình an toàn cho Windows)...")
    results = model.train(
        # --- Đường dẫn và đầu ra ---
        data=dataset_yaml,
        project="runs/detect",
        name="base_model",
        exist_ok=True,             

        # --- Siêu tham số tối ưu hiệu năng và an toàn ---
        cache=True,               # Bật cache RAM (Xóa nghẽn cổ chai ổ cứng)
        workers=0,                # Đưa workers về 0 để tránh lỗi CUDA "unspecified launch failure" trên Windows do tràn bộ nhớ chia sẻ
        batch=16,                 # Giữ batch=16 để an toàn VRAM
        imgsz=640,                
        device=0,                 
        amp=True,                 # Bật AMP (Mixed Precision) giúp tăng tốc độ xử lý trên GPU (Tensor Cores) và giảm VRAM

        # --- Tham số thuật toán & Hội tụ ---
        epochs=100,               
        patience=20,              
        optimizer="auto",         

        # --- Augmentation nâng cao cho Domain Generalization ---
        mosaic=1.0,               
        mixup=0.1,                
        close_mosaic=15,          
    )
    print("🎉 Quá trình huấn luyện đã kết thúc thành công!")

if __name__ == '__main__':
    main()
