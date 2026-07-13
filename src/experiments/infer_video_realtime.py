import argparse
import sys
import os
import cv2
import time
from pathlib import Path
from ultralytics import YOLO

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import SimAM modules and helper
from src.models.simam_module import apply_simam_patch, SimAM, C2f_SimAM

# Monkey-patching
apply_simam_patch()

# Ensure it's available in __main__ for torch.load compatibility
sys.modules['__main__'].C2f_SimAM = C2f_SimAM
sys.modules['__main__'].SimAM = SimAM

def main():
    parser = argparse.ArgumentParser(description="Run YOLO pothole detection real-time")
    parser.add_argument(
        "--model", 
        type=str, 
        default=os.path.join(str(PROJECT_ROOT), "runs", "detect", "yolov8s_simam", "weights", "best.pt"), 
        help="Path to the model weights (default uses best.pt from yolov8s_simam)"
    )
    parser.add_argument(
        "--source", 
        type=str, 
        default=os.path.join(str(PROJECT_ROOT), "data", "VietNamVideo", "20251108_Quốc Lộ 62 con đường đau khổ nhất miền tây_clip_004.mp4"), 
        help="Path to the source video or camera index (e.g., 0)"
    )
    parser.add_argument("--show", action="store_true", default=True, help="Show the video during inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    
    if str(args.source).isdigit():
        source_path = int(args.source)
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"Error: Source video not found at {source_path}")
            sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"🚀 Running Real-time inference on {source_path}...")
    
    cap = cv2.VideoCapture(str(source_path) if isinstance(source_path, Path) else source_path)
    if not cap.isOpened():
        print("❌ Lỗi: Không mở được video hoặc camera!")
        sys.exit(1)

    prev_time = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # Dùng predict thay vì track. verbose=False để tắt log rác trên Terminal làm chậm máy
        results = model.predict(frame, conf=args.conf, verbose=False, device=0)
        annotated_frame = results[0].plot()
        
        # Tính toán FPS thực tế
        curr_time = time.time()
        fps_live = 1 / (curr_time - prev_time)
        prev_time = curr_time
        
        # Vẽ FPS lên góc màn hình (màu đỏ, to, rõ)
        cv2.putText(annotated_frame, f"FPS: {int(fps_live)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        if args.show:
            cv2.imshow("YOLOv8-SimAM Real-time Inference", annotated_frame)
            # Dùng waitKey(1) để chuyển frame mượt mà ngay lập tức
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n✅ Real-time inference completed!")

if __name__ == "__main__":
    main()
