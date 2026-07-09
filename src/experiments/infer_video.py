import argparse
import sys
import os
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
    parser = argparse.ArgumentParser(description="Run YOLO pothole detection on a video")
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
        help="Path to the source video"
    )
    parser.add_argument("--save", action="store_true", default=True, help="Save the output video with bounding boxes")
    parser.add_argument("--show", action="store_true", default=False, help="Show the video during inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    source_path = Path(args.source)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    if not source_path.exists():
        print(f"Error: Source video not found at {source_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print(f"Running inference on {source_path}...")
    # Run inference on the video
    # Ensure absolute path to prevent double nesting like runs/detect/runs/detect
    project_path = PROJECT_ROOT / "runs" / "detect"
    
    results = model.track(
        source=str(source_path), 
        save=args.save, 
        show=args.show, 
        conf=args.conf,
        project=str(project_path), 
        name="video_inference", 
        exist_ok=True
    )
    
    print(f"\nInference completed successfully!")
    if args.save:
        print(f"Output video with bounding boxes saved in: runs/detect/video_inference")

if __name__ == "__main__":
    main()
