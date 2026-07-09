# YOLOv8-SimAM: Enhanced Pothole Detection in Vietnam

This repository contains the official PyTorch implementation for our research on applying attention mechanisms to object detection models for road damage assessment. Specifically, we integrate the **Parameter-Free Spatial Attention Module (SimAM)** into **YOLOv8** to improve pothole detection under varying lighting and environmental conditions in Vietnam.

## 📌 Key Contributions
- **Attention Integration**: Dynamically monkey-patches the YOLOv8 `C2f` block with `C2f_SimAM` without altering the core `ultralytics` source code.
- **Robust Baseline Comparison**: Reproducible training pipelines for both standard YOLOv8s and YOLOv8s+SimAM with perfectly mirrored hyperparameters for fair comparison.
- **Comprehensive Ablation Study**: A robust evaluation suite (`Zero-shot`, `Few-shot`, and `5-Fold LP-FT`) directly applied to the Vietnam Dataset to validate domain adaptation capabilities.

## 📂 Project Structure
```text
├── configs/            # Configuration files (Hyperparameters, dataset mappings)
├── data/               # Raw and processed datasets (RDD2022, BharatPotHole, Vietnam)
├── models/             # Downloaded weights and pre-trained checkpoints
├── notebooks/          # Jupyter Notebooks for EDA and Data Preparation
├── reports/            # Output figures (F1 curves, Confusion Matrices) and metrics
├── runs/               # YOLO training logs, weights, and evaluation outputs
├── src/                # Core implementation source code
│   ├── models/         # Architecture definitions (e.g., simam_module.py)
│   ├── experiments/    # Scripts for baseline, SimAM training, and fine-tuning
└── requirements.txt    # Python dependencies
```

## 🛠️ Installation
Ensure you have Python 3.8+ and CUDA available. Clone the repository and install the dependencies:
```bash
git clone https://github.com/anle26/Pothole-Detection-in-VietNam.git
cd Pothole-Detection-in-VietNam
pip install -r requirements.txt
```

## 📊 Datasets
Our experiments utilize three distinct datasets:
1. **RDD-2022**: Global Road Damage Detection challenge dataset.
2. **BharatPotHole**: Focused pothole dataset.
3. **Vietnam Dataset**: Local dataset used exclusively for domain adaptation (Fine-tuning & Ablation Study).

*Note: Raw datasets should be placed inside `data/raw/` and processed into `data/processed/` using the Jupyter notebooks provided in `notebooks/`.*

## 🚀 Pre-training Phase
To train the models on the combined source datasets (RDD-2022 + BharatPotHole) to learn generalized pothole features:

**1. Train YOLOv8 Baseline:**
```bash
python src/experiments/train_baseline.py
```

**2. Train YOLOv8 + SimAM (Proposed Method):**
```bash
python src/experiments/train_simam.py
```

## 🔬 Evaluation & Fine-Tuning (Target Domain: Vietnam)
We evaluate the pre-trained models on the target **Vietnam Dataset** across three rigorous scenarios:
- **Zero-Shot Transfer** (Direct evaluation without target training data)
- **Few-Shot Learning** (Linear Probing followed by Fine Tuning on 50 samples)
- **5-Fold Cross Validation** (Full LP-FT on the entire dataset)

Run the automated evaluation suites (uses multiprocessing to prevent CUDA memory leaks):
```bash
# Run full evaluation suite for Baseline model
python src/experiments/finetune_vietnam_baseline.py --task all

# Run full evaluation suite for Proposed SimAM model
python src/experiments/finetune_vietnam_simam.py --task all
```

To summarize and compare the final validation metrics (mAP, Precision, Recall):
```bash
python src/experiments/evaluate_vietnam_comparison.py
```

## 📈 Results
Evaluation metrics and efficiency profiling (Parameters, GFLOPs, Inference Latency) are automatically aggregated in the `reports/` directory as PNG curves and CSV files for easy integration into research publications.

## 📜 License
This project is licensed under the [MIT License](LICENSE).
