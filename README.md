# An Efficient Deep Learning Model for Apple Leaf Disease Detection

## Project Overview

Title：《Apple Disease Leaf Detection Based on Improved YOLOv12》

This project proposes an improved YOLOv12 model for apple leaf disease detection. By integrating a P2 detection branch, a Multi-stage Squeeze and Excite Fusion (MSEF) module, and a Multi-scale Feature Enhancement (MFE) block, it achieves high-precision detection of multiple diseases (Healthy, Scab, Black_rot, Cedar_rust). The model effectively suppresses complex background noise and enhances small lesion detection, providing a lightweight and efficient solution for intelligent orchard management.

## 1. Research Background and Model Positioning

Apple leaf diseases severely affect fruit yield and quality. Traditional detection methods are inefficient and fail to achieve large-scale coverage. This project aims to construct an efficient deep learning model based on an improved YOLOv12 architecture, providing a fast, accurate, and deployable solution for real-time detection of apple leaf diseases in complex orchard environments.

## 2. Core Innovations

1. **P2 Detection Branch:**
   Adds an independent P2 detection branch at the 4x downsampling layer of the backbone, reducing the minimum downsampling factor from 8 to 4 to enhance spatial detail capture for small-scale early lesions.

2. **MSEF (Multi-stage Squeeze and Excite Fusion):**
   Embeds MSEF modules after the P3 and P5 layers. It achieves feature recalibration via spatial and channel dual-branch fusion, suppressing background noise and enhancing weak spectral features of dark lesions (e.g., Scab and Black_rot).

3. **MFE (Multi-scale Feature Enhancement):**
   Introduces the MFE module after the P4 layer. It utilizes multi-branch dilated convolutions (dilation rates of 1, 2, 4, 8) to adaptively fuse multi-scale contextual features, improving robustness to scale variations.

## 3. Efficiency and Accuracy

The model maintains a lightweight design with only **3.52M parameters** and **6.6 GFLOPs**. It achieves **99.4% mAP@0.5** and **96.6% mAP@0.5:0.95** on the self-built dataset, demonstrating superior performance compared to the baseline YOLOv12 and other mainstream detectors. The AdamW optimizer and cosine annealing learning rate scheduling ensure stable training. This makes it highly suitable for deployment on edge devices (e.g., UAVs, mobile terminals) for real-time orchard monitoring.

### 3.1 Experimental Dataset

This study uses two independent datasets:
- **Main Dataset (Self-built):** 7,232 images (Healthy: 1993, Scab: 1991, Black_rot: 1497, Cedar_rust: 1751). Split into train/val/test at a 3:1:1 ratio.
- **Generalization Test Dataset (Open-source):** 3,444 images (Apple Leaf Diseases dataset). Split into train/val/test at a 3:1:1 ratio.


### 3.2 Dataset Structure

The dataset folder is organized as follows:

```plaintext
dataset_new_apple/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
├── dataset_image_counts_single_area.csv
└── dataset_image_counts_single_label.csv
```

A `data.yaml` file is essential for training. It defines the absolute paths to the `train`, `valid`, and `test` directories, along with the mapping of class indices to their names (e.g., `0: Healthy`, `1: Scab`, `2: Black_rot`, `3: Cedar_rust`).

## 4. Experimental Environment Configuration

### 4.1 Dependency Installation

We recommend using Anaconda to create a virtual environment.

```
1. Create and activate a virtual environment
conda create -n yolo12-apple python=3.10
conda activate yolo12-apple

2. Install PyTorch (compatible with CUDA 12.8)
pip install torch==2.7.0 torchvision==0.22.0

3. Install other dependencies
pip install -r requirements.txt
```
### 4.2 Hardware Requirements

OS: Windows 10 x64

Python: 3.10

CUDA: 12.8, cuDNN: 8.7

Framework: Ultralytics

Hardware Tested On: AMD Ryzen 9 8945HX, 32GB RAM, NVIDIA GeForce RTX 5060

## 5. Experimental Results

Performance comparison between the proposed model and baseline YOLOv12:

|Category|mAP@0.5|mAP@0.5:0.95|
|-|-|-|
|YOLOv12|98.6%|94.3%|
|Proposed|99.4%|96.6%|

Note: The MSEF module adds less than 0.2M parameters. The MFE module uses serial dilated convolutions without causing parameter explosion. The P2 detection head directly reuses backbone output, avoiding extra high-resolution extraction parameters.

## 6. Code Usage Instructions

### 6.1 Model Training
Run the train.py script to start training. The model configuration is defined in the YAML file.
```
python main.py
```
Main configuration items (can be modified in main.py):

|Category|Precision (%)|
|-|-|
|data|dataset_new_apple/data.yaml|
|imgsz|640|
|batch|30|
|epochs|100|
|lr0|0.001|
|optimizer|AdamW|

## 7. Project File Structure
```
Apple-Disease-Leaf-Detection/
├── dataset_new_apple/          # Dataset directory (contains train, valid, test)
├── main.py                     # Main training and evaluation script
├── yolov12_P2_MFE_MSEF.yaml    # Improved YOLOv12 network configuration
├── train_batch0.jpg            # Visualization of training batch 0 (Mosaic augmentation)
├── train_batch1.jpg            # Visualization of training batch 1
├── train_batch2.jpg            # Visualization of training batch 2
└── README.md
```
Note: The implementations of the MSEF and MFE modules are integrated within the model configuration and main.py.
## 8. Training Visualization

The following images show the Mosaic augmentation applied during the training process, demonstrating how the model learns to detect small lesions in complex backgrounds:

https://train_batch0.jpg/
https://train_batch1.jpg/
https://train_batch2.jpg/


## 9. Known Issues and Precautions


1.Small Object Detection: The P2 branch significantly improves small lesion detection, but image resolution should be maintained at ≥ 640x640 during acquisition to avoid losing micro-crack features.

2.CUDA Compatibility: If you encounter CUDA incompatibility, ensure your PyTorch version matches your local CUDA version (12.8 recommended).

3.Memory Usage: If you encounter out-of-memory errors, reduce batch size from 32 to 16 or 8.

4.Background Noise: Although MSEF suppresses background noise, images from highly complex natural orchards may require fine-tuning with a few target domain samples.


## 10. References and Contact Information

### 10.1 Reference Method

The paper is currently in the submission stage. Currently available for temporary reference:

```
@article{yolov12_apple_2026,
  title={An Efficient Deep Learning Model for Apple Leaf Disease Detection},
  author={Liu, Chenshuo and ...},
  journal={（待录用后补充）},
  year={2026},
  note={Manuscript submitted for publication}
}
```

### 10.2 Contact Information

If you encounter code running issues or academic exchange needs, please contact:

- Email: liuchenshuohuuc@yeah.net

- GitHub Issue: Submit an issue directly to this repository and we will respond within 1-3 business days.

