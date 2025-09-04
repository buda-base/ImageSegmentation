# ImageSegmentation

A comprehensive pipeline for training and deploying binary and multi-class image segmentation networks, specifically designed for document layout analysis tasks.

## Overview

This project provides a complete framework for semantic segmentation of document images, enabling the identification and classification of different document elements such as text regions, images, margins, captions, headers, footers, and lines. The pipeline supports both binary segmentation (foreground/background) and multi-class segmentation for detailed document layout analysis.

## Key Features

- **Dual Training Modes**: Supports both binary and multi-class segmentation training
- **Document-Specific Classes**: Pre-configured for document layout elements (text, images, margins, captions, etc.)
- **Data Augmentation**: Built-in augmentation pipeline for improved model robustness
- **Multiple Loss Functions**: Supports Dice loss, Focal loss, BCE loss, and combined loss functions
- **Model Export**: ONNX export functionality for deployment
- **PageXML Export**: Export predictions in PageXML format for document analysis workflows
- **Layout Detection**: ONNX-based inference pipeline for real-time layout detection

## Architecture

The codebase is organized into several key components:

### 1. Configuration Layer (`Config.py`)
- Defines color mappings for different document element classes
- Contains pre-configured class sets for different document types (MODERN_CLASSES, PERIG_CLASSES)

### 2. Data Layer (`Source/Data.py`, `Source/Dataset.py`)
- **Data Structures**: Enums and dataclasses for segmentation types and bounding boxes
- **Dataset Classes**: Custom PyTorch datasets for binary and multi-class segmentation
- **Data Loading**: Handles image and mask loading with proper preprocessing

### 3. Training Layer (`Source/Trainer.py`)
- **BinarySegmentationTrainer**: Complete training pipeline for binary segmentation tasks
- **MultiSegmentationTrainer**: Training pipeline for multi-class segmentation with early stopping
- **Model Management**: Checkpoint saving/loading, training history tracking, and model configuration export

### 4. Augmentation Layer (`Source/Augmentations.py`)
- Albumentations-based data augmentation pipeline
- Separate transforms for training, validation, and color augmentation

### 5. Utility Layer (`Source/Utils.py`)
- Image processing utilities (binarization, rotation correction, patching)
- Data manipulation functions (dataset splitting, shuffling)
- Visualization tools for samples and overlays
- File and directory management utilities

### 6. Inference Layer (`Source/Layout.py`)
- **LayoutDetection**: ONNX-based inference engine for document layout analysis
- Real-time prediction with configurable class thresholds
- Contour extraction and visualization capabilities

### 7. Export Layer (`Source/Exporter.py`)
- **PageXMLExporter**: Exports segmentation results to PageXML format
- Supports text line detection and reading order generation
- Compatible with Transkribus and other document analysis tools

## Model Architecture

The segmentation models are based on **DeepLabV3Plus** architecture from the `segmentation_models_pytorch` library, which provides:

- **Encoder-Decoder Structure**: Efficient feature extraction and upsampling
- **Atrous Spatial Pyramid Pooling (ASPP)**: Multi-scale feature aggregation
- **Skip Connections**: Preserves fine-grained spatial information
- **Flexible Backbone**: Support for different encoder architectures

### Training Pipeline

1. **Data Preparation**: Images and masks are loaded and preprocessed
2. **Augmentation**: Random augmentations applied during training
3. **Model Training**: Uses combined loss (Dice + Focal) with mixed precision training
4. **Validation**: Regular validation with Dice and Jaccard metrics
5. **Model Export**: Best model saved with ONNX export option

### Loss Functions

- **Dice Loss**: Optimizes for overlap between predicted and ground truth masks
- **Focal Loss**: Addresses class imbalance by focusing on hard examples
- **Combined Loss**: Weighted combination of Dice and Focal losses
- **BCE Loss**: Binary cross-entropy for binary classification tasks

## Usage

### Binary Segmentation
```python
from Source.Trainer import BinarySegmentationTrainer

trainer = BinarySegmentationTrainer(
    train_x=train_images,
    train_y=train_masks,
    valid_x=val_images,
    valid_y=val_masks,
    test_x=test_images,
    test_y=test_masks,
    image_width=512,
    image_height=512,
    batch_size=32
)

trainer.train(epochs=50, model_name="binary_segmentation")
```

### Multi-class Segmentation
```python
from Source.Trainer import MultiSegmentationTrainer
from Config import MODERN_CLASSES

trainer = MultiSegmentationTrainer(
    train_x=train_images,
    train_y=train_masks,
    valid_x=val_images,
    valid_y=val_masks,
    test_x=test_images,
    test_y=test_masks,
    classes=MODERN_CLASSES,
    image_width=512,
    image_height=512
)

trainer.train(epochs=50, patience=5, model_name="multiclass_segmentation")
```

### Layout Detection
```python
from Source.Layout import LayoutDetection

detector = LayoutDetection(config_file="model_config.json")
predictions = detector.run(image_patches, class_threshold=0.6)
```

## Dependencies

- PyTorch (>= 1.7.0)
- segmentation_models_pytorch
- OpenCV (cv2)
- Albumentations
- NumPy
- scikit-learn
- TorchMetrics
- ONNX Runtime
- Matplotlib

## File Structure

```
ImageSegmentation/
├── Config.py                      # Color and class configurations
├── Source/
│   ├── Data.py                    # Data structures and enums
│   ├── Dataset.py                 # PyTorch dataset classes
│   ├── Trainer.py                 # Training pipeline classes
│   ├── Utils.py                   # Utility functions
│   ├── Augmentations.py           # Data augmentation
│   ├── Layout.py                  # ONNX inference engine
│   └── Exporter.py                # PageXML export functionality
├── Demo-TrainBinary.ipynb         # Binary training demonstration
├── Demo-Train_Multiclass.ipynb    # Multi-class training demonstration
└── Download_Dataset.ipynb         # Dataset preparation
```

This pipeline is particularly well-suited for document digitization projects, historical document analysis, and automated document processing workflows.
