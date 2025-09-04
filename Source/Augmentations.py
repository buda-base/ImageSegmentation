
"""
Data augmentation utilities for image segmentation training.

This module provides augmentation pipelines using the Albumentations library
to improve model robustness and generalization during training.
"""

import albumentations as A


def get_augmentations(image_width: int, image_height: int):
    """
    Creates augmentation transforms for training, validation, and color enhancement.
    
    This function returns three separate augmentation pipelines:
    1. Base transforms: Geometric augmentations applied during training
    2. Color transforms: Color space augmentations for improved robustness
    3. Validation transforms: Minimal transforms applied during validation
    
    Args:
        image_width (int): Target width for resizing images
        image_height (int): Target height for resizing images
        
    Returns:
        tuple: A tuple containing (base_transforms, color_transforms, val_transforms)
            - base_transforms: Geometric augmentations for training
            - color_transforms: Color space augmentations
            - val_transforms: Minimal transforms for validation
    """
    # Base geometric augmentations applied during training
    # Includes resizing, rotation, and flipping for spatial variation
    base_transforms = A.Compose([
            A.Resize(height=image_height, width=image_width),  # Resize to target dimensions
            A.Rotate(limit=4, p=0.6),        # Small random rotations (±4 degrees) with 60% probability
            A.VerticalFlip(p=0.5)            # Vertical flip with 50% probability
        ])

    # Color space augmentations to improve robustness to lighting variations
    # Applied independently of geometric transforms
    color_transforms = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,        # Random brightness adjustment (±30%)
                contrast_limit=0.3,          # Random contrast adjustment (±30%)
                p=0.5)                       # Applied with 50% probability
        ])

    # Minimal transforms for validation - only resizing to ensure consistent input size
    val_transforms = A.Compose([
            A.Resize(height=image_height, width=image_width),  # Resize only, no augmentation
        ])

    return base_transforms, color_transforms, val_transforms
