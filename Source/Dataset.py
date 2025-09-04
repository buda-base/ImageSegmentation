"""
PyTorch Dataset classes for binary and multi-class image segmentation.

This module provides custom PyTorch Dataset implementations for loading and
preprocessing image-mask pairs for segmentation training. Includes support
for data augmentation and different normalization strategies.
"""

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from Source.Utils import binarize
from torch.utils.data import Dataset
from Config import COLOR_DICT

import albumentations as A
from albumentations.pytorch import ToTensorV2

class BinaryDataset(Dataset):
    """
    PyTorch Dataset for binary segmentation tasks.
    
    Handles loading and preprocessing of image-mask pairs for binary segmentation
    where masks contain only foreground/background classification. Supports
    data augmentation and multiple normalization strategies.
    
    Attributes:
        images (list[str]): List of image file paths
        masks (list[str]): List of corresponding mask file paths
        normalization_type (int): Normalization strategy (0: divide by 255, 1: cv2.normalize)
        transforms: Albumentations transforms for augmentation
        color_transforms: Additional color space augmentations
    """
    
    def __init__(
        self,
        images: list[str],
        masks: list[str],
        normalization_type: int = 0,
        augmentation_transforms=None,
        color_transforms=None
    ) -> None:
        """
        Initialize the binary segmentation dataset.
        
        Args:
            images (list[str]): List of image file paths
            masks (list[str]): List of mask file paths (must correspond to images)
            normalization_type (int, optional): Normalization method. Defaults to 0.
            augmentation_transforms: Albumentations transforms for data augmentation
            color_transforms: Additional color space augmentations
        """
        super().__init__()

        self.images = images
        self.masks = masks
        self.normalization_type = normalization_type
        self.transforms = augmentation_transforms
        self.color_transforms = color_transforms

    def load_image(self, image_path: str, binarize: bool = False) -> np.array:
        """
        Load and preprocess an image file.
        
        Loads an image, optionally applies binarization, and normalizes
        the pixel values according to the specified normalization type.
        
        Args:
            image_path (str): Path to the image file
            binarize (bool, optional): Whether to apply binarization. Defaults to False.
            
        Returns:
            np.array: Preprocessed image with values normalized to [0, 1]
        """
        image = cv2.imread(image_path)

        if binarize:
            image = binarize(image)

        image = image.astype(np.float32)

        if self.normalization_type == 0:
            # Simple normalization: divide by 255
            image /= 255.0
        else:
            # OpenCV normalization to [0, 1] range
            image = cv2.normalize(
                image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        return image

    def encode_mask(self, y):
        """
        Encode color mask to binary format.
        
        Converts a color mask to binary format where white pixels (255,255,255)
        become foreground (1) and all other pixels become background (0).
        
        Args:
            y: Input color mask
            
        Returns:
            Binary mask with shape (H, W) containing 0s and 1s
        """
        label_map = np.zeros((y.shape[0], y.shape[1], 1), dtype=np.uint8)
        # White pixels become foreground (class 1)
        label_map[np.all(y == [255, 255, 255], axis=-1)] = 1
        label_map = label_map[:, :, 0]  # Remove channel dimension
        return label_map

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single image-mask pair by index.
        
        Loads the image and mask, applies preprocessing and augmentations,
        and returns them in the format expected by PyTorch models.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (image, mask) where image has shape (C, H, W) and mask has shape (H, W)
        """
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load and preprocess image
        x = self.load_image(image_path)
        
        # Load and preprocess mask
        y = cv2.imread(mask_path)
        y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        y = self.encode_mask(y)

        # Apply geometric augmentations to both image and mask
        if self.transforms is not None:
            aug = self.transforms(image=x, mask=y)
            x = aug["image"]
            y = aug["mask"]

        # Apply color augmentations to image only
        if self.color_transforms is not None:
            color_aug = self.color_transforms(image=x, mask=y)
            x = color_aug["image"]

        # Convert image to PyTorch format: (H, W, C) -> (C, H, W)
        x = np.transpose(x, axes=[2, 0, 1])   

        return x, y


class MulticlassDataset(Dataset):
    """
    PyTorch Dataset for multi-class segmentation tasks.
    
    Handles loading and preprocessing of image-mask pairs for multi-class segmentation
    where masks contain multiple distinct classes encoded as different colors.
    Supports data augmentation and automatic tensor conversion.
    
    Attributes:
        images (list[str]): List of image file paths
        masks (list[str]): List of corresponding mask file paths
        classes (dict): Dictionary mapping class names to their properties
        normalization_type (int): Normalization strategy
        base_transforms: Geometric augmentations
        color_transforms: Color space augmentations
    """
    
    def __init__(self, images: list[str], masks: list[str], classes: dict, normalization_type: int = 0, augmentation_transforms=None, color_transforms=None) -> None:
        """
        Initialize the multi-class segmentation dataset.
        
        Args:
            images (list[str]): List of image file paths
            masks (list[str]): List of mask file paths (must correspond to images)
            classes (dict): Dictionary defining the classes for segmentation
            normalization_type (int, optional): Normalization method. Defaults to 0.
            augmentation_transforms: Albumentations transforms for geometric augmentation
            color_transforms: Additional color space augmentations
        """
        super().__init__()

        self.images = images
        self.masks = masks
        self.base_transforms = augmentation_transforms
        self.color_transforms = color_transforms
        self.classes = classes
        self.class_indices = [x for x in range(len(self.classes))]
        self.normalization_type = normalization_type

        # Initialize tensor conversion transform
        self.to_tensor = A.Compose(
            [
                ToTensorV2(),  # Converts numpy arrays to PyTorch tensors
            ]
        )

    def load_image(self, image_path: str, binarize: bool = False):
        """
        Load and preprocess an image file for multi-class segmentation.
        
        Loads an image, converts to grayscale, optionally applies binarization,
        normalizes values, and converts back to 3-channel format required by models.
        
        Args:
            image_path (str): Path to the image file
            binarize (bool, optional): Whether to apply binarization. Defaults to False.
            
        Returns:
            Preprocessed 3-channel image with values normalized to [0, 1]
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if binarize:
            image = binarize(image)

        image = image.astype(np.float32)

        if self.normalization_type == 0:
            # Simple normalization: divide by 255
            image /= 255.0
        else:
            # OpenCV normalization to [0, 1] range
            image = cv2.normalize(
                image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F
            )
        # Convert to 3-channel (most networks require 3 input channels)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        return image
    
    def get_color(self, key: str) -> list[int]:
        """
        Get RGB color values for a class name.
        
        Retrieves the RGB color values associated with a class name from
        the COLOR_DICT configuration and converts to integer list.
        
        Args:
            key (str): Class name to look up
            
        Returns:
            list[int]: RGB color values as [R, G, B] integers
        """
        color = COLOR_DICT[key]
        color = color.split(",")
        color = [x.strip() for x in color]  # Remove whitespace
        color = [int(x) for x in color]      # Convert to integers

        return color
    
    def encode_mask(self, y):
        """
        Encode color mask to class indices for multi-class segmentation.
        
        Converts a color-coded mask where each class has a specific RGB color
        to a class-indexed mask where each pixel contains the class index.
        
        Args:
            y: Input color mask with shape (H, W, 3)
            
        Returns:
            Class-encoded mask with shape (H, W) containing class indices
        """
        label_map = np.zeros(y.shape, dtype=np.uint8)

        # Map each class color to its corresponding index
        for idx, _class in enumerate(self.classes):
            color = self.get_color(_class)
            # Find pixels matching this class color and assign class index
            label_map[np.all(y==color, axis=-1)] = idx
        
        # Return only the first channel (all channels are identical after encoding)
        label_map = label_map[:,:,0]
        return label_map

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Get a single image-mask pair by index.
        
        Loads the image and mask, applies preprocessing and augmentations,
        converts to tensors, and returns them in PyTorch format.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tuple: (image_tensor, mask_tensor) where both are PyTorch tensors
        """
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load and preprocess image
        x = self.load_image(image_path)
        
        # Load and encode mask
        y = cv2.imread(mask_path)
        y = self.encode_mask(y)

        # Apply geometric augmentations to both image and mask
        if self.base_transforms is not None:
            base_aug = self.base_transforms(image=x, mask=y)
            x = base_aug["image"]
            y = base_aug["mask"]

        # Apply color augmentations to image only
        if self.color_transforms is not None:
            color_aug = self.color_transforms(image=x, mask=y)
            x = color_aug["image"]

        # Convert to PyTorch tensors
        tensor_transf = self.to_tensor(image=x, mask=y)

        x = tensor_transf["image"]
        y = tensor_transf["mask"]
        
        return x, y