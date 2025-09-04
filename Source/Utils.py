"""
Utility functions for image processing, data manipulation, and visualization.

This module provides a comprehensive set of utility functions used throughout
the image segmentation pipeline, including image processing operations,
dataset management, visualization tools, and file I/O operations.
"""

import os
import cv2
import math
import random
import logging
import statistics
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from numpy.typing import NDArray
from datetime import datetime
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class Bbox:
    """
    Bounding box data structure for object detection and region annotation.
    
    Attributes:
        x (int): X-coordinate of the top-left corner
        y (int): Y-coordinate of the top-left corner
        w (int): Width of the bounding box
        h (int): Height of the bounding box
    """
    x: int
    y: int
    w: int
    h: int

@dataclass
class LineData:
    """
    Data structure representing a text line with its geometric properties.
    
    Attributes:
        contour (List): Contour points defining the line boundary
        bbox (Bbox): Bounding box encompassing the line
        center (Tuple[int, int]): Center point coordinates of the line
    """
    contour: List
    bbox: Bbox
    center: Tuple[int, int]

@dataclass
class PerigPrediction:
    """
    Container for segmentation predictions from PERIG-style models.
    
    Stores prediction arrays for different document elements typically
    found in historical documents and manuscripts.
    
    Attributes:
        images (NDArray): Predicted image/illustration regions
        lines (NDArray): Predicted text lines
        captions (NDArray): Predicted caption regions
        margins (NDArray): Predicted marginal annotations
    """
    images: NDArray
    lines: NDArray
    captions: NDArray
    margins: NDArray

@dataclass
class LayoutData:
    """
    Comprehensive layout analysis results container.
    
    Stores all detected layout elements and their geometric properties
    for document structure analysis and export.
    
    Attributes:
        images (List[Bbox]): Detected image regions
        text_bboxes (List[Bbox]): Detected text block regions
        lines (List[LineData]): Detected text lines with detailed properties
        captions (List[Bbox]): Detected caption regions
        margins (List[Bbox]): Detected margin annotation regions
        predictions (Dict): Raw prediction data from the model
    """
    images: List[Bbox]
    text_bboxes: List[Bbox]
    lines: List[LineData]
    captions: List[Bbox]
    margins: List[Bbox]
    predictions: Dict


def get_utc_time() -> str:
    """
    Generate UTC timestamp string in ISO format.
    
    Creates a timestamp string suitable for metadata and logging,
    formatted as YYYY-MM-DDTHH:MM:SS.mmm
    
    Returns:
        str: UTC timestamp in ISO format with milliseconds
    """
    t = datetime.now()
    s = t.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    s = s.split(" ")

    return f"{s[0]}T{s[1]}"


def show_sample_pair(img_patch: NDArray, mask_patch: NDArray):
    """
    Display an image and its corresponding mask side by side.
    
    Creates a visualization showing the original image and its segmentation
    mask for visual inspection and debugging purposes.
    
    Args:
        img_patch (NDArray): Input image patch to display
        mask_patch (NDArray): Corresponding segmentation mask
    """
    fig = plt.figure(figsize=(16, 16))
    rows = 1
    columns = 2

    # Display original image
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img_patch)
    plt.axis("off")
    plt.title("Image")

    # Display segmentation mask
    fig.add_subplot(rows, columns, 2)
    plt.imshow(mask_patch)
    plt.axis("off")
    plt.title("Mask")


def show_sample_overlay(img_patch: NDArray, mask_patch: NDArray, alpha: float = 0.4):
    """
    Display an image with its segmentation mask overlaid.
    
    Creates a visualization where the segmentation mask is overlaid on top
    of the original image with transparency for easy comparison.
    
    Args:
        img_patch (NDArray): Input image patch
        mask_patch (NDArray): Segmentation mask to overlay
        alpha (float, optional): Transparency of the mask overlay. Defaults to 0.4.
    """
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Image-Mask overlay")
    plt.imshow(img_patch)
    plt.imshow(mask_patch, alpha=alpha)


def create_dir(dir_name: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Safely creates a directory and any necessary parent directories.
    No error is raised if the directory already exists.
    
    Args:
        dir_name (str): Path of the directory to create
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def get_filename(file_path: str) -> str:
    """
    Extract filename without extension from a file path.
    
    Parses a file path and returns just the filename portion without
    the file extension, handling multiple dots in filenames correctly.
    
    Args:
        file_path (str): Full path to the file
        
    Returns:
        str: Filename without extension
    """
    name_segments = os.path.basename(file_path).split(".")[:-1]
    name = "".join(f"{x}." for x in name_segments)
    return name.rstrip(".")


def shuffle(a, b):
    """
    Shuffle two lists in the same order to maintain correspondence.
    
    Randomly shuffles two lists while preserving the element-wise
    correspondence between them. Useful for shuffling paired data
    like images and their corresponding labels.
    
    Args:
        a: First list to shuffle
        b: Second list to shuffle (must be same length as a)
        
    Returns:
        tuple: Tuple of (shuffled_a, shuffled_b) maintaining correspondence
    """
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)

    return list(a), list(b)


def split_dataset(images: List[str], masks: List[str], train_val_split: float = 0.2, val_test_split: float = 0.5, seed: int = 42):
    """
    Split dataset into training, validation, and test sets.
    
    Performs stratified splitting of image-mask pairs into three sets while
    maintaining correspondence between images and masks. Uses sklearn's
    train_test_split for consistent random splitting.
    
    Args:
        images (List[str]): List of image file paths
        masks (List[str]): List of corresponding mask file paths
        train_val_split (float, optional): Fraction for validation+test split. Defaults to 0.2.
        val_test_split (float, optional): Fraction of val+test to use for test. Defaults to 0.5.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
    Returns:
        tuple: (train_images, train_masks, val_images, val_masks, test_images, test_masks)
    """
    # First split: separate training from validation+test
    train_images, valtest_images, train_masks, valtest_masks = train_test_split(
        images, masks, test_size=train_val_split, random_state=seed)
    
    # Second split: separate validation from test
    val_images, test_images, val_masks, test_masks = train_test_split(
        valtest_images, valtest_masks, test_size=val_test_split, random_state=seed)

    # Shuffle each split to ensure random order
    train_images, train_masks = shuffle(train_images, train_masks)
    val_images, val_masks = shuffle(val_images, val_masks)
    test_images, test_masks = shuffle(test_images, test_masks)

    return train_images, train_masks, val_images, val_masks, test_images, test_masks


def binarize(image: NDArray) -> NDArray:
    """
    Convert an image to binary using adaptive thresholding with contrast enhancement.
    
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) followed
    by adaptive thresholding to create a clean binary image suitable for
    document analysis tasks.
    
    Args:
        image (NDArray): Input color image
        
    Returns:
        NDArray: Binary image converted back to 3-channel RGB format
    """
    # Convert to grayscale for processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(24, 24))
    image = clahe.apply(image)
    
    # Apply adaptive thresholding for binarization
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 11
    )
    
    # Convert back to 3-channel for consistency
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image

def resize_to_width(image: NDArray, target_width: int) -> NDArray:
    """
    Resize an image to a specific width while maintaining aspect ratio.
    
    Calculates the appropriate height to maintain the original aspect ratio
    when resizing to the target width.
    
    Args:
        image (NDArray): Input image to resize
        target_width (int): Desired width in pixels
        
    Returns:
        NDArray: Resized image with maintained aspect ratio
    """
    width_ratio = target_width / image.shape[1]
    image = cv2.resize(
        image,
        (target_width, int(image.shape[0] * width_ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    return image


def rotate_from_hough(image: NDArray) -> Tuple[NDArray, float]:
    """
    Automatically correct image rotation using Hough line detection.
    
    Detects lines in the image using the Hough transform and calculates
    the median angle to determine the rotation correction needed. This is
    particularly useful for correcting skewed document scans.
    
    Args:
        image (NDArray): Input image to rotate
        
    Returns:
        Tuple[NDArray, float]: Tuple of (rotated_image, rotation_angle)
            - rotated_image: The corrected image
            - rotation_angle: The angle used for correction (in degrees)
    """
    # Convert to grayscale and enhance contrast for line detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8,8))
    cl_img = clahe.apply(gray)
    blurred = cv2.GaussianBlur(cl_img, (13, 13), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 11)

    # Apply Hough line detection
    lines = cv2.HoughLinesP(
        thresh,
        1,                    # Distance resolution in pixels
        np.pi / 180,         # Angle resolution in radians
        threshold=130,        # Minimum number of intersections
        minLineLength=40,     # Minimum line length
        maxLineGap=8         # Maximum gap between line segments
    )

    # Handle case where no lines are detected
    if lines is None or len(lines) == 0:
        logging.warning(f"No lines found in image, skipping...")
        return image, 0
    
    prev_img = image.copy()
    angles = []
    zero_angles = []

    # Calculate angles for all detected lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2-y1, x2-x1) * 180 / np.pi

        # Collect small non-zero angles
        if abs(angle) < 5 and abs(angle) > 0:
            angles.append(angle)
        # Collect zero angles separately
        elif int(angle) == 0:
            zero_angles.append(angle)

        # Draw detected lines for debugging (on copy)
        cv2.line(prev_img, (x1, y1), (x2, y2), (100, 100, 0), 3)

    # Calculate rotation angle based on detected line angles
    if len(angles) != 0:     
        avg_angle = statistics.median(angles)
        ratio = (len(zero_angles) / len(angles))

        # Determine rotation angle based on ratio of zero to non-zero angles
        if ratio < 0.5:
            rot_angle = avg_angle
        elif ratio > 0.5 and ratio < 0.9:
            rot_angle = avg_angle / 2
        else:
            rot_angle = 0.0
    else:
        logging.warning("No angle data found in image.")
        rot_angle = 0

    # Apply rotation correction
    rows, cols = image.shape[:2]
    rot_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), (rot_angle), 1)
    rotated_img = cv2.warpAffine(image, rot_matrix, (cols, rows), borderValue=(0, 0, 0))

    return rotated_img, rot_angle



def patch_image(
    img: NDArray, patch_size: int = 64, overlap: int = 2, is_mask=False
) -> tuple[list, int]:
    """
    Divide an image into non-overlapping patches for processing.
    
    Splits an image into square patches of the specified size. The input image
    dimensions should be divisible by patch_size for optimal results.
    
    Args:
        img (NDArray): Input image to patch
        patch_size (int, optional): Size of square patches. Defaults to 64.
        overlap (int, optional): Overlap between patches (currently unused). Defaults to 2.
        is_mask (bool, optional): Whether the image is a mask (currently unused). Defaults to False.
        
    Returns:
        tuple[list, int]: Tuple of (patches_list, y_steps)
            - patches_list: List of image patches
            - y_steps: Number of patches in the vertical direction
    """
    y_steps = img.shape[0] // patch_size
    x_steps = img.shape[1] // patch_size

    patches = []

    for y_step in range(0, y_steps):
        for x_step in range(0, x_steps):
            x_start = x_step * patch_size
            x_end = (x_step * patch_size) + patch_size

            crop_patch = img[
                y_step * patch_size : (y_step * patch_size) + patch_size, x_start:x_end
            ]
            patches.append(crop_patch)

    return patches, y_steps


def unpatch_image(image, pred_patches: list) -> NDArray:
    """
    Reconstruct an image from its patches.
    
    Takes a list of predicted patches and reconstructs the full image
    by stitching them back together in the correct spatial arrangement.
    
    Args:
        image: Reference image (used for dimension calculation)
        pred_patches (list): List of predicted patches to reconstruct
        
    Returns:
        NDArray: Reconstructed full image with pixel values scaled to 0-255
    """
    patch_size = pred_patches[0].shape[1]

    x_step = math.ceil(image.shape[1] / patch_size)

    # Group patches into rows
    list_chunked = [
        pred_patches[i : i + x_step] for i in range(0, len(pred_patches), x_step)
    ]

    final_out = np.zeros(shape=(1, patch_size * x_step))

    # Reconstruct image row by row
    for y_idx in range(0, len(list_chunked)):
        x_stack = list_chunked[y_idx][0]

        # Horizontally stack patches in current row
        for x_idx in range(1, len(list_chunked[y_idx])):
            patch_stack = np.hstack((x_stack, list_chunked[y_idx][x_idx]))
            x_stack = patch_stack

        # Vertically stack completed rows
        final_out = np.vstack((final_out, x_stack))

    # Remove the initial zero row and scale to 0-255
    final_out = final_out[1:, :]
    final_out *= 255

    return final_out


def pad_image(
    img: NDArray, patch_size: int = 64, is_mask=False, pad_value: int = 255
) -> Tuple[NDArray, Tuple[float, float]]:
    """
    Pad an image to make its dimensions divisible by patch_size.
    
    Adds padding to the right and bottom edges of an image to ensure
    the dimensions are multiples of the patch size. Uses different
    padding values for images vs masks.
    
    Args:
        img (NDArray): Input image to pad
        patch_size (int, optional): Target patch size for divisibility. Defaults to 64.
        is_mask (bool, optional): Whether the image is a mask (affects padding color). Defaults to False.
        pad_value (int, optional): Padding value for non-mask images. Defaults to 255 (white).
        
    Returns:
        Tuple[NDArray, Tuple[float, float]]: Tuple of (padded_image, (x_pad, y_pad))
            - padded_image: The padded image
            - (x_pad, y_pad): Amount of padding added in x and y directions
    """
    # Calculate required padding
    x_pad = (math.ceil(img.shape[1] / patch_size) * patch_size) - img.shape[1]
    y_pad = (math.ceil(img.shape[0] / patch_size) * patch_size) - img.shape[0]

    if is_mask:
        # For masks, use black padding (value 0)
        pad_y = np.zeros(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.zeros(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
    else:
        # For images, use specified padding value (typically white)
        pad_y = np.ones(shape=(y_pad, img.shape[1], 3), dtype=np.uint8)
        pad_x = np.ones(shape=(img.shape[0] + y_pad, x_pad, 3), dtype=np.uint8)
        pad_y *= pad_value
        pad_x *= pad_value

    # Apply padding: first vertically, then horizontally
    img = np.vstack((img, pad_y))
    img = np.hstack((img, pad_x))

    return img, (x_pad, y_pad)


def unpatch_prediction(prediction: NDArray, y_splits: int) -> NDArray:
    """
    Reconstruct a prediction image from patches using array operations.
    
    Efficiently reconstructs a full prediction image from an array of patches
    by splitting and concatenating along the appropriate axes.
    
    Args:
        prediction (NDArray): Array of patches to reconstruct
        y_splits (int): Number of rows of patches
        
    Returns:
        NDArray: Reconstructed prediction image with values scaled to 0-255
    """
    # Scale predictions to 0-255 range
    prediction *= 255
    
    # Split patches into rows
    prediction_sliced = np.array_split(prediction, y_splits, axis=0)
    
    # Concatenate patches horizontally within each row
    prediction_sliced = [np.concatenate(x, axis=1) for x in prediction_sliced]
    
    # Stack rows vertically to form complete image
    prediction_sliced = np.vstack(np.array(prediction_sliced))

    return prediction_sliced


def load_image(img_path: str) -> Tuple[NDArray, NDArray]:
    """
    Load an image from file with error handling.
    
    Safely loads an image file using OpenCV with proper error handling
    and logging for debugging purposes.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        Tuple[NDArray, NDArray]: Loaded image array (returns just the image, 
                                 despite tuple annotation which appears to be a legacy)
    """
    try:
        img = cv2.imread(img_path, 1)
        return img
    except BaseException as e:
        logging.error(f"Failed to load image: {img_path}, {e}")


def optimize_countour(cnt, e=0.001):
    """
    Optimize contour by approximating with fewer points.
    
    Uses Douglas-Peucker algorithm to reduce the number of points in a contour
    while preserving its general shape. Useful for reducing memory usage and
    improving processing speed.
    
    Args:
        cnt: Input contour to optimize
        e (float, optional): Approximation accuracy factor. Defaults to 0.001.
        
    Returns:
        Optimized contour with fewer points
    """
    epsilon = e * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)


def prepare_img_patches(image: NDArray) -> NDArray:
    """
    Prepare image patches for neural network inference.
    
    Converts a color image to the format expected by segmentation models:
    grayscale, normalized to [0, 1], and converted back to 3-channel format.
    
    Args:
        image (NDArray): Input color image
        
    Returns:
        NDArray: Preprocessed image ready for model inference
    """
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Normalize to [0, 1] range
    image = image.astype(np.float32)
    image /= 255.0
    # Convert back to 3-channel format (models expect 3 channels)
    image = np.dstack([image, image, image])
    return image
