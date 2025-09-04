"""
ONNX-based layout detection for document image analysis.

This module provides the LayoutDetection class for performing inference using
pre-trained ONNX models to detect layout elements in document images. Supports
real-time prediction with configurable thresholds and visualization capabilities.
"""

import os
import cv2
import json
import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from Source.Utils import optimize_countour

class LayoutDetection:
    """
    ONNX-based layout detection engine for document image analysis.
    
    This class provides real-time layout detection capabilities using pre-trained
    ONNX models. It can detect various document elements such as text lines, images,
    margins, and captions in document images with configurable detection thresholds.
    
    The class supports both CPU and GPU execution providers and includes visualization
    capabilities for debugging and result inspection.
    
    Attributes:
        _config_file (str): Path to the model configuration JSON file
        _onnx_model_file (str): Path to the ONNX model file
        _input_width (int): Expected input image width
        _input_height (int): Expected input image height
        _class_dict (dict): Dictionary mapping class names to colors
        _can_run (bool): Whether the model is properly initialized
        _inference: ONNX Runtime inference session
        execution_providers (list): Available execution providers for ONNX Runtime
    """
    
    def __init__(
        self,
        config_file: str,
    ) -> None:
        """
        Initialize the layout detection engine.
        
        Loads the model configuration and initializes the ONNX Runtime session
        with appropriate execution providers (CPU/CUDA).
        
        Args:
            config_file (str): Path to JSON configuration file containing:
                - model: Path to ONNX model file
                - input_width: Expected input image width
                - input_height: Expected input image height
                - classes: Dictionary mapping class names to colors
        """
        self._config_file = config_file
        self._onnx_model_file = None
        self._input_width = 1024
        self._input_height = 320
        self._class_dict = None
        self._can_run = False
        self._inference = None
        
        # Add execution providers (CUDA first for GPU acceleration if available)
        self.execution_providers = ["CPUExecutionProvider", "CUDAExecutionProvider"]

        self._init()

    def _init(self) -> None:
        """
        Initialize the ONNX model and configuration from the config file.
        
        Loads the JSON configuration, extracts model parameters, and creates
        the ONNX Runtime inference session. Sets _can_run flag based on success.
        """
        _file = open(self._config_file)
        json_content = json.loads(_file.read())
        
        # Extract configuration parameters
        self._onnx_model_file = json_content["model"]
        self._input_width = json_content["input_width"]
        self._input_height = json_content["input_height"]
        self._class_dict = json_content["classes"]

        if self._onnx_model_file is not None:
            # Construct full path to model file
            model_file_path = f"{os.path.dirname(self._config_file)}/{self._onnx_model_file}"
            try:
                # Initialize ONNX Runtime session
                self._inference = ort.InferenceSession(
                    model_file_path, providers=self.execution_providers
                )
                self.can_run = True

            except Exception as error:
                print.error(f"Error loading model file: {error}")
                self.can_run = False
        else:
            self.can_run = False

        print(f"Layout Inference -> Init(): {self.can_run}")

    def prepare_img_patches(self, image: np.array) -> np.array:
        """
        Preprocess image patches for model inference.
        
        Converts color images to the format expected by the segmentation model:
        grayscale, normalized, and converted to 3-channel format.
        
        Args:
            image (np.array): Input image patch
            
        Returns:
            np.array: Preprocessed image ready for inference
        """
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float32)
        image /= 255.0  # Normalize to [0, 1]
        # Convert to 3-channel format (many models expect 3 channels)
        image = np.dstack([image, image, image])
        return image

    def get_contours(
        self, prediction: np.array, optimize: bool = True, return_bbox: bool = False
    ) -> list:
        """
        Extract contours from prediction masks.
        
        Processes prediction masks to extract contours representing detected
        layout elements. Optionally optimizes contours to reduce point count.
        
        Args:
            prediction (np.array): Binary prediction mask
            optimize (bool, optional): Whether to optimize contours. Defaults to True.
            return_bbox (bool, optional): Whether to return bounding boxes. Defaults to False.
            
        Returns:
            list: List of contours or bounding boxes for detected elements
        """
        # Threshold prediction to binary mask
        prediction = np.where(prediction > 200, 255, 0)
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            # Find contours in the binary mask
            contours, _ = cv2.findContours(
                prediction, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

            if return_bbox:
                # TODO: implement bounding box extraction
                pass
            elif optimize:
                # Optimize contours to reduce point count
                contours = [optimize_countour(x) for x in contours]
            else:
                return contours
        else:
            return []

    def generate_page_data(
        self, original_image: np.array, predictions: np.array, alpha: float = 0.4
    ) -> np.array:
        
        predictions = cv2.resize(
                predictions, (original_image.shape[1], original_image.shape[0])
            )

        pred_images = predictions[:, :, 0]
        pred_lines = predictions[:, :, 1]
        pred_margin = predictions[:, :, 2]
        pred_caption = predictions[:, :, 3]

        preview_img = np.zeros(
            shape=(predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8
        )

        if np.sum(pred_lines) > 0:
            contours, _ = cv2.findContours(
                pred_lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["line"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        if np.sum(pred_images) > 0:
            contours, _ = cv2.findContours(
                pred_images, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["image"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        if np.sum(pred_margin) > 0:
            contours, _ = cv2.findContours(
                pred_margin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["margin"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        if np.sum(pred_caption) > 0:
            contours, _ = cv2.findContours(
                pred_caption, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            color = tuple([int(x) for x in self._class_dict["caption"].split(",")])

            for idx, _ in enumerate(contours):
                cv2.drawContours(
                    preview_img, contours, contourIdx=idx, color=color, thickness=-1
                )

        preview_img = cv2.resize(
            preview_img,
            (original_image.shape[1], original_image.shape[0]),
        )

        cv2.addWeighted(
            preview_img, alpha, original_image, 1 - alpha, 0, original_image
        )

        return original_image

    def create_preview_image(
        self,
        image: np.array,
        image_predictions: list,
        line_predictions: list,
        caption_predictions: list,
        margin_predictions: list,
        alpha: float = 0.4,
    ) -> np.array:
        # preview_image = image.copy() # huh?
        mask = np.zeros(image.shape, dtype=np.uint8)

        if len(image_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["image"].split(",")])

            for idx, _ in enumerate(image_predictions):
                cv2.drawContours(
                    mask, image_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(line_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["line"].split(",")])

            for idx, _ in enumerate(line_predictions):
                cv2.drawContours(
                    mask, line_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(caption_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["caption"].split(",")])

            for idx, _ in enumerate(caption_predictions):
                cv2.drawContours(
                    mask, caption_predictions, contourIdx=idx, color=color, thickness=-1
                )

        if len(margin_predictions) > 0:
            color = tuple([int(x) for x in self._class_dict["margin"].split(",")])

            for idx, _ in enumerate(margin_predictions):
                cv2.drawContours(
                    mask, margin_predictions, contourIdx=idx, color=color, thickness=-1
                )

        cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)

        return image

    def _predict(self, img_patches: np.array, class_threshold: float) -> np.array:
        """
        Perform model inference on image patches.
        
        Runs the ONNX model on preprocessed image patches and returns
        thresholded predictions for each class.
        
        Args:
            img_patches (np.array): Preprocessed image patches
            class_threshold (float): Threshold for positive class predictions
            
        Returns:
            np.array: Thresholded predictions with shape (H, W, num_classes)
        """
        # Preprocess image patches
        img_batch = [self.prepare_img_patches(x) for x in img_patches]
        img_batch = np.array(img_batch)
        img_bach = np.transpose(img_batch, axes=[0, 3, 1, 2])  # BHWC -> BCHW
        print(f"Input: {img_batch.shape}")
        
        # Run inference
        img_bach = ort.OrtValue.ortvalue_from_numpy(img_bach)
        pred_batch = self._inference.run(None, {"input": img_batch})
        pred_batch = pred_batch[0].numpy()
        print(f"Predictions: {pred_batch.shape}")
        
        # Post-process predictions
        predictions = np.squeeze(predictions[0], axis=0)
        predictions = softmax(predictions, axis=0)  # Apply softmax
        predictions = np.transpose(predictions, axes=[1, 2, 0])  # CHW -> HWC

        predictions = predictions[:, :, 1:]  # Remove background class
        predictions = np.where(predictions > class_threshold, 1.0, 0)  # Threshold
        predictions *= 255  # Scale to 0-255 range

        predictions = predictions.astype(np.uint8)
        return predictions

    def run(self, img_patches, class_threshold: float = 0.6) -> np.array:
        """
        Run layout detection on image patches.
        
        Main inference method that processes image patches and returns
        layout element predictions.
        
        Args:
            img_patches: Input image patches for processing
            class_threshold (float, optional): Detection threshold. Defaults to 0.6.
            
        Returns:
            np.array: Layout element predictions
        """
        return self._predict(img_patches, class_threshold)

    def run_debug(self, img, class_threshold: float = 0.6):
        """
        Run layout detection with visualization for debugging.
        
        Performs layout detection and generates a preview image with
        detected elements overlaid for visual inspection.
        
        Args:
            img: Input image for analysis
            class_threshold (float, optional): Detection threshold. Defaults to 0.6.
            
        Returns:
            tuple: (predictions, preview_image) for analysis and visualization
        """
        predictions = self._predict(img, class_threshold)
        preview_img = self.generate_page_data(img, predictions)

        return predictions, preview_img