
"""
Data structures and enumerations for image segmentation tasks.

This module provides core data structures used throughout the segmentation pipeline,
including segmentation type definitions and bounding box representations.
"""

from enum import Enum
from dataclasses import dataclass

class SegmentationType(Enum):
    """
    Enumeration defining the types of segmentation tasks supported.
    
    Attributes:
        Binary: Binary segmentation (foreground vs background)
        Multiclass: Multi-class segmentation (multiple distinct classes)
    """
    Binary = 0      # Binary segmentation: 0=background, 1=foreground
    Multiclass = 1  # Multi-class segmentation: 0=background, 1...n=classes

@dataclass
class BRect:
    """
    Bounding rectangle data structure for representing rectangular regions.
    
    This class represents a bounding box with top-left corner coordinates
    and dimensions, commonly used for object detection and region annotation.
    
    Attributes:
        x (int): X-coordinate of the top-left corner
        y (int): Y-coordinate of the top-left corner  
        w (int): Width of the rectangle
        h (int): Height of the rectangle
    """
    x: int  # X-coordinate of top-left corner
    y: int  # Y-coordinate of top-left corner
    w: int  # Width of the bounding rectangle
    h: int  # Height of the bounding rectangle