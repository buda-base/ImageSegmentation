"""
Configuration file for image segmentation classes and color mappings.

This module defines the color schemes and class configurations used for 
document layout segmentation tasks. Colors are specified in RGB format 
as comma-separated strings for compatibility with mask encoding/decoding.
"""

# Color dictionary mapping document element classes to RGB color values
# Colors are stored as comma-separated strings in "R, G, B" format
# These colors are used for mask encoding/decoding and visualization
COLOR_DICT = {
        "background": "0, 0, 0",        # Black - background/non-content areas
        "image": "45, 255, 0",          # Bright green - image regions
        "text": "255, 243, 0",          # Yellow - main text content
        "margin": "0, 0, 255",          # Blue - margin annotations and notes
        "caption": "255, 100, 243",     # Pink - image captions and descriptions
        "table": "0, 255, 0",           # Green - table structures
        "pagenr": "0, 100, 15",         # Dark green - page numbers
        "header": "255, 0, 0",          # Red - document headers
        "footer": "255, 255, 100",      # Light yellow - document footers
        "line": "0, 100, 255"           # Light blue - text lines and separators
    }


# Class configuration for modern document layouts
# Typically used for contemporary documents with standard layouts
MODERN_CLASSES = [
    "background",   # Non-content background
    "image",        # Images, figures, illustrations
    "line",         # Text lines and line separators
    "header",       # Page headers and titles
    "footer",       # Page footers and page numbers
]

# Class configuration for historical document layouts (PERIG dataset style)
# Optimized for historical manuscripts and documents with margin annotations
PERIG_CLASSES = [
    "background",   # Non-content background
    "image",        # Images, illuminations, figures
    "line",         # Text lines and decorative lines
    "margin",       # Marginal notes and annotations
    "caption"       # Image captions and descriptions
]