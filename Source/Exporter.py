"""
PageXML export functionality for document layout analysis results.

This module provides the PageXMLExporter class for converting segmentation
results into PageXML format, which is a standard format for document layout
analysis used by tools like Transkribus and other document digitization platforms.
"""

import cv2
import numpy as np
from datetime import datetime
from xml.dom import minidom
import xml.etree.ElementTree as etree
from Source.Utils import Bbox, LayoutData, get_utc_time


class PageXMLExporter:
    """
    Export segmentation results to PageXML format.
    
    PageXML is a standardized XML format for representing document layout
    information, including text regions, reading order, and geometric coordinates.
    This exporter converts layout analysis results into PageXML format compatible
    with Transkribus and other document analysis tools.
    
    Attributes:
        output_dir (str): Directory path for saving exported XML files
    """
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize the PageXML exporter.
        
        Args:
            output_dir (str): Directory where XML files will be saved
        """
        self.output_dir = output_dir

    def get_bbox(self, bbox: Bbox) -> tuple[int, int, int, int]:
        """
        Extract bounding box coordinates from a Bbox object.
        
        Converts a Bbox dataclass to a tuple of coordinates for easier manipulation.
        
        Args:
            bbox (Bbox): Bounding box object
            
        Returns:
            tuple[int, int, int, int]: (x, y, width, height) coordinates
        """
        x = bbox.x
        y = bbox.y
        w = bbox.w
        h = bbox.h

        return (x, y, w, h)

    def get_text_points(self, contour):
        """
        Convert contour points to PageXML coordinate string format.
        
        Transforms OpenCV contour coordinates into the space-separated
        coordinate pairs format required by PageXML.
        
        Args:
            contour: OpenCV contour object
            
        Returns:
            str: Space-separated coordinate pairs in "x1,y1 x2,y2 ..." format
        """
        points = ""
        for box in contour:
            point = f"{box[0][0]},{box[0][1]} "
            points += point
        return points

    def get_bbox_points(self, bbox: tuple[int]):
        """
        Convert bounding box to PageXML coordinate string format.
        
        Creates a coordinate string representing the four corners of a
        rectangular bounding box in PageXML format.
        
        Args:
            bbox (tuple[int]): Bounding box as (x, y, width, height)
            
        Returns:
            str: Four corner coordinates in "x1,y1 x2,y2 x3,y3 x4,y4" format
        """
        x, y, w, h = bbox
        points = f"{x},{y} {x+w},{y} {x+w},{y+h} {x},{y+h}"
        return points

    
    def get_text_line_block(self, coordinate, baseline_points, index, unicode_text):
        """
        Create a PageXML TextLine element with associated metadata.
        
        Generates a complete TextLine XML element including coordinates,
        baseline information, and text content for a single text line.
        
        Args:
            coordinate: Coordinate string for the text line boundary
            baseline_points: Coordinate string for the text baseline
            index: Index number for reading order
            unicode_text: Actual text content of the line
            
        Returns:
            XML element representing the text line
        """
        text_line = etree.Element(
            "Textline", id="", custom=f"readingOrder {{index:{index};}}"
        )
        text_line = etree.Element("TextLine")
        text_line_coords = coordinate

        # Set unique ID and reading order
        text_line.attrib["id"] = f"line_9874_{str(index)}"
        text_line.attrib["custom"] = f"readingOrder {{index: {str(index)};}}"

        # Add coordinate information
        coords_points = etree.SubElement(text_line, "Coords")
        coords_points.attrib["points"] = text_line_coords
        
        # Add baseline information
        baseline = etree.SubElement(text_line, "Baseline")
        baseline.attrib["points"] = baseline_points

        # Add text content
        text_equiv = etree.SubElement(text_line, "TextEquiv")
        unicode_field = etree.SubElement(text_equiv, "Unicode")
        unicode_field.text = unicode_text

        return text_line
    


    def build_xml_document(self,
        image: np.array,
        image_name: str,
        images: tuple[int],
        lines,
        margins: tuple[int],
        captions: tuple[int],
        text_region_bbox: tuple[int],
        text_lines: list[str] | None,
    ):
        """
        Build a complete PageXML document from layout analysis results.
        
        Creates a full PageXML document structure including metadata, page information,
        reading order, and all detected layout elements (text regions, images, margins, etc.).
        
        Args:
            image (np.array): Source image for getting dimensions
            image_name (str): Name of the image file
            images (tuple[int]): Detected image region coordinates
            lines: Detected text line coordinates
            margins (tuple[int]): Detected margin annotation coordinates
            captions (tuple[int]): Detected caption region coordinates
            text_region_bbox (tuple[int]): Main text region bounding box
            text_lines (list[str] | None): Transcribed text for each line (optional)
            
        Returns:
            str: Pretty-formatted XML document string
        """
        # Create root element with PageXML namespace
        root = etree.Element("PcGts")
        root.attrib[
            "xmlns"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        root.attrib["xmlns:xsi"] = "http://www.w3.org/2001/XMLSchema-instance"
        root.attrib[
            "xsi:schemaLocation"
        ] = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

        # Add metadata section
        metadata = etree.SubElement(root, "Metadata")
        creator = etree.SubElement(metadata, "Creator")
        creator.text = "Transkribus"
        created = etree.SubElement(metadata, "Created")
        created.text = get_utc_time()

        # Add page information
        page = etree.SubElement(root, "Page")
        page.attrib["imageFilename"] = image_name
        page.attrib["imageWidth"] = f"{image.shape[1]}"
        page.attrib["imageHeight"] = f"{image.shape[0]}"

        # Add reading order information
        reading_order = etree.SubElement(page, "ReadingOrder")
        ordered_group = etree.SubElement(reading_order, "OrderedGroup")
        ordered_group.attrib["id"] = f"1234_{0}"
        ordered_group.attrib["caption"] = "Regions reading order"

        region_ref_indexed = etree.SubElement(reading_order, "RegionRefIndexed")
        region_ref_indexed.attrib["index"] = "0"
        region_ref = "region_main"
        region_ref_indexed.attrib["regionRef"] = region_ref

        # Add main text region
        text_region = etree.SubElement(page, "TextRegion")
        text_region.attrib["id"] = region_ref
        text_region.attrib["custom"] = "readingOrder {index:0;}"

        text_region_coords = etree.SubElement(text_region, "Coords")
        text_region_coords.attrib["points"] = self.get_bbox_points(text_region_bbox)

        def get_line_baseline(bbox: tuple[int, int, int, int]) -> str:  
            """
            Generate baseline coordinates for a text line bounding box.
            
            Creates baseline coordinates along the bottom edge of the text line,
            which is used for text alignment and reading order in PageXML.
            
            Args:
                bbox (tuple[int, int, int, int]): Text line bounding box
                
            Returns:
                str: Baseline coordinate string
            """
            x, y, w, h = bbox
            return f"{x},{y+h} {x+w},{y+h}"

        # Add text lines to the main text region
        for i in range(0, len(lines)):
            text_coords = self.get_bbox_points(lines[i])
            base_line_coords = get_line_baseline(lines[i])
            
            if text_lines != None and len(text_lines) > 0:
                # Add text line with transcribed content
                text_region.append(
                    self.get_text_line_block(coordinate=text_coords, baseline_points=base_line_coords, index=i, unicode_text=text_lines[i])
                )
            else:
                # Add text line without transcribed content
                text_region.append(self.get_text_line_block(coordinate=text_coords, baseline_points=base_line_coords, index=i, unicode_text=""))

        # Add image regions
        if len(images) > 0:
            for idx, bbox in enumerate(images):
                image_region = etree.SubElement(page, "ImageRegion")
                image_region.attrib["id"] = "Image_1234"
                image_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}}"

                coords_points = etree.SubElement(image_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)

        # Add margin regions
        if len(margins) > 0:
            for idx, bbox in enumerate(margins):
                margin_region = etree.SubElement(page, "TextRegion")
                margin_region.attrib["id"] = f"margin_1234_{idx}"
                margin_region.attrib["type"] = "margin"
                margin_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}} structure {{type:marginalia;}}"

                coords_points = etree.SubElement(margin_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)
        
        # Add caption regions
        if len(captions) > 0:
            for idx, bbox in enumerate(captions):
                captions_region = etree.SubElement(page, "TextRegion")
                captions_region.attrib["id"] = f"caption_1234_{idx}"
                captions_region.attrib["type"] = "caption"
                captions_region.attrib["custom"] = f"readingOrder {{index: {str(idx)};}} structure {{type:caption;}}"

                coords_points = etree.SubElement(captions_region, "Coords")
                coords_points.attrib["points"] = self.get_bbox_points(bbox)
        
        # Format and return the XML document
        xmlparse = minidom.parseString(etree.tostring(root))
        prettyxml = xmlparse.toprettyxml()

        return prettyxml


    def export(self, image: np.array, image_name: str, layout_data: LayoutData, text_lines: list[str]):
        """
        Export layout analysis results to a PageXML file.
        
        Main export function that takes layout analysis results and converts them
        to a complete PageXML document, saving it to the specified output directory.
        
        Args:
            image (np.array): Source image for getting dimensions
            image_name (str): Name of the image file (used for XML filename)
            layout_data (LayoutData): Complete layout analysis results
            text_lines (list[str]): Transcribed text for each detected line
        """
        # Convert layout data to coordinate tuples
        image_boxes = [self.get_bbox(x) for x in layout_data.images]
        caption_boxes = [self.get_bbox(x) for x in layout_data.captions]
        margin_boxes = [self.get_bbox(x) for x in layout_data.margins]
        line_boxes = [self.get_bbox(x.bbox) for x in layout_data.lines]
        text_bbox = self.get_bbox(layout_data.text_bboxes[0])

        # Build the complete XML document
        xml_doc = self.build_xml_document(
            image, 
            image_name, 
            images=image_boxes, 
            lines=line_boxes, 
            margins=margin_boxes, 
            captions=caption_boxes, 
            text_region_bbox=text_bbox, 
            text_lines=text_lines
            )
        
        # Save XML document to file
        xml_out = f"{self.output_dir}/{image_name}.xml"
        with open(xml_out, "w", encoding="utf-8") as f:
            f.write(xml_doc)