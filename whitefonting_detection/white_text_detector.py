# whitefonting_detection/white_text_detector.py
import pandas as pd
import numpy as np
import re
import logging
from PIL import Image
import cv2
import io
import fitz  # PyMuPDF
import os

logger = logging.getLogger(__name__)


class WhiteTextDetector:
    """
    Detects hidden white text (whitefonting) in documents.

    This class uses font metadata to detect and analyze hidden white
    text in documents, which is a technique sometimes used to stuff
    keywords into resumes.
    """

    def __init__(self, white_threshold=240, similarity_threshold=0.9):
        """
        Initialize the WhiteTextDetector.

        Args:
            white_threshold (int): RGB threshold for considering text as white.
            similarity_threshold (float): Threshold for background similarity.
        """
        self.white_threshold = white_threshold
        self.similarity_threshold = similarity_threshold
        logger.info("Initializing WhiteTextDetector")

    def detect_white_text(self, font_df):
        """
        Detect white text from font metadata.

        Args:
            font_df (pd.DataFrame): DataFrame containing font metadata.

        Returns:
            pd.DataFrame: DataFrame containing only white text.
        """
        if font_df.empty:
            return pd.DataFrame()

        # Filter for white or near-white text
        white_text_df = font_df[font_df["is_white_or_near_white"]]

        logger.info(
            f"Detected {len(white_text_df)} white text spans out of {len(font_df)} total"
        )

        return white_text_df

    def analyze_white_text(self, white_text_df):
        """
        Analyze detected white text.

        Args:
            white_text_df (pd.DataFrame): DataFrame containing white text.

        Returns:
            dict: Analysis of the white text content.
        """
        if white_text_df.empty:
            return {
                "has_white_text": False,
                "white_text_count": 0,
                "white_text_content": "",
                "common_words": {},
                "potential_keywords": [],
            }

        # Combine all white text
        white_text_content = " ".join(white_text_df["text"])

        # Extract and count words
        words = re.findall(r"\b\w+\b", white_text_content.lower())
        word_counts = pd.Series(words).value_counts().head(20)

        # Identify potential keywords that might be used for keyword stuffing
        # This is a simplified approach - in a real implementation, this would
        # be more sophisticated and possibly use ML to identify relevant keywords
        common_resume_keywords = [
            "machine learning",
            "data science",
            "artificial intelligence",
            "python",
            "java",
            "c++",
            "javascript",
            "react",
            "node",
            "project management",
            "leadership",
            "team",
            "analytics",
            "aws",
            "cloud",
            "database",
            "sql",
            "communication",
            "problem solving",
            "algorithm",
        ]

        keyword_matches = []
        for keyword in common_resume_keywords:
            if keyword.lower() in white_text_content.lower():
                keyword_matches.append(keyword)

        # Determine if keyword stuffing is detected
        # This is a simplified heuristic - a real implementation would be more nuanced
        keyword_stuffing_detected = len(keyword_matches) > 3

        return {
            "has_white_text": True,
            "white_text_count": len(white_text_df),
            "white_text_content": white_text_content,
            "common_words": word_counts.to_dict(),
            "potential_keywords": keyword_matches,
            "keyword_stuffing_detected": keyword_stuffing_detected,
        }

    def create_visual_heatmap(self, document, white_text_df, output_path=None):
        """
        Create a visual heatmap highlighting white text.

        This method creates heatmap visualizations that highlight locations
        of detected white text in the document. These visualizations can be
        saved to disk and/or returned for display.

        Args:
            document (dict): Document object returned by DocumentLoader.
            white_text_df (pd.DataFrame): DataFrame containing white text.
            output_path (str, optional): Path to save the heatmap images.

        Returns:
            list: List of dictionaries containing heatmap images.
        """
        if document["metadata"]["file_type"] != "pdf" or white_text_df.empty:
            logger.warning("Cannot create heatmap: Not a PDF or no white text detected")
            return []

        doc_obj = document["doc_obj"]
        results = []

        for page_num in range(len(doc_obj)):
            page = doc_obj[page_num]

            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert to numpy array for OpenCV processing
            img_np = np.array(img)

            # Create a copy for the heatmap
            heatmap = img_np.copy()

            # Filter white text for the current page
            page_white_text = white_text_df[white_text_df["page"] == page_num + 1]

            # Get text blocks from the page
            blocks = page.get_text("dict")["blocks"]

            # Highlight white text spans
            for index, span_data in page_white_text.iterrows():
                # Find the corresponding block and span
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if (
                                    span["text"] == span_data["text"]
                                    and span["font"] == span_data["font"]
                                    and span["size"] == span_data["size"]
                                ):

                                    # Get the bounding box
                                    bbox = span["bbox"]
                                    x0, y0, x1, y1 = (
                                        int(bbox[0] * 2),
                                        int(bbox[1] * 2),
                                        int(bbox[2] * 2),
                                        int(bbox[3] * 2),
                                    )

                                    # Draw rectangle around the white text
                                    cv2.rectangle(
                                        heatmap, (x0, y0), (x1, y1), (255, 0, 0), 2
                                    )

                                    # Create a semi-transparent overlay
                                    overlay = heatmap.copy()
                                    cv2.rectangle(
                                        overlay, (x0, y0), (x1, y1), (255, 0, 0), -1
                                    )
                                    alpha = 0.3
                                    heatmap = cv2.addWeighted(
                                        overlay, alpha, heatmap, 1 - alpha, 0
                                    )

            # Convert the heatmap back to PIL Image
            heatmap_img = Image.fromarray(heatmap)

            # Save the heatmap if output path is provided
            if output_path:
                import os

                os.makedirs(output_path, exist_ok=True)
                base_filename = os.path.basename(
                    document["metadata"]["file_path"]
                ).replace(".pdf", "")
                output_file = (
                    f"{output_path}/{base_filename}_page{page_num+1}_heatmap.png"
                )
                heatmap_img.save(output_file)
                logger.info(f"Saved heatmap to {output_file}")

            results.append(
                {
                    "page_num": page_num + 1,
                    "original_image": img,
                    "heatmap_image": heatmap_img,
                }
            )

        return results
