import pandas as pd
import numpy as np
import re
import logging
from PIL import Image
import cv2
import io
import fitz
import os

logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")

class WhiteTextDetector:

    def __init__(self, white_threshold=240, similarity_threshold=0.9):
        self.white_threshold = white_threshold
        self.similarity_threshold = similarity_threshold
        logger.info("Initializing WhiteTextDetector")

    def detect_white_text(self, font_df):
        if font_df.empty:
            return pd.DataFrame()

        white_text_df = font_df[font_df["is_white_or_near_white"]]

        logger.info(
            f"Detected {len(white_text_df)} white text spans out of {len(font_df)} total"
        )

        return white_text_df

    def analyze_white_text(self, white_text_df):
        if white_text_df.empty:
            return {
                "has_white_text": False,
                "white_text_count": 0,
                "white_text_content": "",
                "common_words": {},
                "potential_keywords": [],
            }

        white_text_content = " ".join(white_text_df["text"])

        words = re.findall(r"\b\w+\b", white_text_content.lower())
        word_counts = pd.Series(words).value_counts().head(20)

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

        keyword_stuffing_detected = len(keyword_matches) > 3
        
        results = {
            "has_white_text": True,
            "white_text_count": len(white_text_df),
            "white_text_content": white_text_content,
            "common_words": word_counts.to_dict(),
            "potential_keywords": keyword_matches,
            "keyword_stuffing_detected": keyword_stuffing_detected,
        }

        return results

    def create_visual_heatmap(self, document, white_text_df, output_path=None):
        if document["metadata"]["file_type"] != "pdf" or white_text_df.empty:
            logger.warning("Cannot create heatmap: Not a PDF or no white text detected")
            return []

        doc_obj = document["doc_obj"]
        results = []

        for page_num in range(len(doc_obj)):
            page = doc_obj[page_num]

            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            img_np = np.array(img)

            heatmap = img_np.copy()

            page_white_text = white_text_df[white_text_df["page"] == page_num + 1]

            blocks = page.get_text("dict")["blocks"]

            for index, span_data in page_white_text.iterrows():

                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if (
                                    span["text"] == span_data["text"]
                                    and span["font"] == span_data["font"]
                                    and span["size"] == span_data["size"]
                                ):

                                    bbox = span["bbox"]
                                    x0, y0, x1, y1 = (
                                        int(bbox[0] * 2),
                                        int(bbox[1] * 2),
                                        int(bbox[2] * 2),
                                        int(bbox[3] * 2),
                                    )

                                    cv2.rectangle(
                                        heatmap, (x0, y0), (x1, y1), (255, 0, 0), 2
                                    )

                                    overlay = heatmap.copy()
                                    cv2.rectangle(
                                        overlay, (x0, y0), (x1, y1), (255, 0, 0), -1
                                    )
                                    alpha = 0.3
                                    heatmap = cv2.addWeighted(
                                        overlay, alpha, heatmap, 1 - alpha, 0
                                    )

            heatmap_img = Image.fromarray(heatmap)

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
