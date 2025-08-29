import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FontAnalyzer:
    def __init__(self):
        logger.info("Initializing FontAnalyzer")

    def analyze_fonts(self, document):
        if document["metadata"]["file_type"] != "pdf":
            raise ValueError("Font analysis is only supported for PDF documents")

        doc_obj = document["doc_obj"]
        text_with_metadata = []

        logger.info(f"Analyzing fonts in {document['metadata']['file_path']}")

        for page_num in range(len(doc_obj)):
            page = doc_obj[page_num]
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            font_name = span["font"]
                            font_size = span["size"]
                            color = span["color"]

                            r = (color >> 16) & 0xFF
                            g = (color >> 8) & 0xFF
                            b = color & 0xFF

                            is_white_or_near_white = r > 240 and g > 240 and b > 240

                            is_similar_to_background = is_white_or_near_white

                            text_with_metadata.append(
                                {
                                    "page": page_num + 1,
                                    "text": text,
                                    "font": font_name,
                                    "size": font_size,
                                    "r": r,
                                    "g": g,
                                    "b": b,
                                    "color_hex": f"#{r:02x}{g:02x}{b:02x}",
                                    "is_white_or_near_white": is_white_or_near_white,
                                    "is_similar_to_background": is_similar_to_background,
                                }
                            )

        font_df = pd.DataFrame(text_with_metadata)
        logger.info(f"Analyzed {len(font_df)} text spans")

        return font_df

    def get_font_statistics(self, font_df):
        if font_df.empty:
            return {
                "total_spans": 0,
                "unique_fonts": 0,
                "unique_sizes": 0,
                "unique_colors": 0,
                "white_text_percentage": 0,
            }

        total_spans = len(font_df)

        unique_fonts = font_df["font"].nunique()
        unique_sizes = font_df["size"].nunique()
        unique_colors = font_df["color_hex"].nunique()

        white_text_count = font_df["is_white_or_near_white"].sum()
        white_text_percentage = (
            (white_text_count / total_spans) * 100 if total_spans > 0 else 0
        )

        color_stats = font_df.groupby("color_hex").size().reset_index(name="count")
        color_stats["percentage"] = (color_stats["count"] / total_spans) * 100
        color_stats = color_stats.sort_values("count", ascending=False)

        color_distribution = color_stats.to_dict("records")

        return {
            "total_spans": total_spans,
            "unique_fonts": unique_fonts,
            "unique_sizes": unique_sizes,
            "unique_colors": unique_colors,
            "white_text_count": white_text_count,
            "white_text_percentage": white_text_percentage,
            "color_distribution": color_distribution,
        }
