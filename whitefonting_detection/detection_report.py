# whitefonting_detection/detection_report.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

logger = logging.getLogger(__name__)


class DetectionReportGenerator:
    """
    Generates reports from whitefonting detection results.

    This class creates comprehensive reports based on the results
    of whitefonting detection, including visualizations and analysis.
    """

    def __init__(self, output_dir="reports"):
        """
        Initialize the DetectionReportGenerator.

        Args:
            output_dir (str): Directory where reports will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initializing DetectionReportGenerator (output_dir: {output_dir})")

    def generate_report(self, document, detection_results, include_visualizations=True):
        """
        Generate a comprehensive report of whitefonting detection results.

        Args:
            document (dict): Document object returned by DocumentLoader.
            detection_results (dict): Results from whitefonting detection.
            include_visualizations (bool): Whether to include visualizations.

        Returns:
            str: Path to the generated report.
        """
        # Create a unique filename for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(document["metadata"]["file_path"]).replace(
            ".pdf", ""
        )
        report_filename = f"{base_filename}_whitefonting_report_{timestamp}.pdf"
        report_path = os.path.join(self.output_dir, report_filename)

        # Create the report document
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create a title style
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=16, spaceAfter=12
        )

        # Create a subtitle style
        subtitle_style = ParagraphStyle(
            "Subtitle", parent=styles["Heading2"], fontSize=14, spaceAfter=10
        )

        # Create report elements
        elements = []

        # Add report title
        elements.append(Paragraph("Whitefonting Detection Report", title_style))
        elements.append(Spacer(1, 12))

        # Add document information section
        elements.append(Paragraph("Document Information", subtitle_style))
        elements.append(
            Paragraph(
                f"Filename: {document['metadata']['file_path']}", styles["Normal"]
            )
        )
        elements.append(
            Paragraph(
                f"File Type: {document['metadata']['file_type'].upper()}",
                styles["Normal"],
            )
        )
        elements.append(
            Paragraph(f"Pages: {document['metadata']['page_count']}", styles["Normal"])
        )
        elements.append(
            Paragraph(
                f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 12))

        # Add detection summary
        elements.append(Paragraph("Detection Summary", subtitle_style))

        font_stats = detection_results.get("font_statistics", {})
        white_text_analysis = detection_results.get("white_text_analysis", {})
        has_white_text = white_text_analysis.get("has_white_text", False)

        # Create detection summary table
        summary_data = [
            ["Total Text Spans", str(font_stats.get("total_spans", 0))],
            ["White Text Spans", str(white_text_analysis.get("white_text_count", 0))],
            [
                "White Text Percentage",
                f"{font_stats.get('white_text_percentage', 0):.2f}%",
            ],
            ["Whitefonting Detected", "Yes" if has_white_text else "No"],
            [
                "Keyword Stuffing Detected",
                (
                    "Yes"
                    if white_text_analysis.get("keyword_stuffing_detected", False)
                    else "No"
                ),
            ],
        ]

        # Create the table
        summary_table = Table(summary_data, colWidths=[200, 300])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.black),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    (
                        "BACKGROUND",
                        (0, -1),
                        (-1, -1),
                        colors.pink if has_white_text else colors.white,
                    ),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                ]
            )
        )

        elements.append(summary_table)
        elements.append(Spacer(1, 12))

        # Add white text content if detected
        if has_white_text:
            elements.append(Paragraph("Hidden White Text Content", subtitle_style))
            white_text_content = white_text_analysis.get("white_text_content", "")
            elements.append(Paragraph(white_text_content, styles["Normal"]))
            elements.append(Spacer(1, 12))

            # Add common words section
            elements.append(
                Paragraph("Most Common Words in Hidden Text", subtitle_style)
            )
            common_words = white_text_analysis.get("common_words", {})

            if common_words:
                # Create word count table
                word_count_data = [["Word", "Count"]]
                for word, count in common_words.items():
                    word_count_data.append([word, str(count)])

                # Limit to top 10 words
                word_count_data = word_count_data[:11]

                # Create the table
                word_count_table = Table(word_count_data, colWidths=[150, 100])
                word_count_table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                            ("GRID", (0, 0), (-1, -1), 1, colors.black),
                        ]
                    )
                )

                elements.append(word_count_table)
            else:
                elements.append(Paragraph("No common words found.", styles["Normal"]))

            elements.append(Spacer(1, 12))

            # Add detected keywords section
            elements.append(
                Paragraph("Potential Resume Keywords Detected", subtitle_style)
            )
            keywords = white_text_analysis.get("potential_keywords", [])

            if keywords:
                keyword_text = ", ".join(keywords)
                elements.append(Paragraph(keyword_text, styles["Normal"]))
            else:
                elements.append(
                    Paragraph("No specific resume keywords detected.", styles["Normal"])
                )

            elements.append(Spacer(1, 20))

        # Include visualizations if available and requested
        if include_visualizations and "heatmap_paths" in detection_results:
            elements.append(Paragraph("Visualizations", subtitle_style))
            elements.append(
                Paragraph(
                    "The following images highlight areas where white text was detected:",
                    styles["Normal"],
                )
            )
            elements.append(Spacer(1, 12))

            for heatmap_path in detection_results["heatmap_paths"]:
                if os.path.exists(heatmap_path):
                    # Add the image with a width that fits the page
                    img = Image(heatmap_path, width=450, height=600)
                    elements.append(img)
                    elements.append(Spacer(1, 12))

        # Add recommendations section
        elements.append(Paragraph("Recommendations", subtitle_style))

        if has_white_text:
            recommendations = [
                "This document contains hidden white text, which is often used to manipulate resume screening systems.",
                "The presence of keyword stuffing suggests an attempt to game automated screening algorithms.",
                "We recommend manual review of this resume and possibly contacting the candidate for clarification.",
                "Consider this as a potential red flag in the candidate evaluation process.",
            ]
        else:
            recommendations = [
                "No hidden text detected in this document.",
                "The document appears to be properly formatted without attempts to manipulate screening systems.",
                "Continue with normal evaluation process.",
            ]

        for recommendation in recommendations:
            elements.append(Paragraph(f"• {recommendation}", styles["Normal"]))

        # Build the report
        doc.build(elements)

        logger.info(f"Generated detection report: {report_path}")
        return report_path

    def generate_json_report(self, document, detection_results):
        """
        Generate a JSON report of whitefonting detection results.

        Args:
            document (dict): Document object returned by DocumentLoader.
            detection_results (dict): Results from whitefonting detection.

        Returns:
            str: Path to the generated JSON report.
        """
        # Create a unique filename for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(document["metadata"]["file_path"]).replace(
            ".pdf", ""
        )
        json_filename = f"{base_filename}_whitefonting_report_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)

        # Prepare report data
        report_data = {
            "document_info": {
                "filename": document["metadata"]["file_path"],
                "file_type": document["metadata"]["file_type"],
                "page_count": document["metadata"]["page_count"],
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "detection_results": {
                "has_white_text": detection_results.get("white_text_analysis", {}).get(
                    "has_white_text", False
                ),
                "white_text_count": detection_results.get(
                    "white_text_analysis", {}
                ).get("white_text_count", 0),
                "white_text_percentage": detection_results.get(
                    "font_statistics", {}
                ).get("white_text_percentage", 0),
                "keyword_stuffing_detected": detection_results.get(
                    "white_text_analysis", {}
                ).get("keyword_stuffing_detected", False),
            },
        }

        # Add white text content if detected
        if report_data["detection_results"]["has_white_text"]:
            report_data["white_text_content"] = detection_results.get(
                "white_text_analysis", {}
            ).get("white_text_content", "")
            report_data["common_words"] = detection_results.get(
                "white_text_analysis", {}
            ).get("common_words", {})
            report_data["potential_keywords"] = detection_results.get(
                "white_text_analysis", {}
            ).get("potential_keywords", [])

        # Save the report
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=4)

        logger.info(f"Generated JSON detection report: {json_path}")
        return json_path

    def generate_summary_report(self, batch_results):
        """
        Generate a summary report for multiple documents.

        Args:
            batch_results (list): List of detection results for multiple documents.

        Returns:
            str: Path to the generated summary report.
        """
        # Create a unique filename for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"whitefonting_batch_summary_{timestamp}.pdf"
        summary_path = os.path.join(self.output_dir, summary_filename)

        # Create the report document
        doc = SimpleDocTemplate(summary_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create a title style
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"], fontSize=16, spaceAfter=12
        )

        # Create a subtitle style
        subtitle_style = ParagraphStyle(
            "Subtitle", parent=styles["Heading2"], fontSize=14, spaceAfter=10
        )

        # Create report elements
        elements = []

        # Add report title
        elements.append(Paragraph("Whitefonting Detection Batch Summary", title_style))
        elements.append(Spacer(1, 12))

        # Add batch information
        elements.append(Paragraph("Batch Information", subtitle_style))
        elements.append(
            Paragraph(f"Number of Documents: {len(batch_results)}", styles["Normal"])
        )
        elements.append(
            Paragraph(
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 12))

        # Add summary table
        elements.append(Paragraph("Detection Summary", subtitle_style))

        # Create summary table data
        summary_data = [["Filename", "White Text", "White Text %", "Keyword Stuffing"]]

        for result in batch_results:
            document = result["document"]
            detection = result["detection_results"]

            filename = os.path.basename(document["metadata"]["file_path"])
            has_white_text = detection.get("white_text_analysis", {}).get(
                "has_white_text", False
            )
            white_text_pct = detection.get("font_statistics", {}).get(
                "white_text_percentage", 0
            )
            keyword_stuffing = detection.get("white_text_analysis", {}).get(
                "keyword_stuffing_detected", False
            )

            summary_data.append(
                [
                    filename,
                    "Yes" if has_white_text else "No",
                    f"{white_text_pct:.2f}%",
                    "Yes" if keyword_stuffing else "No",
                ]
            )

        # Create the table
        summary_table = Table(summary_data, colWidths=[200, 80, 80, 120])
        summary_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("GRID", (0, 0), (-1, -1), 1, colors.black),
                    # Highlight rows with white text
                    *[
                        ("BACKGROUND", (0, i + 1), (-1, i + 1), colors.lightpink)
                        for i, row in enumerate(summary_data[1:])
                        if row[1] == "Yes"
                    ],
                ]
            )
        )

        elements.append(summary_table)
        elements.append(Spacer(1, 20))

        # Add statistics
        elements.append(Paragraph("Batch Statistics", subtitle_style))

        # Calculate statistics
        total_docs = len(batch_results)
        docs_with_white_text = sum(
            1
            for r in batch_results
            if r["detection_results"]
            .get("white_text_analysis", {})
            .get("has_white_text", False)
        )
        docs_with_keyword_stuffing = sum(
            1
            for r in batch_results
            if r["detection_results"]
            .get("white_text_analysis", {})
            .get("keyword_stuffing_detected", False)
        )

        # Add statistics text
        elements.append(
            Paragraph(
                f"Documents with White Text: {docs_with_white_text} ({docs_with_white_text/total_docs*100:.1f}%)",
                styles["Normal"],
            )
        )
        elements.append(
            Paragraph(
                f"Documents with Keyword Stuffing: {docs_with_keyword_stuffing} ({docs_with_keyword_stuffing/total_docs*100:.1f}%)",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 12))

        # Build the report
        doc.build(elements)

        logger.info(f"Generated batch summary report: {summary_path}")
        return summary_path

    def _create_visualizations(self, detection_results, output_prefix):
        """
        Create visualizations for detection results.

        Args:
            detection_results (dict): Detection results.
            output_prefix (str): Prefix for output files.

        Returns:
            list: Paths to generated visualization files.
        """
        visualization_paths = []

        # Skip if no white text detected
        if not detection_results.get("white_text_analysis", {}).get(
            "has_white_text", False
        ):
            return visualization_paths

        # Create a bar chart of most common words
        common_words = detection_results.get("white_text_analysis", {}).get(
            "common_words", {}
        )
        if common_words:
            plt.figure(figsize=(10, 6))
            words = list(common_words.keys())[:10]  # Top 10 words
            counts = [common_words[word] for word in words]

            # Create the bar chart
            sns.barplot(x=counts, y=words)
            plt.title("Most Common Words in Hidden Text")
            plt.xlabel("Count")
            plt.tight_layout()

            # Save the figure
            word_chart_path = f"{output_prefix}_common_words.png"
            plt.savefig(word_chart_path)
            plt.close()

            visualization_paths.append(word_chart_path)

        return visualization_paths
