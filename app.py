import os
import logging
from rich.console import Console
from rich.logging import RichHandler
import argparse
import yaml
import sys
import json
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from whitefonting_detection.semantic_analyzer import SemanticAnalyzer

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console), logging.FileHandler("resume_analysis.log")],
)

logger = logging.getLogger(__name__)


def convert_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_types(i) for i in obj]
    elif hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except:
            return obj
    else:
        return obj


from rich.table import Table
from rich.panel import Panel

def pretty_print_json(data):
    table = Table(title="JSON Data", show_header=True, header_style="bold magenta")
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(str(key), str(value))
    console.print(Panel(table, title="Formatted JSON", border_style="blue"))


class ResumeAnalysisSystem:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.output_dir = self.config.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self._init_modules()
        print("\n")
        logger.info("Resume Analysis System initialized")

    def _load_config(self, config_path):
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

            return {
                "output_dir": "output",
                "whitefonting_detection": {"enabled": True, "min_confidence": 60},
                "classification": {
                    "enabled": True,
                    "model_path": "models/resume_classifier_rf.pkl",
                },
                "turnover_prediction": {
                    "enabled": True,
                    "model_path": "models/turnover_rf_model.pkl",
                },
                "visualization": {"enabled": True, "interactive": True},
            }

    def _init_modules(self):
        from data_processing.document_loader import DocumentLoader
        from data_processing.text_extractor import TextExtractor
        from data_processing.feature_extraction import FeatureExtractor
        from whitefonting_detection.font_analyzer import FontAnalyzer
        from whitefonting_detection.white_text_detector import WhiteTextDetector
        from whitefonting_detection.detection_report import DetectionReportGenerator

        self.document_loader = DocumentLoader()
        self.text_extractor = TextExtractor(
            ocr_enabled=self.config.get("data_processing", {}).get("ocr_enabled", False)
        )
        self.feature_extractor = FeatureExtractor()

        self.font_analyzer = FontAnalyzer()
        self.white_text_detector = WhiteTextDetector(
            white_threshold=self.config.get("whitefonting_detection", {}).get(
                "white_threshold", 240
            )
        )
        self.semantic_analyzer = SemanticAnalyzer()
        self.detection_report_generator = DetectionReportGenerator(
            output_dir=os.path.join(self.output_dir, "whitefonting_reports")
        )

        if self.config.get("classification", {}).get("enabled", True):
            from resume_classification.category_predictor import CategoryPredictor

            model_path = self.config.get("classification", {}).get("model_path")
            self.category_predictor = CategoryPredictor(model_path=model_path)
        else:
            self.category_predictor = None

        if self.config.get("turnover_prediction", {}).get("enabled", True):
            from turnover_prediction.turnover_predictor import TurnoverPredictor

            model_path = self.config.get("turnover_prediction", {}).get("model_path")
            self.turnover_predictor = TurnoverPredictor(model_path=model_path)
        else:
            self.turnover_predictor = None

        if self.config.get("visualization", {}).get("enabled", True):
            from visualization.report_generator import ReportGenerator
            from visualization.data_visualizer import DataVisualizer
            from visualization.dashboard import DashboardGenerator

            self.report_generator = ReportGenerator(
                output_dir=os.path.join(self.output_dir, "reports")
            )
            self.data_visualizer = DataVisualizer(
                output_dir=os.path.join(self.output_dir, "visualizations")
            )
            self.dashboard_generator = DashboardGenerator(
                output_dir=os.path.join(self.output_dir, "dashboards")
            )
        else:
            self.report_generator = None
            self.data_visualizer = None
            self.dashboard_generator = None

    def analyze_resume(
        self,
        file_path,
        generate_visuals=True,
        generate_report=True,
        generate_dashboard=True,
    ):
        logger.debug(f"Analyzing resume: {file_path}")

        try:

            document = self.document_loader.load_document(file_path)

            if "text_content" not in document:
                text_content = self.text_extractor.extract_text(document)
                document["text_content"] = text_content

            features = self.feature_extractor.extract_features(document["text_content"])

            whitefonting_results = self._detect_whitefonting(document, generate_visuals)

            classification_results = self._classify_resume(document["text_content"])

            turnover_results = self._predict_turnover(document["text_content"])

            analysis_results = {
                "document_info": document["metadata"],
                "features": features,
                "whitefonting_detection": whitefonting_results,
                "classification": classification_results,
                "turnover_prediction": turnover_results,
            }

            if whitefonting_results:
                self.detection_report_generator.generate_report(
                    document, whitefonting_results
                )

            if generate_report and self.report_generator:
                report_path = self.report_generator.generate_comprehensive_report(
                    analysis_results, include_visualizations=generate_visuals
                )
                analysis_results["report_path"] = report_path

            if generate_dashboard and self.dashboard_generator:
                dashboard_path = self.dashboard_generator.generate_individual_dashboard(
                    analysis_results
                )
                analysis_results["dashboard_path"] = dashboard_path

            logger.info(f"Resume analysis completed: {file_path}")
            return analysis_results

        except Exception as e:
            logger.error(f"Error analyzing resume {file_path}: {str(e)}")
            return {"error": str(e), "file_path": file_path}

    def batch_analyze(self, directory_path, file_pattern="*.pdf", **kwargs):
        logger.info(
            f"Batch analyzing resumes in {directory_path} matching {file_pattern}"
        )

        file_extensions = []
        for pattern in file_pattern.split(';'):
            extension = os.path.splitext(pattern)[1].lower()
            if extension: 
                file_extensions.append(extension)

        batch_docs = self.document_loader.batch_load(
            directory_path, file_types=file_extensions
        )

        if not batch_docs:
            logger.warning(f"No matching documents found in {directory_path}")
            return [], None

        batch_results = []
        for doc in batch_docs:
            file_path = doc["metadata"]["file_path"]
            try:

                if "text_content" not in doc:
                    text_content = self.text_extractor.extract_text(doc)
                    doc["text_content"] = text_content

                kwargs["generate_report"] = False
                kwargs["generate_dashboard"] = False

                whitefonting_results = self._detect_whitefonting(
                    doc, generate_visuals=False
                )

                classification_results = self._classify_resume(doc["text_content"])

                turnover_results = self._predict_turnover(doc["text_content"])

                result = {
                    "document_info": doc["metadata"],
                    "whitefonting_detection": whitefonting_results,
                    "classification": classification_results,
                    "turnover_prediction": turnover_results,
                }

                batch_results.append(result)

            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                batch_results.append(
                    {"error": str(e), "document_info": {"file_path": file_path}}
                )

        batch_report_path = None
        batch_dashboard_path = None

        if self.report_generator:
            batch_report_path = self.report_generator.generate_batch_report(
                batch_results
            )

        if self.dashboard_generator:
            batch_dashboard_path = self.dashboard_generator.generate_batch_dashboard(
                batch_results
            )

            logger.info(f"Batch analysis completed for {len(batch_results)} resumes")
        return batch_results, batch_report_path, batch_dashboard_path

    def _detect_whitefonting(self, document, generate_visuals=True):

        font_df = self.font_analyzer.analyze_fonts(document)

        font_statistics = self.font_analyzer.get_font_statistics(font_df)

        white_text_df = self.white_text_detector.detect_white_text(font_df)

        white_text_analysis = self.white_text_detector.analyze_white_text(white_text_df)

        semantic_results = self.semantic_analyzer.analyze_semantic_content(
            document, white_text_analysis["white_text_content"]
        )

        heatmap_results = None
        if generate_visuals and white_text_analysis["has_white_text"]:
            heatmap_path = os.path.join(self.output_dir, "heatmaps")
            heatmap_results = self.white_text_detector.create_visual_heatmap(
                document, white_text_df, output_path=heatmap_path
            )

        converted_font_result = convert_types(font_statistics)
        converted_semantic_result = convert_types(semantic_results)
        pretty_print_json(converted_font_result)
        pretty_print_json(white_text_analysis)
        pretty_print_json(converted_semantic_result)

        return {
            "font_statistics": font_statistics,
            "white_text_analysis": white_text_analysis,
            "semantic_analysis": semantic_results,
            "has_white_text": white_text_analysis["has_white_text"],
            "white_text_percentage": font_statistics["white_text_percentage"],
            "white_text_content": white_text_analysis.get("white_text_content", ""),
            "potential_keywords": white_text_analysis.get("potential_keywords", []),
            "heatmap_results": heatmap_results,
        }

    def _classify_resume(self, text_content):
        if not self.category_predictor:
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "category_probabilities": {},
            }

        try:

            if not self.category_predictor.is_model_ready:
                logger.warning(
                    "Classification model not ready, returning default values"
                )
                return {
                    "predicted_category": "Unknown",
                    "confidence": 0.0,
                    "category_probabilities": {},
                }

            prediction = self.category_predictor.predict_category(text_content)
            pretty_print_json(prediction)
            return prediction

        except Exception as e:
            logger.error(f"Error classifying resume: {str(e)}")
            return {
                "predicted_category": "Unknown",
                "confidence": 0.0,
                "category_probabilities": {},
                "error": str(e),
            }

    def _predict_turnover(self, text_content):
        if not self.turnover_predictor:
            return {
                "prediction": {
                    "will_leave": False,
                    "leave_probability": 0.0,
                    "stay_probability": 1.0,
                },
                "employment_pattern": {},
                "contextual_analysis": {
                    "risk_level": "Unknown",
                    "insights": [],
                    "recommendations": [],
                },
            }

        try:
            if not self.turnover_predictor.is_model_ready:
                logger.warning(
                    "Turnover prediction model not ready, using employment analysis only"
                )

                employment_records = self.turnover_predictor.employment_analyzer.extract_employment_history(
                    text_content
                )
                    
                pattern_analysis = self.turnover_predictor.employment_analyzer.analyze_employment_patterns(
                    employment_records
                )

                return {
                    "prediction": {
                        "will_leave": False,
                        "leave_probability": 0.0,
                        "stay_probability": 1.0,
                    },
                    "employment_pattern": pattern_analysis,
                    "contextual_analysis": {
                        "risk_level": "Unknown",
                        "insights": [],
                        "recommendations": [],
                    },
                }

            prediction_results = self.turnover_predictor.predict_turnover(text_content)

            pretty_print_json(prediction_results)

            return prediction_results

        except Exception as e:
            logger.error(f"Error predicting turnover: {str(e)}")
            return {
                "prediction": {
                    "will_leave": False,
                    "leave_probability": 0.0,
                    "stay_probability": 1.0,
                },
                "employment_pattern": {},
                "contextual_analysis": {
                    "risk_level": "Unknown",
                    "insights": [],
                    "recommendations": [],
                },
                "error": str(e),
            }

    def generate_sample_documents(
        self, output_dir="sample_resumes", count=2, include_white_text=True
    ):
        try:

            from whitefonting_detection.white_text_detector import (
                generate_sample_resume,
            )

            os.makedirs(output_dir, exist_ok=True)

            sample_paths = []

            for i in range(count):
                output_path = os.path.join(output_dir, f"normal_resume_{i+1}.pdf")
                generate_sample_resume(output_path, include_white_text=False)
                sample_paths.append(output_path)

            if include_white_text:
                for i in range(count):
                    output_path = os.path.join(
                        output_dir, f"manipulated_resume_{i+1}.pdf"
                    )
                    generate_sample_resume(output_path, include_white_text=True)
                    sample_paths.append(output_path)

            logger.info(f"Generated {len(sample_paths)} sample resumes in {output_dir}")
            return sample_paths

        except Exception as e:
            logger.error(f"Error generating sample documents: {str(e)}")
            return []


def main():
    parser = argparse.ArgumentParser(description="Resume Analysis System")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single resume")
    analyze_parser.add_argument("file_path", help="Path to the resume file")
    analyze_parser.add_argument(
        "--no-visuals", action="store_true", help="Disable visualization generation"
    )
    analyze_parser.add_argument(
        "--no-report", action="store_true", help="Disable report generation"
    )
    analyze_parser.add_argument(
        "--no-dashboard", action="store_true", help="Disable dashboard generation"
    )

    batch_parser = subparsers.add_parser("batch", help="Analyze multiple resumes")
    batch_parser.add_argument("directory", help="Directory containing resume files")
    batch_parser.add_argument(
        "--pattern", default="*.pdf", help='File pattern (e.g., "*.pdf;*.docx")'
    )

    sample_parser = subparsers.add_parser(
        "generate-samples", help="Generate sample resumes"
    )
    sample_parser.add_argument(
        "--count", type=int, default=2, help="Number of sample resumes to generate"
    )
    sample_parser.add_argument(
        "--output-dir", default="sample_resumes", help="Output directory"
    )
    sample_parser.add_argument(
        "--no-white-text", action="store_true", help="Disable whitefonting in samples"
    )

    args = parser.parse_args()

    system = ResumeAnalysisSystem(config_path=args.config)

    if args.command == "analyze":
        results = system.analyze_resume(
            args.file_path,
            generate_visuals=not args.no_visuals,
            generate_report=not args.no_report,
            generate_dashboard=not args.no_dashboard,
        )

        print("\nAnalysis Results:")
        print(f"File: {args.file_path}")
        pretty_print_json({"Document Info": results["document_info"]})

        pretty_print_json({"Features": results["features"]})

        print(
            f"White Text Detected: {'Yes' if results['whitefonting_detection']['has_white_text'] else 'No'}"
        )
        print(
            f"Has Suspicious Content: {results['whitefonting_detection']['semantic_analysis']['has_suspicious_content']}"
        )
        print(f"Classification: {results['classification']['predicted_category']}")
        print(
            f"Turnover Risk: {'High' if results['turnover_prediction']['prediction']['will_leave'] else 'Low'}"
        )

        if "report_path" in results:
            print(f"\nReport generated: {results['report_path']}")

        if "dashboard_path" in results:
            print(f"Interactive dashboard: {results['dashboard_path']}")

    elif args.command == "batch":
        batch_results, batch_report, batch_dashboard = system.batch_analyze(
            args.directory, file_pattern=args.pattern
        )

        console.print(f"\nBatch Analysis Results ({len(batch_results)} resumes):", style="bold cyan")

        white_text_count = sum(
            1
            for r in batch_results
            if r.get("whitefonting_detection", {}).get("has_white_text", False)
        )
        high_risk_count = sum(
            1
            for r in batch_results
            if r.get("turnover_prediction", {})
            .get("prediction", {})
            .get("will_leave", False)
        )

        summary_table = Table(title="Batch Analysis Summary", header_style="bold magenta")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")
        summary_table.add_row("Total Resumes Processed", str(len(batch_results)))
        summary_table.add_row("White Text Detected", f"{white_text_count} ({white_text_count/len(batch_results)*100:.1f}%)")
        summary_table.add_row("High Turnover Risk", f"{high_risk_count} ({high_risk_count/len(batch_results)*100:.1f}%)")

        console.print(summary_table)

        if batch_report:
            console.print(f"\nBatch report generated: [link={batch_report}]{batch_report}[/link]", style="bold yellow")

        if batch_dashboard:
            console.print(f"Interactive dashboard: [link={batch_dashboard}]{batch_dashboard}[/link]", style="bold yellow")

    elif args.command == "generate-samples":
        sample_paths = system.generate_sample_documents(
            output_dir=args.output_dir,
            count=args.count,
            include_white_text=not args.no_white_text,
        )

        if sample_paths:
            print(f"\nGenerated {len(sample_paths)} sample resumes:")
            for path in sample_paths:
                print(f"- {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
