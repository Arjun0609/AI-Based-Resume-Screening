import os
import logging
import re
import shutil
from rich.console import Console
from rich.logging import RichHandler
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import argparse
import yaml
import sys
import json
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import psutil
import time
import uuid
import nltk
from dotenv import load_dotenv

from whitefonting_detection.semantic_analyzer import SemanticAnalyzer

console = Console()

load_dotenv()

# Set up console logging for general messages
console_handler = RichHandler(console=console)
console_handler.setLevel(logging.INFO)

# Set up root logger for console only
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[console_handler],
)

analysis_logger = logging.getLogger("analysis")
analysis_logger.setLevel(logging.INFO)
analysis_file_handler = logging.FileHandler("resume_analysis.log")
analysis_file_handler.setFormatter(logging.Formatter("%(message)s"))
analysis_logger.addHandler(analysis_file_handler)
analysis_logger.propagate = False  

logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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

def pretty_print_table(data, title="Data Table"):
    table = Table(title=title, show_header=True, header_style="bold magenta", title_align="left")
    if data and isinstance(data, list) and isinstance(data[0], dict):
        for key in data[0].keys():
            table.add_column(key, style="cyan")
        for row in data:
            table.add_row(*[str(row.get(key, "")) for key in data[0].keys()])
    console.print(table)

class ResumeAnalysisSystem:
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.output_dir = self.config.get("output_dir", "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self._init_modules()
        print("\n")
        logger.info("Resume Analysis System initialized")

    def allowed_file(self, filename):
        """Check if file has an allowed extension"""
        ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt'}
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
        
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet')
        ]
    
        for data_path, download_name in required_data:
            try:
                nltk.data.find(data_path)
                logger.info(f"✓ Found {download_name}")
            except LookupError:
                logger.info(f"Downloading {download_name}...")
                try:
                    nltk.download(download_name, quiet=True)
                    logger.info(f"✓ Downloaded {download_name}")
                except Exception as e:
                    logger.error(f"✗ Failed to download {download_name}: {e}")

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
        start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_id = str(uuid.uuid4())[:8]
        analysis_logger.info(f"<BEGIN ANALYSIS id:{run_id} mode:analyze file:{file_path} timestamp:{start_timestamp}>")
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
                "id": run_id,
                "document_info": document["metadata"],
                "features": features,
                "whitefonting_detection": whitefonting_results,
                "classification": classification_results,
                "turnover_prediction": turnover_results,
            }
            
            analysis_logger.info(f"{whitefonting_results}")
            new_metadata = {key: value for key, value in document['metadata'].items() if key != 'creation_date'}
            analysis_logger.info(f"{new_metadata}")
            analysis_logger.info("{" + f"'skills': {features['skills']}" + "}")

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
            end_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_logger.info(f"<END ANALYSIS id:{run_id} timestamp:{end_timestamp}>")
            return analysis_results

        except Exception as e:
            end_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            logger.error(f"Error analyzing resume {file_path}: {str(e)}")
            analysis_logger.info(f"<END ANALYSIS id:{run_id} timestamp:{end_timestamp}>")
            return {"error": str(e), "file_path": file_path}

    def batch_analyze(self, directory_path, file_pattern="*.pdf", **kwargs):
        start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_id = str(uuid.uuid4())[:8]  
        analysis_logger.info(f"<BEGIN ANALYSIS id:{batch_id} mode:batch file:{directory_path} pattern:{file_pattern} timestamp:{start_timestamp}>")
        logger.info(f"Batch analyzing resumes in {directory_path} matching {file_pattern}")

        file_extensions = []
        if file_pattern == '*':
            # Use all supported formats
            file_extensions = None
            logger.info("Using all supported file formats for batch processing")
        else:
            for pattern in file_pattern.split(';'):
                extension = os.path.splitext(pattern)[1].lower()
                if extension: 
                    file_extensions.append(extension)
            logger.info(f"Using specific file extensions: {file_extensions}")

        logger.info(f"Directory contents: {os.listdir(directory_path) if os.path.exists(directory_path) else 'Directory not found'}")
        
        batch_docs = self.document_loader.batch_load(
            directory_path, file_types=file_extensions
        )

        if not batch_docs:
            logger.warning(f"No matching documents found in {directory_path}")
            end_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_logger.info(f"<END ANALYSIS id:{batch_id} timestamp:{end_timestamp}>")
            return [], None, None, batch_id

        batch_results = []
        for doc in batch_docs:
            file_path = doc["metadata"]["file_path"]
            try:
                logger.info(f"Processing file: {file_path}")
                start_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                run_id = str(uuid.uuid4())[:8]
                analysis_logger.info(f"<BEGIN ANALYSIS id:{run_id} mode:analyze file:{file_path} timestamp:{start_timestamp}>")

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
                    "id": run_id,
                    "document_info": doc["metadata"],
                    "whitefonting_detection": whitefonting_results,
                    "classification": classification_results,
                    "turnover_prediction": turnover_results,
                }

                batch_results.append(result)
                end_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                analysis_logger.info(f"<END ANALYSIS id:{run_id} timestamp:{end_timestamp}>")
                logger.info(f"Completed processing: {file_path}")

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
        end_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        analysis_logger.info(f"<END ANALYSIS id:{batch_id} timestamp:{end_timestamp}>")
        return batch_results, batch_report_path, batch_dashboard_path, batch_id

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

    def parse_analysis_history(self, log_file_path="resume_analysis.log", limit=10, mode="server"):
        """
        Parse the analysis log file to extract the latest analysis runs.
        
        Args:
            log_file_path: Path to the log file
            limit: Number of latest runs to return
            
        Returns:
            List of analysis run dictionaries
        """
        analysis_runs = []
        current_batch = None
        current_file_analysis = None
        
        try:
            if not os.path.exists(log_file_path):
                return []
                
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()

                if line.startswith('<BEGIN ANALYSIS'):
                    try:
                        parts = line[15:-1]
                        
                        new_analysis = {
                            'status': 'running',
                            'start_time': None,
                            'end_time': None,
                            'mode': None,
                            'file': None,
                            'pattern': None,
                            'duration_seconds': None,
                            'id': None,
                            'files_processed': [],
                            'whitefonting_results': {},
                            'classification_results': {},
                            'turnover_results': {}
                        }
                        
                        for part in parts.split():
                            if ':' in part:
                                key, value = part.split(':', 1)
                                if key == 'id':
                                    new_analysis['id'] = value
                                elif key == 'mode':
                                    new_analysis['mode'] = value
                                elif key == 'file':
                                    new_analysis['file'] = value
                                elif key == 'pattern':
                                    new_analysis['pattern'] = value
                                elif key == 'timestamp':
                                    new_analysis['start_time'] = value
                        
                        if new_analysis['mode'] == 'batch':
                            current_batch = new_analysis
                            current_file_analysis = None
                        elif new_analysis['mode'] == 'analyze' and current_batch:
                            current_file_analysis = new_analysis
                        else:
                            current_batch = new_analysis
                            current_file_analysis = None
                                    
                    except Exception as e:
                        logger.error(f"Error parsing BEGIN marker: {e}")
                        continue
                        
                elif line.startswith('<END ANALYSIS'):
                    try:
                        parts = line[13:-1]  
                        analysis_id = None
                        end_timestamp = None
                        
                        for part in parts.split():
                            if ':' in part:
                                key, value = part.split(':', 1)
                                if key == 'timestamp':
                                    end_timestamp = value
                                elif key == 'id':
                                    analysis_id = value
                        
                        if current_file_analysis and analysis_id == current_file_analysis.get('id'):
                            current_file_analysis['end_time'] = end_timestamp
                            current_file_analysis['status'] = 'completed'
                            
                            if current_file_analysis['start_time'] and end_timestamp:
                                try:
                                    start_dt = datetime.strptime(current_file_analysis['start_time'], '%Y%m%d_%H%M%S')
                                    end_dt = datetime.strptime(end_timestamp, '%Y%m%d_%H%M%S')
                                    duration = (end_dt - start_dt).total_seconds()
                                    current_file_analysis['duration_seconds'] = duration
                                except Exception:
                                    current_file_analysis['duration_seconds'] = None
                            
                            if current_batch:
                                current_batch['files_processed'].append(current_file_analysis.copy())
                            
                            current_file_analysis = None
                            
                        elif current_batch and analysis_id == current_batch.get('id'):
                            current_batch['end_time'] = end_timestamp
                            current_batch['status'] = 'completed'
                            
                            if current_batch['start_time'] and end_timestamp:
                                try:
                                    start_dt = datetime.strptime(current_batch['start_time'], '%Y%m%d_%H%M%S')
                                    end_dt = datetime.strptime(end_timestamp, '%Y%m%d_%H%M%S')
                                    duration = (end_dt - start_dt).total_seconds()
                                    current_batch['duration_seconds'] = duration
                                except Exception:
                                    current_batch['duration_seconds'] = None
                            
                            analysis_runs.append(current_batch)
                            current_batch = None
                            current_file_analysis = None
                        
                    except Exception as e:
                        logger.error(f"Error parsing END marker: {e}")
                        if current_file_analysis:
                            current_file_analysis['status'] = 'error'
                            if current_batch:
                                current_batch['files_processed'].append(current_file_analysis.copy())
                            current_file_analysis = None
                        elif current_batch:
                            current_batch['status'] = 'error'
                            analysis_runs.append(current_batch)
                            current_batch = None
                            
                else:
                    try:
                        json_string = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '"', line)

                        json_string = json_string.replace("True", "true").replace("False", "false").replace("None", "null")
                        
                        json_data = json.loads(json_string)
                        target_analysis = current_file_analysis if current_file_analysis else current_batch
                        
                        if 'white_text_analysis' in json_data:
                            target_analysis['whitefonting_results'] = json_data
                        elif 'predicted_category' in json_data:
                            target_analysis['classification_results'] = json_data
                        elif 'prediction' in json_data:
                            target_analysis['turnover_results'] = json_data
                            
                    except json.JSONDecodeError:
                        if mode == "server":
                            logger.warning(f"Skipping, not valid JSON: {line}")
                    except Exception as e:
                        logger.error(f"Error parsing JSON result: {e}")
            
            if current_batch:
                current_batch['status'] = 'running'
                if current_file_analysis:
                    current_file_analysis['status'] = 'running'
                    current_batch['files_processed'].append(current_file_analysis)
                analysis_runs.append(current_batch)

            return analysis_runs[-limit:] if len(analysis_runs) >= limit else analysis_runs
            
        except Exception as e:
            logger.error(f"Error reading analysis history: {e}")
            return []

    
def init_server(system):
    # Configure file upload settings
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
    app.config['UPLOAD_FOLDER'] = 'temp_uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    @app.route('/health', methods=['GET'])
    def health_check():
        process = psutil.Process()
        uptime = time.time() - process.create_time()
        memory_info = process.memory_info()

        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 2),
            "memory_usage_mb": round(memory_info.rss / (1024 * 1024), 2),
            "active_endpoints": [rule.rule for rule in app.url_map.iter_rules()]
        }), 200

    @app.route('/metrics', methods=['GET'])
    def metrics():
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_usage = psutil.disk_usage('/')

        return jsonify({
            "cpu_usage_percent": cpu_percent,
            "memory": {
                "total_mb": round(memory.total / (1024 * 1024), 2),
                "used_mb": round(memory.used / (1024 * 1024), 2),
                "available_mb": round(memory.available / (1024 * 1024), 2),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": round(disk_usage.total / (1024 * 1024 * 1024), 2),
                "used_gb": round(disk_usage.used / (1024 * 1024 * 1024), 2),
                "free_gb": round(disk_usage.free / (1024 * 1024 * 1024), 2),
                "percent_used": disk_usage.percent
            }
        }), 200

    @app.route('/analyze', methods=['POST'])
    def analyze_resume():
        try:
            # Handle file upload
            if 'file_path' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file_path']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400

            if not system.allowed_file(file.filename):
                return jsonify({"error": "File type not allowed. Please upload PDF, DOC, DOCX, or TXT files."}), 400
            
            # Get options from form data
            generate_visuals = request.form.get('generate_visuals', 'true').lower() == 'true'
            generate_report = request.form.get('generate_report', 'true').lower() == 'true'
            generate_dashboard = request.form.get('generate_dashboard', 'true').lower() == 'true'
            
            # Save uploaded file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
            file.save(file_path)
            
            try:
                # Analyze the resume
                results = system.analyze_resume(
                    file_path,
                    generate_visuals=generate_visuals,
                    generate_report=generate_report,
                    generate_dashboard=generate_dashboard,
                )
                
                return jsonify({
                    "status": "success",
                    "id": results["id"],
                    "data": results["document_info"],
                    "timestamp": datetime.now().isoformat(),
                })
                
            finally:
                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
        except Exception as e:
            logger.error(f"Error analyzing resume: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/batch', methods=['POST'])
    def batch_analyze():
        try:
            # Handle multiple file uploads
            if 'files' not in request.files:
                return jsonify({"error": "No files provided"}), 400
            
            files = request.files.getlist('files')
            if not files or all(f.filename == '' for f in files):
                return jsonify({"error": "No files selected"}), 400
            
            # Get options from form data
            generate_visuals = request.form.get('generate_visuals', 'true').lower() == 'true'
            generate_report = request.form.get('generate_report', 'true').lower() == 'true'
            generate_dashboard = request.form.get('generate_dashboard', 'true').lower() == 'true'
            
            # Create temporary directory for batch processing
            batch_temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{uuid.uuid4()}")
            os.makedirs(batch_temp_dir, exist_ok=True)
            
            try:
                # Save all uploaded files
                temp_file_paths = []
                for file in files:
                    if file.filename != '' and system.allowed_file(file.filename):
                        filename = secure_filename(file.filename)
                        file_path = os.path.join(batch_temp_dir, f"{uuid.uuid4()}_{filename}")
                        file.save(file_path)
                        temp_file_paths.append(file_path)
                
                if not temp_file_paths:
                    return jsonify({"error": "No valid files to process"}), 400
                
                logger.info(f"Processing {len(temp_file_paths)} files in batch: {batch_temp_dir}")
                logger.info(f"Uploaded files: {[os.path.basename(f) for f in temp_file_paths]}")
                
                # Analyze the batch using the temporary directory
                batch_results, batch_report, batch_dashboard, batch_id = system.batch_analyze(
                    batch_temp_dir, 
                    file_pattern='*',
                    generate_visuals=generate_visuals,
                    generate_report=generate_report,
                    generate_dashboard=generate_dashboard
                )
                
                return jsonify({
                    "status": "success",
                    "id": batch_id,
                    "total": len(batch_results),
                    "timestamp": datetime.now().isoformat(),
                })
                
            finally:
                # Clean up temporary files and directory
                if os.path.exists(batch_temp_dir):
                    shutil.rmtree(batch_temp_dir)
                    
        except Exception as e:
            logger.error(f"Error in batch analysis: {str(e)}")
            return jsonify({"error": str(e)}), 500

    @app.route('/generate-samples', methods=['POST'])
    def generate_samples():
        data = request.json
        output_dir = data.get('output_dir', 'sample_resumes')
        count = data.get('count', 2)
        include_white_text = data.get('include_white_text', True)

        sample_paths = system.generate_sample_documents(
            output_dir=output_dir,
            count=count,
            include_white_text=include_white_text,
        )
        return jsonify({"sample_paths": sample_paths})

    @app.route('/history', methods=['GET'])
    def get_analysis_history():
        """
        Get the latest analysis runs from the log file.
        
        Query parameters:
            - limit: Number of runs to return (default: 10, max: 100)
        """
        try:
            limit = request.args.get('limit', 10, type=int)
            
            if limit < 1:
                limit = 10
            elif limit > 100:
                limit = 100
                
            history = system.parse_analysis_history(limit=limit)
            
            response_data = {
                "status": "success",
                "total_runs": len(history),
                "limit": limit,
                "runs": []
            }
            
            for run in reversed(history):  # Most recent first
                run_data = {
                    "id": run.get('id', 'unknown'),
                    "mode": run.get('mode', 'unknown'),
                    "file": run.get('file', 'unknown'),
                    "status": run.get('status', 'unknown'),
                    "start_time": run.get('start_time'),
                    "end_time": run.get('end_time'),
                    "duration_seconds": run.get('duration_seconds')
                }
                
                # Add pattern for batch runs
                if run.get('pattern'):
                    run_data['pattern'] = run.get('pattern')
                
                # Add analysis results
                if run.get('whitefonting_results'):
                    run_data['whitefonting_results'] = run.get('whitefonting_results')
                
                if run.get('classification_results'):
                    run_data['classification_results'] = run.get('classification_results')
                    
                if run.get('turnover_results'):
                    run_data['turnover_results'] = run.get('turnover_results')
                
                # Add files processed for batch runs with their analysis results
                if run.get('mode') == 'batch' and run.get('files_processed'):
                    processed_files = []
                    for file_info in run.get('files_processed', []):
                        file_data = {
                            'id': file_info.get('id', 'unknown'),
                            'file': file_info.get('file', 'unknown'),
                            'status': file_info.get('status', 'unknown'),
                            'start_time': file_info.get('start_time'),
                            'end_time': file_info.get('end_time'),
                            'duration_seconds': file_info.get('duration_seconds')
                        }
                        
                        # Include analysis results for each file
                        if file_info.get('whitefonting_results'):
                            file_data['whitefonting_results'] = file_info.get('whitefonting_results')
                        
                        if file_info.get('classification_results'):
                            file_data['classification_results'] = file_info.get('classification_results')
                            
                        if file_info.get('turnover_results'):
                            file_data['turnover_results'] = file_info.get('turnover_results')
                        
                        processed_files.append(file_data)
                    
                    run_data['files_processed'] = processed_files
                    run_data['total_files'] = len(processed_files)
                    
                    # Calculate summary statistics
                    completed_files = [f for f in processed_files if f.get('status') == 'completed']
                    error_files = [f for f in processed_files if f.get('status') == 'error']
                    running_files = [f for f in processed_files if f.get('status') == 'running']
                    
                    run_data['completed_files'] = len(completed_files)
                    run_data['error_files'] = len(error_files)
                    run_data['running_files'] = len(running_files)
                    
                    # Summary of detection results across all files
                    whitefonting_detections = sum(1 for f in completed_files 
                                                if f.get('whitefonting_results', {}).get('has_white_text', False))
                    keyword_stuffing_detections = sum(1 for f in completed_files 
                                                    if f.get('whitefonting_results', {}).get('keyword_stuffing_detected', False))
                    
                    run_data['summary'] = {
                        'whitefonting_detections': whitefonting_detections,
                        'keyword_stuffing_detections': keyword_stuffing_detections,
                        'total_completed': len(completed_files)
                    }
                    
                    # Most common predicted categories
                    categories = [f.get('classification_results', {}).get('predicted_category') 
                                for f in completed_files if f.get('classification_results', {}).get('predicted_category')]
                    if categories:
                        from collections import Counter
                        category_counts = Counter(categories)
                        run_data['summary']['top_categories'] = dict(category_counts.most_common(3))
                    
                    # Turnover prediction summary
                    turnover_predictions = [f.get('turnover_results', {}).get('prediction', {}).get('will_leave') 
                                        for f in completed_files if f.get('turnover_results', {}).get('prediction')]
                    if turnover_predictions:
                        will_leave_count = sum(1 for pred in turnover_predictions if pred)
                        run_data['summary']['turnover_risk'] = {
                            'high_risk_count': will_leave_count,
                            'low_risk_count': len(turnover_predictions) - will_leave_count,
                            'total_predictions': len(turnover_predictions)
                        }
                    
                response_data["runs"].append(run_data)
            
            return jsonify(response_data), 200
            
        except Exception as e:
            logger.error(f"Error in /history endpoint: {e}")
            return jsonify({
                "status": "error",
                "message": f"Failed to retrieve analysis history: {str(e)}"
            }), 500

    @app.route('/clear-log', methods=['GET'])
    def clear_analysis_log():
        """
        Clear the resume_analysis.log file to remove all analysis history.
        
        Returns:
            JSON response indicating success or failure
        """
        try:
            log_file_path = "resume_analysis.log"
            
            if os.path.exists(log_file_path):
                file_size = os.path.getsize(log_file_path)
                
                with open(log_file_path, 'w') as f:
                    pass 
                
                logger.info(f"Analysis log cleared. Previous size: {file_size} bytes")
                
                return jsonify({
                    "status": "success",
                    "message": "Analysis log cleared successfully",
                    "previous_size_bytes": file_size,
                    "timestamp": datetime.now().isoformat()
                }), 200
            else:
                return jsonify({
                    "status": "success", 
                    "message": "Log file does not exist, nothing to clear",
                    "timestamp": datetime.now().isoformat()
                }), 200
                
        except PermissionError:
            logger.error("Permission denied when trying to clear analysis log")
            return jsonify({
                "status": "error",
                "message": "Permission denied. Cannot clear the analysis log file."
            }), 403
            
        except Exception as e:
            logger.error(f"Error clearing analysis log: {e}")
            return jsonify({
                "status": "error",
                "message": f"Failed to clear analysis log: {str(e)}"
            }), 500

    @app.route('/analysis/<analysis_id>', methods=['GET'])
    def get_analysis_by_id(analysis_id):
        """
        Get a specific analysis result by ID from the log file.
        
        Args:
            analysis_id: The unique ID of the analysis to retrieve
            
        Returns:
            JSON response with the analysis data or error message
        """
        try:
            if not analysis_id:
                return jsonify({
                    "status": "error",
                    "message": "Analysis ID is required"
                }), 400
            
            # Search through the analysis history for the specific ID
            analysis_data = find_analysis_by_id(analysis_id)
            
            if not analysis_data:
                return jsonify({
                    "status": "error",
                    "message": f"Analysis with ID '{analysis_id}' not found"
                }), 404
            
            return jsonify({
                "status": "success",
                "analysis_id": analysis_id,
                "data": analysis_data
            }), 200
            
        except Exception as e:
            logger.error(f"Error in /analysis/<id> endpoint: {e}")
            return jsonify({
                "status": "error",
                "message": f"Failed to retrieve analysis: {str(e)}"
            }), 500

    def find_analysis_by_id(target_id, log_file_path="resume_analysis.log"):
        """
        Search for a specific analysis ID in the log file and return its data.
        
        Args:
            target_id: The ID to search for
            log_file_path: Path to the log file
            
        Returns:
            Dictionary containing the analysis data or None if not found
        """
        try:
            if not os.path.exists(log_file_path):
                return None
                
            with open(log_file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_analysis = None
            found_analysis = None
            
            for line in lines:
                line = line.strip()
                
                # Check for BEGIN ANALYSIS marker
                if line.startswith('<BEGIN ANALYSIS'):
                    try:
                        parts = line[15:-1]
                        
                        analysis = {
                            'id': None,
                            'mode': None,
                            'file': None,
                            'pattern': None,
                            'timestamp': None,
                            'start_time': None,
                            'end_time': None,
                            'status': 'running',
                            'duration_seconds': None,
                            'whitefonting_results': {},
                            'classification_results': {},
                            'turnover_results': {},
                            'file_metadata': {},
                            'skills': [],
                        }
                        
                        # Parse the BEGIN marker
                        for part in parts.split():
                            if ':' in part:
                                key, value = part.split(':', 1)
                                if key == 'id':
                                    analysis['id'] = value
                                elif key == 'mode':
                                    analysis['mode'] = value
                                elif key == 'file':
                                    analysis['file'] = value
                                elif key == 'pattern':
                                    analysis['pattern'] = value
                                elif key == 'timestamp':
                                    analysis['timestamp'] = value
                                    analysis['start_time'] = value
                        
                        # Check if this is the target ID
                        if analysis['id'] == target_id:
                            current_analysis = analysis
                            found_analysis = analysis
                        else:
                            current_analysis = None
                            
                    except Exception as e:
                        logger.error(f"Error parsing BEGIN marker: {e}")
                        continue
                        
                # Check for END ANALYSIS marker
                elif line.startswith('<END ANALYSIS') and current_analysis:
                    try:
                        parts = line[13:-1]
                        analysis_id = None
                        end_timestamp = None
                        
                        for part in parts.split():
                            if ':' in part:
                                key, value = part.split(':', 1)
                                if key == 'timestamp':
                                    end_timestamp = value
                                elif key == 'id':
                                    analysis_id = value
                        
                        # Verify this END marker matches our current analysis
                        if analysis_id == current_analysis['id']:
                            current_analysis['end_time'] = end_timestamp
                            current_analysis['status'] = 'completed'
                            
                            # Calculate duration if both timestamps are available
                            if current_analysis['start_time'] and end_timestamp:
                                try:
                                    start_dt = datetime.strptime(current_analysis['start_time'], '%Y%m%d_%H%M%S')
                                    end_dt = datetime.strptime(end_timestamp, '%Y%m%d_%H%M%S')
                                    duration = (end_dt - start_dt).total_seconds()
                                    current_analysis['duration_seconds'] = duration
                                except Exception:
                                    current_analysis['duration_seconds'] = None
                            
                            # If this was our target, we're done
                            if current_analysis['id'] == target_id:
                                return found_analysis
                            
                            current_analysis = None
                        
                    except Exception as e:
                        logger.error(f"Error parsing END marker: {e}")
                        if current_analysis:
                            current_analysis['status'] = 'error'
                            if current_analysis['id'] == target_id:
                                return found_analysis
                        current_analysis = None
                        
                elif current_analysis and current_analysis['id'] == target_id:
                    try:
                        json_string = re.sub(r"(?<![a-zA-Z])'|'(?![a-zA-Z])", '"', line)

                        json_string = json_string.replace("True", "true").replace("False", "false").replace("None", "null")
                        
                        json_data = json.loads(json_string)
                        
                        # Categorize the JSON data based on its content
                        if 'white_text_analysis' in json_data:
                            current_analysis['whitefonting_results'] = json_data
                        elif 'predicted_category' in json_data:
                            current_analysis['classification_results'] = json_data
                        elif 'prediction' in json_data:
                            current_analysis['turnover_results'] = json_data
                        elif 'skills' in json_data:
                            current_analysis['skills'] = json_data
                        elif 'author' in json_data:
                            current_analysis['file_metadata'] = json_data
                            
                    except json.JSONDecodeError:
                        # Skip lines that aren't valid JSON
                        logger.warning(f"Skipping line, not valid JSON: {json_string}")
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing JSON result: {e}")
                        continue
            
            # If we found the analysis but didn't encounter an END marker, mark as running
            if found_analysis and found_analysis['status'] == 'running':
                return found_analysis
                
            return None
            
        except Exception as e:
            logger.error(f"Error searching for analysis ID {target_id}: {e}")
            return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Analysis System")
    parser.add_argument("command", choices=["server", "analyze", "batch", "history", "generate-samples"], help="Command to execute")
    parser.add_argument("file_path", nargs="?", help="Path to the resume file (required for 'analyze')")
    parser.add_argument("--limit", type=int, default=10, help="Limit for history command")
    args = parser.parse_args()
    system = ResumeAnalysisSystem()

    if args.command == "server":
        init_server(system)
        app.run(host="0.0.0.0", port=8500)
    elif args.command == "analyze":
        if not args.file_path:
            print("Error: 'file_path' is required for the 'analyze' command.")
            sys.exit(1)
        results = system.analyze_resume(args.file_path)
        pretty_print_json(results)
    elif args.command == "batch":
        if not args.file_path:
            print("Error: 'file_path' is required for the 'batch' command.")
            sys.exit(1)
        batch_results, batch_report, batch_dashboard, batch_id = system.batch_analyze(args.file_path)
        print(f"Batch Analysis ID: {batch_id}")
        pretty_print_json({"Batch Results": batch_results})
        if batch_report:
            print(f"Batch Report Path: {batch_report}")
        if batch_dashboard:
            print(f"Batch Dashboard Path: {batch_dashboard}")
    elif args.command == "history":
        history = system.parse_analysis_history(limit=args.limit, mode="cli")
        print("\n")
        logger.info("Analysis History:")
        for idx, run in enumerate(history, start=1):
            run_data = {
                "ID": run.get('id', 'unknown'),
                "Mode": run.get('mode', 'unknown'),
                "File": run.get('file', 'unknown'),
                "Status": run.get('status', 'unknown'),
                "Start Time": run.get('start_time'),
                "End Time": run.get('end_time'),
                "Duration (seconds)": run.get('duration_seconds')
            }
            print("\n")
            pretty_print_table([run_data], title=f"Run {idx}")
    elif args.command == "generate-samples":
        output_dir = "sample_resumes"
        count = 2
        include_white_text = True
        sample_paths = system.generate_sample_documents(output_dir=output_dir, count=count, include_white_text=include_white_text)
        print(f"Generated {len(sample_paths)} sample resumes:")
        for path in sample_paths:
            print(f"- {path}")
