# visualization/dashboard.py
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import base64
from datetime import datetime

logger = logging.getLogger(__name__)

class DashboardGenerator:
    """
    Creates interactive web-based dashboards for resume analysis.
    
    This class generates interactive HTML dashboards that include
    visualizations and analysis results from all modules in the system.
    """
    
    def __init__(self, output_dir="dashboards"):
        """
        Initialize the DashboardGenerator.
        
        Args:
            output_dir (str): Directory where dashboards will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initializing DashboardGenerator (output_dir: {output_dir})")
    
    def generate_individual_dashboard(self, analysis_results):
        """
        Generate an individual dashboard for a single resume.
        
        Args:
            analysis_results (dict): Results from all analysis modules.
            
        Returns:
            str: Path to the generated dashboard HTML file.
        """
        # Extract document info
        document_info = analysis_results.get('document_info', {})
        filename = os.path.basename(document_info.get('file_path', 'Unknown Document'))
        
        # Create a unique filename for the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_filename = f"{filename.replace('.pdf', '')}_dashboard_{timestamp}.html"
        dashboard_path = os.path.join(self.output_dir, dashboard_filename)
        
        # Import visualization module locally to avoid circular imports
        from visualization.data_visualizer import DataVisualizer
        visualizer = DataVisualizer()
        
        # Generate the dashboard using the visualizer
        dashboard_path = visualizer.create_dashboard(
            analysis_results, 
            dashboard_path, 
            title=f"Resume Analysis: {filename}"
        )
        
        logger.info(f"Generated individual dashboard: {dashboard_path}")
        return dashboard_path
    
    def generate_batch_dashboard(self, batch_results):
        """
        Generate a dashboard for batch analysis results.
        
        Args:
            batch_results (list): Results from analyzing multiple resumes.
            
        Returns:
            str: Path to the generated dashboard HTML file.
        """
        # Create a unique filename for the dashboard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dashboard_filename = f"batch_dashboard_{timestamp}.html"
        dashboard_path = os.path.join(self.output_dir, dashboard_filename)
        
        # Extract data for visualizations
        total_resumes = len(batch_results)
        
        # Prepare data
        whitefonting_data = self._prepare_whitefonting_data(batch_results)
        classification_data = self._prepare_classification_data(batch_results)
        turnover_data = self._prepare_turnover_data(batch_results)
        
        # Create dashboard HTML
        html_content = self._create_batch_dashboard_html(
            dashboard_title=f"Batch Resume Analysis ({total_resumes} Resumes)",
            whitefonting_data=whitefonting_data,
            classification_data=classification_data,
            turnover_data=turnover_data
        )
        
        # Write to file
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Generated batch dashboard: {dashboard_path}")
        return dashboard_path
    
    def _prepare_whitefonting_data(self, batch_results):
        """Prepare whitefonting data for visualization."""
        # Count detection results
        whitefonting_counts = {
            'Detected': sum(1 for r in batch_results 
                          if r.get('whitefonting_detection', {}).get('has_white_text', False)),
            'None Detected': sum(1 for r in batch_results 
                               if not r.get('whitefonting_detection', {}).get('has_white_text', False))
        }
        
        # Extract white text percentages for histogram
        percentages = []
        filenames = []
        for result in batch_results:
            white_text_pct = result.get('whitefonting_detection', {}).get('white_text_percentage', 0)
            filename = os.path.basename(result.get('document_info', {}).get('file_path', 'Unknown'))
            
            percentages.append(white_text_pct)
            filenames.append(filename)
        
        return {
            'counts': whitefonting_counts,
            'percentages': percentages,
            'filenames': filenames
        }
    
    def _prepare_classification_data(self, batch_results):
        """Prepare classification data for visualization."""
        # Count category distribution
        category_counts = {}
        for result in batch_results:
            category = result.get('classification', {}).get('predicted_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Extract confidence values
        categories = []
        confidences = []
        filenames = []
        for result in batch_results:
            category = result.get('classification', {}).get('predicted_category', 'Unknown')
            confidence = result.get('classification', {}).get('confidence', 0) * 100
            filename = os.path.basename(result.get('document_info', {}).get('file_path', 'Unknown'))
            
            categories.append(category)
            confidences.append(confidence)
            filenames.append(filename)
        
        return {
            'category_counts': category_counts,
            'categories': categories,
            'confidences': confidences,
            'filenames': filenames
        }
    
    def _prepare_turnover_data(self, batch_results):
        """Prepare turnover prediction data for visualization."""
        # Count risk levels
        risk_counts = {
            'High Risk': sum(1 for r in batch_results 
                           if r.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False)),
            'Low Risk': sum(1 for r in batch_results 
                          if not r.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False))
        }
        
        # Extract probabilities
        probabilities = []
        risk_levels = []
        filenames = []
        for result in batch_results:
            prob = result.get('turnover_prediction', {}).get('prediction', {}).get('leave_probability', 0) * 100
            risk = 'High Risk' if result.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False) else 'Low Risk'
            filename = os.path.basename(result.get('document_info', {}).get('file_path', 'Unknown'))
            
            probabilities.append(prob)
            risk_levels.append(risk)
            filenames.append(filename)
        
        return {
            'risk_counts': risk_counts,
            'probabilities': probabilities,
            'risk_levels': risk_levels,
            'filenames': filenames
        }
    
    def _create_batch_dashboard_html(self, dashboard_title, whitefonting_data, classification_data, turnover_data):
        """Create HTML content for the batch dashboard."""
        # Create plotly figures
        whitefonting_fig = self._create_whitefonting_figure(whitefonting_data)
        classification_fig = self._create_classification_figure(classification_data)
        turnover_fig = self._create_turnover_figure(turnover_data)
        
        # Summary statistics
        total_resumes = len(whitefonting_data['filenames'])
        whitefonting_detected = whitefonting_data['counts'].get('Detected', 0)
        high_risk_count = turnover_data['risk_counts'].get('High Risk', 0)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{dashboard_title}</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Open Sans', sans-serif;
                    margin: 0;
                    padding: 0;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .section {{
                    background-color: white;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                .summary {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                }}
                .summary-item {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px;
                    flex: 1;
                    min-width: 200px;
                    text-align: center;
                }}
                .visualization {{
                    width: 100%;
                    margin-top: 20px;
                }}
                h1, h2, h3 {{
                    margin-top: 0;
                }}
                .high-risk {{
                    color: #e74c3c;
                }}
                .warning {{
                    color: #e74c3c;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{dashboard_title}</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Summary Statistics</h2>
                    <div class="summary">
                        <div class="summary-item">
                            <h3>Total Resumes</h3>
                            <p>{total_resumes}</p>
                        </div>
                        
                        <div class="summary-item">
                            <h3>Whitefonting</h3>
                            <p class="{'warning' if whitefonting_detected > 0 else ''}">
                                {whitefonting_detected} ({whitefonting_detected/total_resumes*100:.1f}%)
                            </p>
                        </div>
                        
                        <div class="summary-item">
                            <h3>High Turnover Risk</h3>
                            <p class="{'high-risk' if high_risk_count > 0 else ''}">
                                {high_risk_count} ({high_risk_count/total_resumes*100:.1f}%)
                            </p>
                        </div>
                    </div>
                </div>
        """
        
        # Add whitefonting section
        whitefonting_div = 'whitefonting-plot'
        whitefonting_json = whitefonting_fig.to_json()
        
        html_content += f"""
            <div class="section">
                <h2>Whitefonting Detection</h2>
                <div id="{whitefonting_div}" class="visualization"></div>
            </div>
            
            <script>
                var whitefonting_data = {whitefonting_json};
                Plotly.newPlot('{whitefonting_div}', whitefonting_data.data, whitefonting_data.layout);
            </script>
        """
        
        # Add classification section
        classification_div = 'classification-plot'
        classification_json = classification_fig.to_json()
        
        html_content += f"""
            <div class="section">
                <h2>Resume Classification</h2>
                <div id="{classification_div}" class="visualization"></div>
            </div>
            
            <script>
                var classification_data = {classification_json};
                Plotly.newPlot('{classification_div}', classification_data.data, classification_data.layout);
            </script>
        """
        
        # Add turnover section
        turnover_div = 'turnover-plot'
        turnover_json = turnover_fig.to_json()
        
        html_content += f"""
            <div class="section">
                <h2>Turnover Prediction</h2>
                <div id="{turnover_div}" class="visualization"></div>
            </div>
            
            <script>
                var turnover_data = {turnover_json};
                Plotly.newPlot('{turnover_div}', turnover_data.data, turnover_data.layout);
            </script>
        """
        
        # Close HTML document
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _create_whitefonting_figure(self, data):
        """Create Plotly figure for whitefonting data."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("White Text Detection Results", "White Text Percentage by Resume")
        )
        
        # Add pie chart for detection results
        fig.add_trace(
            go.Pie(
                labels=list(data['counts'].keys()),
                values=list(data['counts'].values()),
                marker_colors=['#ff6384', '#36a2eb']
            ),
            row=1, col=1
        )
        
        # Add bar chart for white text percentages
        fig.add_trace(
            go.Bar(
                x=data['filenames'],
                y=data['percentages'],
                marker_color=['#ff6384' if p > 0 else '#36a2eb' for p in data['percentages']]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            width=1000,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Resume", row=1, col=2)
        fig.update_yaxes(title_text="White Text %", row=1, col=2)
        
        return fig
    
    def _create_classification_figure(self, data):
        """Create Plotly figure for classification data."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Category Distribution", "Classification Confidence by Resume")
        )
        
        # Add pie chart for category distribution
        fig.add_trace(
            go.Pie(
                labels=list(data['category_counts'].keys()),
                values=list(data['category_counts'].values())
            ),
            row=1, col=1
        )
        
        # Add bar chart for classification confidence
        fig.add_trace(
            go.Bar(
                x=data['filenames'],
                y=data['confidences'],
                marker_color='#36a2eb',
                hovertext=[f"Category: {cat}" for cat in data['categories']]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            width=1000,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Resume", row=1, col=2)
        fig.update_yaxes(title_text="Confidence %", row=1, col=2)
        
        return fig
    
    def _create_turnover_figure(self, data):
        """Create Plotly figure for turnover prediction data."""
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Turnover Risk Distribution", "Turnover Probability by Resume")
        )
        
        # Add pie chart for risk distribution
        fig.add_trace(
            go.Pie(
                labels=list(data['risk_counts'].keys()),
                values=list(data['risk_counts'].values()),
                marker_colors=['#ff6384', '#36a2eb']
            ),
            row=1, col=1
        )
        
        # Add bar chart for turnover probabilities
        bar_colors = ['#ff6384' if risk == 'High Risk' else '#36a2eb' for risk in data['risk_levels']]
        
        fig.add_trace(
            go.Bar(
                x=data['filenames'],
                y=data['probabilities'],
                marker_color=bar_colors,
                hovertext=data['risk_levels']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            width=1000,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Resume", row=1, col=2)
        fig.update_yaxes(title_text="Turnover Probability %", row=1, col=2)
        
        # Add a horizontal line at 50% threshold
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=50,
            x1=len(data['filenames']) - 0.5,
            y1=50,
            line=dict(
                color="red",
                width=2,
                dash="dash",
            ),
            row=1, col=2
        )
        
        return fig