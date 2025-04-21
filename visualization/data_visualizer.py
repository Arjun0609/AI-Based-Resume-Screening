# visualization/data_visualizer.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import io
from PIL import Image
import logging
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64

logger = logging.getLogger(__name__)

class DataVisualizer:
    """
    Creates visualizations for different aspects of resume analysis.
    
    This class provides methods to generate various types of visualizations
    for whitefonting detection, resume classification, and turnover prediction
    results, either as static images or interactive plotly charts.
    """
    
    def __init__(self, output_dir="visualizations"):
        """
        Initialize the DataVisualizer.
        
        Args:
            output_dir (str): Directory where visualizations will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set default styling for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        
        logger.info(f"Initializing DataVisualizer (output_dir: {output_dir})")
    
    def visualize_whitefonting(self, whitefonting_results, interactive=False, save_path=None):
        """
        Create visualizations for whitefonting detection results.
        
        Args:
            whitefonting_results (dict): Results from whitefonting detection.
            interactive (bool): Whether to create interactive plotly visualizations.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            str or plotly.graph_objects.Figure: Path to saved visualization or plotly figure.
        """
        # Extract data
        has_white_text = whitefonting_results.get('has_white_text', False)
        
        if not has_white_text:
            logger.info("No white text detected, skipping visualization")
            return None
        
        white_text_percentage = whitefonting_results.get('white_text_percentage', 0)
        common_words = whitefonting_results.get('common_words', {})
        
        if interactive:
            # Create interactive visualization with plotly
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{"type": "pie"}, {"type": "bar"}]],
                               subplot_titles=("White Text Distribution", "Common Hidden Words"))
            
            # Add pie chart for white/visible text distribution
            fig.add_trace(
                go.Pie(
                    labels=["Visible Text", "Hidden White Text"],
                    values=[100 - white_text_percentage, white_text_percentage],
                    marker_colors=['#36a2eb', '#ff6384']
                ),
                row=1, col=1
            )
            
            # Add bar chart for common words
            if common_words:
                words = list(common_words.keys())[:10]  # Top 10 words
                counts = [common_words[word] for word in words]
                
                fig.add_trace(
                    go.Bar(
                        x=counts,
                        y=words,
                        orientation='h'
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                title_text="Whitefonting Detection Results",
                height=500,
                width=900
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive whitefonting visualization saved to {save_path}")
                return save_path
            
            return fig
        else:
            # Create static visualization with matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Create pie chart for turnover prediction
            ax1.pie(
                [stay_probability, leave_probability],
                labels=["Stay", "Leave"],
                autopct='%1.1f%%',
                colors=['#36a2eb', '#ff6384'],
                explode=(0, 0.1),  # Explode the 'Leave' slice
                shadow=True,
                startangle=90
            )
            ax1.set_title("Turnover Prediction")
            
            # Create bar chart for employment metrics
            metrics = ["Avg. Tenure (years)", "Job Count", "Job Change Freq."]
            values = [avg_tenure, job_count, job_changing_frequency]
            
            # Add reference values for comparison
            ref_values = [4.0, 3.0, 0.25]  # Example reference values
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax2.barh(x - width/2, values, width, label='Candidate')
            ax2.barh(x + width/2, ref_values, width, label='Average', alpha=0.7)
            
            ax2.set_yticks(x)
            ax2.set_yticklabels(metrics)
            ax2.set_xlabel("Value")
            ax2.set_title("Employment Metrics")
            ax2.legend()
            
            plt.tight_layout()
            plt.suptitle("Turnover Prediction Analysis", y=1.05)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Static turnover visualization saved to {save_path}")
                return save_path
            
            # Return image as bytes if no save path
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return buf
    
    def visualize_batch_results(self, batch_results, interactive=False, save_path=None):
        """
        Create visualizations for batch analysis results.
        
        Args:
            batch_results (list): Results from batch analysis.
            interactive (bool): Whether to create interactive plotly visualizations.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            str or plotly.graph_objects.Figure: Path to saved visualization or plotly figure.
        """
        # Extract data
        total_resumes = len(batch_results)
        
        # Count whitefonting detection results
        whitefonting_counts = {
            'Detected': sum(1 for r in batch_results 
                        if r.get('whitefonting_detection', {}).get('has_white_text', False)),
            'None Detected': sum(1 for r in batch_results 
                             if not r.get('whitefonting_detection', {}).get('has_white_text', False))
        }
        
        # Count turnover risk
        turnover_counts = {
            'High Risk': sum(1 for r in batch_results 
                         if r.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False)),
            'Low Risk': sum(1 for r in batch_results 
                         if not r.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False))
        }
        
        # Count category distribution
        category_counts = {}
        for result in batch_results:
            category = result.get('classification', {}).get('predicted_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if interactive:
            # Create interactive visualization with plotly
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"}]],
                subplot_titles=("White Text Detection", "Turnover Risk", "Category Distribution")
            )
            
            # Add pie chart for whitefonting detection
            fig.add_trace(
                go.Pie(
                    labels=list(whitefonting_counts.keys()),
                    values=list(whitefonting_counts.values()),
                    marker_colors=['#ff6384', '#36a2eb']
                ),
                row=1, col=1
            )
            
            # Add pie chart for turnover risk
            fig.add_trace(
                go.Pie(
                    labels=list(turnover_counts.keys()),
                    values=list(turnover_counts.values()),
                    marker_colors=['#ff6384', '#36a2eb']
                ),
                row=1, col=2
            )
            
            # Add pie chart for category distribution
            fig.add_trace(
                go.Pie(
                    labels=list(category_counts.keys()),
                    values=list(category_counts.values())
                ),
                row=1, col=3
            )
            
            fig.update_layout(
                title_text=f"Batch Analysis Results ({total_resumes} Resumes)",
                height=500,
                width=1200
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive batch visualization saved to {save_path}")
                return save_path
            
            return fig
        else:
            # Create static visualization with matplotlib
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            
            # Create pie chart for whitefonting detection
            ax1.pie(
                list(whitefonting_counts.values()),
                labels=list(whitefonting_counts.keys()),
                autopct='%1.1f%%',
                colors=['#ff6384', '#36a2eb'],
                startangle=90
            )
            ax1.set_title("White Text Detection")
            
            # Create pie chart for turnover risk
            ax2.pie(
                list(turnover_counts.values()),
                labels=list(turnover_counts.keys()),
                autopct='%1.1f%%',
                colors=['#ff6384', '#36a2eb'],
                startangle=90
            )
            ax2.set_title("Turnover Risk")
            
            # Create pie chart for category distribution
            ax3.pie(
                list(category_counts.values()),
                labels=list(category_counts.keys()),
                autopct='%1.1f%%',
                startangle=90
            )
            ax3.set_title("Category Distribution")
            
            plt.tight_layout()
            plt.suptitle(f"Batch Analysis Results ({total_resumes} Resumes)", y=1.05)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Static batch visualization saved to {save_path}")
                return save_path
            
            # Return image as bytes if no save path
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return buf
    
    def create_dashboard(self, analysis_results, output_path, title="Resume Analysis Dashboard"):
        """
        Create an interactive HTML dashboard with analysis results.
        
        Args:
            analysis_results (dict): Results from all analysis modules.
            output_path (str): Path to save the dashboard HTML file.
            title (str): Title for the dashboard.
            
        Returns:
            str: Path to the generated dashboard.
        """
        # Extract data
        document_info = analysis_results.get('document_info', {})
        filename = os.path.basename(document_info.get('file_path', 'Unknown'))
        
        whitefonting_results = analysis_results.get('whitefonting_detection', {})
        classification_results = analysis_results.get('classification', {})
        turnover_results = analysis_results.get('turnover_prediction', {})
        
        # Generate individual visualizations
        whitefonting_fig = self.visualize_whitefonting(whitefonting_results, interactive=True)
        classification_fig = self.visualize_classification(classification_results, interactive=True)
        turnover_fig = self.visualize_turnover_prediction(turnover_results, interactive=True)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
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
                .low-risk {{
                    color: #2ecc71;
                }}
                .warning {{
                    color: #e74c3c;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>Resume Analysis for: {filename}</p>
            </div>
            
            <div class="container">
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="summary">
        """
        
        # Add summary items
        has_white_text = whitefonting_results.get('has_white_text', False)
        white_text_pct = whitefonting_results.get('white_text_percentage', 0)
        
        predicted_category = classification_results.get('predicted_category', 'Unknown')
        confidence = classification_results.get('confidence', 0) * 100
        
        turnover_risk = turnover_results.get('prediction', {}).get('will_leave', False)
        leave_probability = turnover_results.get('prediction', {}).get('leave_probability', 0) * 100
        
        html_content += f"""
                        <div class="summary-item">
                            <h3>Whitefonting</h3>
                            <p class="{'warning' if has_white_text else ''}">{'Detected' if has_white_text else 'None Detected'}</p>
                            <p>{white_text_pct:.1f}% of text is hidden</p>
                        </div>
                        
                        <div class="summary-item">
                            <h3>Category</h3>
                            <p>{predicted_category}</p>
                            <p>Confidence: {confidence:.1f}%</p>
                        </div>
                        
                        <div class="summary-item">
                            <h3>Turnover Risk</h3>
                            <p class="{'high-risk' if turnover_risk else 'low-risk'}">{'High' if turnover_risk else 'Low'}</p>
                            <p>Probability: {leave_probability:.1f}%</p>
                        </div>
                    </div>
                </div>
        """
        
        # Add whitefonting section if relevant figures exist
        if whitefonting_fig:
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
        
        # Add classification section if relevant figures exist
        if classification_fig:
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
        
        # Add turnover section if relevant figures exist
        if turnover_fig:
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
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Interactive dashboard saved to {output_path}")
        return output_path
        # Create pie chart for white/visible text distribution
        ax1.pie(
            [100 - white_text_percentage, white_text_percentage],
            labels=["Visible Text", "Hidden White Text"],
            autopct='%1.1f%%',
            colors=['#36a2eb', '#ff6384'],
            startangle=90
        )
        ax1.set_title("White Text Distribution")
        
        # Create bar chart for common words
        if common_words:
            words = list(common_words.keys())[:10]  # Top 10 words
            counts = [common_words[word] for word in words]
            
            ax2.barh(words, counts)
            ax2.set_xlabel("Count")
            ax2.set_title("Most Common Words in Hidden Text")
        
        plt.tight_layout()
        plt.suptitle("Whitefonting Detection Results", y=1.05)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Static whitefonting visualization saved to {save_path}")
            return save_path
        
        # Return image as bytes if no save path
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        
        return buf
    
    def visualize_classification(self, classification_results, interactive=False, save_path=None):
        """
        Create visualizations for resume classification results.
        
        Args:
            classification_results (dict): Results from resume classification.
            interactive (bool): Whether to create interactive plotly visualizations.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            str or plotly.graph_objects.Figure: Path to saved visualization or plotly figure.
        """
        # Extract data
        predicted_category = classification_results.get('predicted_category', 'Unknown')
        confidence = classification_results.get('confidence', 0)
        category_probabilities = classification_results.get('category_probabilities', {})
        
        if not category_probabilities:
            logger.warning("No category probabilities available for visualization")
            return None
        
        if interactive:
            # Create interactive visualization with plotly
            categories = list(category_probabilities.keys())
            probabilities = [category_probabilities[cat] * 100 for cat in categories]
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)
            categories = [categories[i] for i in sorted_indices]
            probabilities = [probabilities[i] for i in sorted_indices]
            
            # Color the predicted category differently
            colors = ['lightgrey'] * len(categories)
            pred_index = categories.index(predicted_category)
            colors[pred_index] = '#36a2eb'
            
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=probabilities,
                    y=categories,
                    orientation='h',
                    marker_color=colors
                )
            )
            
            fig.update_layout(
                title_text=f"Resume Classification Results (Predicted: {predicted_category})",
                xaxis_title="Probability (%)",
                yaxis_title="Category",
                height=500,
                width=700
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive classification visualization saved to {save_path}")
                return save_path
            
            return fig
        else:
            # Create static visualization with matplotlib
            plt.figure(figsize=(8, 6))
            
            categories = list(category_probabilities.keys())
            probabilities = [category_probabilities[cat] * 100 for cat in categories]
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)
            categories = [categories[i] for i in sorted_indices]
            probabilities = [probabilities[i] for i in sorted_indices]
            
            # Color the predicted category differently
            colors = ['lightgrey'] * len(categories)
            pred_index = categories.index(predicted_category)
            colors[pred_index] = '#36a2eb'
            
            plt.barh(categories, probabilities, color=colors)
            plt.xlabel("Probability (%)")
            plt.title(f"Resume Classification Results (Predicted: {predicted_category})")
            plt.xlim(0, 100)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Static classification visualization saved to {save_path}")
                return save_path
            
            # Return image as bytes if no save path
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            return buf
    
    def visualize_turnover_prediction(self, turnover_results, interactive=False, save_path=None):
        """
        Create visualizations for turnover prediction results.
        
        Args:
            turnover_results (dict): Results from turnover prediction.
            interactive (bool): Whether to create interactive plotly visualizations.
            save_path (str, optional): Path to save the visualization.
            
        Returns:
            str or plotly.graph_objects.Figure: Path to saved visualization or plotly figure.
        """
        # Extract data
        prediction = turnover_results.get('prediction', {})
        employment_pattern = turnover_results.get('employment_pattern', {})
        
        stay_probability = prediction.get('stay_probability', 0.5) * 100
        leave_probability = prediction.get('leave_probability', 0.5) * 100
        
        # Extract employment pattern data
        avg_tenure = employment_pattern.get('average_tenure_years', 0)
        job_count = employment_pattern.get('job_count', 0)
        job_changing_frequency = employment_pattern.get('job_changing_frequency', 0)
        
        if interactive:
            # Create interactive visualization with plotly
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "bar"}]],
                subplot_titles=("Turnover Prediction", "Employment Metrics")
            )
            
            # Add pie chart for turnover prediction
            fig.add_trace(
                go.Pie(
                    labels=["Stay", "Leave"],
                    values=[stay_probability, leave_probability],
                    marker_colors=['#36a2eb', '#ff6384']
                ),
                row=1, col=1
            )
            
            # Add bar chart for employment metrics
            metrics = ["Avg. Tenure (years)", "Job Count", "Job Change Freq."]
            values = [avg_tenure, job_count, job_changing_frequency]
            
            # Add reference values for comparison
            ref_values = [4.0, 3.0, 0.25]  # Example reference values
            
            fig.add_trace(
                go.Bar(name="Candidate", y=metrics, x=values, orientation='h'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(name="Average", y=metrics, x=ref_values, orientation='h', opacity=0.7),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text="Turnover Prediction Analysis",
                height=500,
                width=900,
                barmode='group'
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive turnover visualization saved to {save_path}")
                return save_path
            
            return fig
        else:
            # Create static visualization with matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))