# visualization/report_generator.py
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import io

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates comprehensive reports from analysis results.
    
    This class creates reports that combine results from all modules
    including whitefonting detection, resume classification, and
    turnover prediction.
    """
    
    def __init__(self, output_dir="reports"):
        """
        Initialize the ReportGenerator.
        
        Args:
            output_dir (str): Directory where reports will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initializing ReportGenerator (output_dir: {output_dir})")
    
    def generate_comprehensive_report(self, resume_analysis_results, include_visualizations=True):
        """
        Generate a comprehensive report combining all analysis results.
        
        Args:
            resume_analysis_results (dict): Combined results from all analysis modules.
            include_visualizations (bool): Whether to include visualizations.
            
        Returns:
            str: Path to the generated report.
        """
        # Extract document info
        document_info = resume_analysis_results.get('document_info', {})
        filename = document_info.get('file_path', 'Unknown Document')
        
        # Create a unique filename for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.basename(filename).replace('.pdf', '')
        report_filename = f"{base_filename}_analysis_report_{timestamp}.pdf"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Create the report document
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        )
        
        section_style = ParagraphStyle(
            'Section',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8
        )
        
        # Create report elements
        elements = []
        
        # Add report title
        elements.append(Paragraph("Resume Analysis Report", title_style))
        elements.append(Spacer(1, 12))
        
        # Add document information section
        elements.append(Paragraph("Document Information", subtitle_style))
        elements.append(Paragraph(f"Filename: {os.path.basename(filename)}", styles['Normal']))
        elements.append(Paragraph(f"File Type: {document_info.get('file_type', 'Unknown').upper()}", styles['Normal']))
        elements.append(Paragraph(f"Pages: {document_info.get('page_count', 'Unknown')}", styles['Normal']))
        elements.append(Paragraph(f"Analyzed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Add executive summary
        elements.append(Paragraph("Executive Summary", subtitle_style))
        
        # Extract key results
        whitefonting_results = resume_analysis_results.get('whitefonting_detection', {})
        classification_results = resume_analysis_results.get('classification', {})
        turnover_results = resume_analysis_results.get('turnover_prediction', {})
        
        has_white_text = whitefonting_results.get('has_white_text', False)
        predicted_category = classification_results.get('predicted_category', 'Unknown')
        turnover_risk = turnover_results.get('prediction', {}).get('will_leave', False)
        turnover_probability = turnover_results.get('prediction', {}).get('leave_probability', 0) * 100
        
        # Create summary table
        summary_data = [
            ['Analysis', 'Result', 'Confidence/Details'],
            ['Whitefonting Detection', 'Detected' if has_white_text else 'None Detected', 
             f"{whitefonting_results.get('white_text_percentage', 0):.1f}% of text is hidden"],
            ['Resume Category', predicted_category, 
             f"{classification_results.get('confidence', 0) * 100:.1f}%"],
            ['Turnover Risk', 'High' if turnover_risk else 'Low', 
             f"{turnover_probability:.1f}% probability of leaving"]
        ]
        
        # Create the table
        summary_table = Table(summary_data, colWidths=[200, 150, 150])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Highlight concerning rows
            ('BACKGROUND', (0, 1), (-1, 1), colors.lightpink if has_white_text else colors.white),
            ('BACKGROUND', (0, 3), (-1, 3), colors.lightpink if turnover_risk else colors.white)
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 16))
        
        # Add whitefonting detection section
        elements.append(Paragraph("Whitefonting Detection Results", subtitle_style))
        
        if has_white_text:
            elements.append(Paragraph("Hidden text was detected in this resume, which could indicate an attempt to manipulate automated screening systems.", styles['Normal']))
            
            # Add white text content
            elements.append(Paragraph("Hidden Text Content:", section_style))
            white_text_content = whitefonting_results.get('white_text_content', 'No content available')
            elements.append(Paragraph(white_text_content, styles['Normal']))
            
            # Add keyword analysis
            potential_keywords = whitefonting_results.get('potential_keywords', [])
            if potential_keywords:
                elements.append(Paragraph("Potential Hidden Keywords:", section_style))
                keyword_text = ", ".join(potential_keywords)
                elements.append(Paragraph(keyword_text, styles['Normal']))
        else:
            elements.append(Paragraph("No hidden text was detected in this resume.", styles['Normal']))
        
        elements.append(Spacer(1, 12))
        
        # Add resume classification section
        elements.append(Paragraph("Resume Classification Results", subtitle_style))
        elements.append(Paragraph(f"This resume is classified as: {predicted_category}", styles['Normal']))
        elements.append(Paragraph(f"Classification confidence: {classification_results.get('confidence', 0) * 100:.1f}%", styles['Normal']))
        
        # Add category probabilities
        cat_probabilities = classification_results.get('category_probabilities', {})
        if cat_probabilities:
            elements.append(Paragraph("Category Probabilities:", section_style))
            
            # Create probability table
            prob_data = [['Category', 'Probability']]
            for category, probability in cat_probabilities.items():
                prob_data.append([category, f"{probability * 100:.1f}%"])
            
            # Sort by probability (descending)
            prob_data[1:] = sorted(prob_data[1:], key=lambda x: float(x[1].replace('%', '')), reverse=True)
            
            # Create the table
            prob_table = Table(prob_data, colWidths=[200, 100])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                # Highlight the highest probability
                ('BACKGROUND', (0, 1), (-1, 1), colors.lightgreen)
            ]))
            
            elements.append(prob_table)
            
        # Add influential keywords
        influential_keywords = classification_results.get('influential_keywords', [])
        if influential_keywords:
            elements.append(Paragraph("Key Terms Supporting Classification:", section_style))
            keywords_text = ", ".join(influential_keywords)
            elements.append(Paragraph(keywords_text, styles['Normal']))
        
        elements.append(Spacer(1, 12))
        
        # Add turnover prediction section
        elements.append(Paragraph("Turnover Prediction Results", subtitle_style))
        
        contextual_analysis = turnover_results.get('contextual_analysis', {})
        risk_level = contextual_analysis.get('risk_level', 'Unknown')
        
        elements.append(Paragraph(f"Turnover Risk Level: {risk_level}", styles['Normal']))
        elements.append(Paragraph(f"Probability of Leaving: {turnover_probability:.1f}%", styles['Normal']))
        
        # Add employment pattern summary
        employment_pattern = turnover_results.get('employment_pattern', {})
        if employment_pattern:
            elements.append(Paragraph("Employment Pattern Summary:", section_style))
            
            avg_tenure = employment_pattern.get('average_tenure_years', 0)
            job_count = employment_pattern.get('job_count', 0)
            job_freq = employment_pattern.get('job_changing_frequency', 0)
            has_gaps = employment_pattern.get('has_gaps', False)
            gap_count = employment_pattern.get('gap_count', 0)
            
            elements.append(Paragraph(f"Average Job Tenure: {avg_tenure:.1f} years", styles['Normal']))
            elements.append(Paragraph(f"Number of Jobs: {job_count}", styles['Normal']))
            elements.append(Paragraph(f"Job Changing Frequency: {job_freq:.2f} jobs per year", styles['Normal']))
            elements.append(Paragraph(f"Employment Gaps: {'Yes' if has_gaps else 'No'} ({gap_count} detected)", styles['Normal']))
        
        # Add insights and recommendations
        insights = contextual_analysis.get('insights', [])
        if insights:
            elements.append(Paragraph("Insights:", section_style))
            for insight in insights:
                elements.append(Paragraph(f"• {insight}", styles['Normal']))
        
        recommendations = contextual_analysis.get('recommendations', [])
        if recommendations:
            elements.append(Paragraph("Recommendations:", section_style))
            for recommendation in recommendations:
                elements.append(Paragraph(f"• {recommendation}", styles['Normal']))
        
        elements.append(Spacer(1, 12))
        
        # Include visualizations if requested
        if include_visualizations:
            elements.append(Paragraph("Visualizations", subtitle_style))
            
            # Generate visualizations
            if has_white_text:
                # Create visualization of white text distribution
                white_text_chart = self._create_white_text_visualization(whitefonting_results)
                if white_text_chart:
                    elements.append(white_text_chart)
                    elements.append(Spacer(1, 8))
            
            # Create category probability chart
            if cat_probabilities:
                category_chart = self._create_category_visualization(classification_results)
                if category_chart:
                    elements.append(category_chart)
                    elements.append(Spacer(1, 8))
            
            # Create turnover prediction visualization
            turnover_chart = self._create_turnover_visualization(turnover_results)
            if turnover_chart:
                elements.append(turnover_chart)
                elements.append(Spacer(1, 8))
        
        # Build the report
        doc.build(elements)
        
        logger.info(f"Generated comprehensive report: {report_path}")
        return report_path
    
    def _create_white_text_visualization(self, whitefonting_results):
        """Create visualization for white text analysis."""
        try:
            # Extract data
            potential_keywords = whitefonting_results.get('potential_keywords', [])
            common_words = whitefonting_results.get('common_words', {})
            
            if not common_words:
                return None
            
            # Create a bar chart of most common words
            plt.figure(figsize=(6, 4))
            words = list(common_words.keys())[:10]  # Top 10 words
            counts = [common_words[word] for word in words]
            
            plt.barh(words, counts)
            plt.xlabel('Count')
            plt.title('Most Common Words in Hidden Text')
            plt.tight_layout()
            
            # Save figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Create ReportLab Image
            buf.seek(0)
            img = Image(buf, width=400, height=300)
            
            return img
        except Exception as e:
            logger.error(f"Error creating white text visualization: {str(e)}")
            return None
    
    def _create_category_visualization(self, classification_results):
        """Create visualization for category probabilities."""
        try:
            # Extract data
            cat_probabilities = classification_results.get('category_probabilities', {})
            
            if not cat_probabilities:
                return None
            
            # Create a bar chart of category probabilities
            plt.figure(figsize=(6, 4))
            categories = list(cat_probabilities.keys())
            probabilities = [cat_probabilities[cat] * 100 for cat in categories]
            
            # Sort by probability
            sorted_indices = np.argsort(probabilities)[::-1]
            categories = [categories[i] for i in sorted_indices]
            probabilities = [probabilities[i] for i in sorted_indices]
            
            # Color the highest probability differently
            colors = ['lightgrey'] * len(categories)
            colors[0] = 'lightgreen'
            
            plt.barh(categories, probabilities, color=colors)
            plt.xlabel('Probability (%)')
            plt.title('Category Classification Probabilities')
            plt.xlim(0, 100)
            plt.tight_layout()
            
            # Save figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Create ReportLab Image
            buf.seek(0)
            img = Image(buf, width=400, height=300)
            
            return img
        except Exception as e:
            logger.error(f"Error creating category visualization: {str(e)}")
            return None
    
    def _create_turnover_visualization(self, turnover_results):
        """Create visualization for turnover prediction."""
        try:
            # Extract data
            prediction = turnover_results.get('prediction', {})
            stay_probability = prediction.get('stay_probability', 0.5) * 100
            leave_probability = prediction.get('leave_probability', 0.5) * 100
            
            # Create a pie chart of turnover probabilities
            plt.figure(figsize=(5, 4))
            labels = ['Stay', 'Leave']
            sizes = [stay_probability, leave_probability]
            colors = ['lightgreen', 'lightcoral']
            explode = (0, 0.1)  # Explode the 'Leave' slice
            
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                    shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Turnover Prediction')
            plt.tight_layout()
            
            # Save figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Create ReportLab Image
            buf.seek(0)
            img = Image(buf, width=300, height=250)
            
            return img
        except Exception as e:
            logger.error(f"Error creating turnover visualization: {str(e)}")
            return None
    
    def generate_batch_report(self, batch_results):
        """
        Generate a summary report for multiple resumes.
        
        Args:
            batch_results (list): Analysis results for multiple resumes.
            
        Returns:
            str: Path to the generated report.
        """
        # Create a unique filename for the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"batch_analysis_report_{timestamp}.pdf"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Create the report document
        doc = SimpleDocTemplate(report_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create styles
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        )
        
        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10
        )
        
        # Create report elements
        elements = []
        
        # Add report title
        elements.append(Paragraph("Batch Resume Analysis Report", title_style))
        elements.append(Spacer(1, 12))
        
        # Add batch information
        elements.append(Paragraph("Batch Information", subtitle_style))
        elements.append(Paragraph(f"Number of Resumes: {len(batch_results)}", styles['Normal']))
        elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        elements.append(Spacer(1, 12))
        
        # Add summary table
        elements.append(Paragraph("Analysis Summary", subtitle_style))
        
        # Create summary table data
        summary_data = [['Filename', 'Category', 'White Text', 'Turnover Risk']]
        
        for result in batch_results:
            filename = os.path.basename(result.get('document_info', {}).get('file_path', 'Unknown'))
            category = result.get('classification', {}).get('predicted_category', 'Unknown')
            has_white_text = result.get('whitefonting_detection', {}).get('has_white_text', False)
            turnover_risk = result.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False)
            
            summary_data.append([
                filename,
                category,
                'Yes' if has_white_text else 'No',
                'High' if turnover_risk else 'Low'
            ])
        
        # Create the table
        summary_table = Table(summary_data, colWidths=[180, 100, 80, 80])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            # Highlight rows with white text or high turnover risk
            *[('BACKGROUND', (2, i+1), (2, i+1), colors.lightpink) 
              for i, row in enumerate(summary_data[1:]) if row[2] == 'Yes'],
            *[('BACKGROUND', (3, i+1), (3, i+1), colors.lightpink) 
              for i, row in enumerate(summary_data[1:]) if row[3] == 'High']
        ]))
        
        elements.append(summary_table)
        elements.append(Spacer(1, 20))
        
        # Add statistics
        elements.append(Paragraph("Batch Statistics", subtitle_style))
        
        # Calculate statistics
        total_resumes = len(batch_results)
        resumes_with_white_text = sum(1 for r in batch_results 
                                    if r.get('whitefonting_detection', {}).get('has_white_text', False))
        resumes_with_high_turnover = sum(1 for r in batch_results 
                                       if r.get('turnover_prediction', {}).get('prediction', {}).get('will_leave', False))
        
        # Create category distribution
        category_counts = {}
        for result in batch_results:
            category = result.get('classification', {}).get('predicted_category', 'Unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Add statistics text
        elements.append(Paragraph(f"Resumes with White Text: {resumes_with_white_text} ({resumes_with_white_text/total_resumes*100:.1f}%)", styles['Normal']))
        elements.append(Paragraph(f"Resumes with High Turnover Risk: {resumes_with_high_turnover} ({resumes_with_high_turnover/total_resumes*100:.1f}%)", styles['Normal']))
        elements.append(Spacer(1, 8))
        
        # Add category distribution
        elements.append(Paragraph("Category Distribution:", styles['Normal']))
        
        # Create category table
        category_data = [['Category', 'Count', 'Percentage']]
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            category_data.append([
                category,
                str(count),
                f"{count/total_resumes*100:.1f}%"
            ])
        
        # Create the table
        category_table = Table(category_data, colWidths=[150, 80, 100])
        category_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(category_table)
        elements.append(Spacer(1, 12))
        
        # Add visualization of key statistics
        category_chart = self._create_category_distribution_chart(category_counts)
        if category_chart:
            elements.append(category_chart)
        
        # Build the report
        doc.build(elements)
        
        logger.info(f"Generated batch report: {report_path}")
        return report_path
    
    def _create_category_distribution_chart(self, category_counts):
        """Create a visualization of category distribution."""
        try:
            # Create a pie chart of category distribution
            plt.figure(figsize=(6, 4))
            
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            
            # Sort by count
            sorted_indices = np.argsort(counts)[::-1]
            categories = [categories[i] for i in sorted_indices]
            counts = [counts[i] for i in sorted_indices]
            
            plt.pie(counts, labels=categories, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Resume Category Distribution')
            plt.tight_layout()
            
            # Save figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            
            # Create ReportLab Image
            buf.seek(0)
            img = Image(buf, width=400, height=300)
            
            return img
        except Exception as e:
            logger.error(f"Error creating category distribution chart: {str(e)}")
            return None