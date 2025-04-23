# Resume Analysis System Documentation

## Overview
The Resume Analysis System is a comprehensive tool designed to analyze resumes for various characteristics including semantic manipulation (whitefonting), job category classification, and turnover risk prediction. The system combines multiple NLP and machine learning techniques to provide insights into resume content.

## Components

### 1. Main Application (`app.py`)
**Purpose**: Orchestrates the entire resume analysis workflow.

**Workflow:**
1. Document loading and text extraction
2. Feature extraction
3. Whitefonting detection (semantic and visual)
4. Resume Classification
5. Turnover Prediction
6. Reporting and Visualization

### 1. Document Processing Module
**Purpose**: Handles the complete document processing pipeline from file loading to feature extraction.

**Key Features:**
- Multi-format Support
- Intelligent Text Extraction
- Metadata Extraction
- Feature Extraction

**Technologies Used:**:
- PyMuPDF (fitz)
- docx2txt
- Pytesseract (Tesseract OCR) for image-based text extraction
- PIL for image processing during OCR operations
- NLTK for advanced text processing and tokenization

### 2. Whitefonting Detection
**Purpose**: Detects white text, and semantic manipulation in resumes by comparing visible and hidden text content.

**Key Features:**
- White Fonting Detection
- Heatmap Generation
- Term frequency analysis
- Semantic similarity comparison
- Industry/skill term matching
- Pattern detection (keyword stuffing, etc.)
- Intent classification
- Context relevance analysis

**Technologies Used:**
- Vision Transformers for Heatmaps
- spaCy for NLP processing
- BERT (transformers) for semantic embeddings
- scikit-learn for TF-IDF and cosine similarity
- NLTK for text preprocessing
- PhraseMatcher for industry term matching

### 3. Category Prediction
**Purpose**: Classifies resumes into job categories.

**Key Features:**
- Text preprocessing
- Category prediction
- Influential keyword extraction
- Confidence scoring
- Batch prediction

**Technologies Used:**
- scikit-learn - Random Forest, Naive Bayes, Logistic Regression
- NLTK - WordNetLemmatizer

### 4. Turnover Prediction
**Purpose**: Predicts likelihood of candidate turnover based on employment history.

**Key Features:**
- Employment pattern extraction
- Feature engineering
- Turnover probability prediction
- Contextual risk analysis
- Feature importance analysis

**Technologies Used:**
- scikit-learn - Random Forest, Logistic Regression, Gradient Boosting, SVM
- Pandas for data handling
- Custom employment pattern analyzer

### Configuration (`config.yaml`)
**Purpose**: Centralized system configuration.

**Configurable Aspects:**
- Output directories
- Module enable/disable switches
- Model paths
- Threshold values
- Visualization preferences

## Output
The system generates:
- PDF reports
- Interactive dashboards
- Visual heatmaps (for whitefonting detection)
- Log files for debugging