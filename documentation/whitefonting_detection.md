# Whitefonting Detection Modules Documentation

## SemanticAnalyzer
**Purpose**: Analyzes semantic manipulation in resumes by comparing visible and hidden text.

**Key Features**:
- Preprocessing of text using spaCy and NLTK
- Term frequency analysis for hidden text
- Semantic similarity comparison using transformer models
- Industry and skill term matching
- Detection of suspicious patterns and intent
- Context relevance analysis for hidden entities

**Key Methods**:
- `preprocess_text(text)`: Cleans and tokenizes text for analysis.
- `analyze_semantic_content(visible_text, hidden_text)`: Performs semantic analysis on visible and hidden text.
- `_analyze_term_frequency(processed_visible, processed_hidden, raw_visible, raw_hidden)`: Analyzes term frequency in hidden text.
- `_analyze_semantic_similarity(processed_visible, processed_hidden, raw_visible, raw_hidden)`: Compares semantic similarity between visible and hidden text.
- `_analyze_industry_terms(processed_visible, processed_hidden, raw_visible, raw_hidden)`: Matches industry and skill terms in hidden text.
- `_analyze_patterns(processed_visible, processed_hidden, raw_visible, raw_hidden)`: Detects suspicious patterns in hidden text.
- `_analyze_intent(processed_visible, processed_hidden, raw_visible, raw_hidden)`: Identifies intent behind hidden text.
- `_analyze_context_relevance(processed_visible, processed_hidden, raw_visible, raw_hidden)`: Analyzes context relevance for hidden entities.

**Technologies Used**:
- spaCy for NLP processing
- Transformers (BERT, Facebook Bart) for semantic embeddings and classification
- NLTK for text preprocessing
- scikit-learn for TF-IDF and cosine similarity
- PhraseMatcher for term matching

---

## FontAnalyzer
**Purpose**: Analyzes font properties in resumes to detect hidden text.

**Key Features**:
- Extraction of font statistics
- Identification of white text based on font properties

**Key Methods**:
- `analyze_fonts(document)`: Extracts font properties from a document.
- `get_font_statistics(font_df)`: Calculates statistics for font properties.

**Technologies Used**:
- PyMuPDF for font extraction
- Pandas for data manipulation

---

## WhiteTextDetector
**Purpose**: Detects and analyzes white text in resumes.

**Key Features**:
- Detection of white text based on font properties
- Analysis of hidden text content
- Generation of visual heatmaps for white text

**Key Methods**:
- `detect_white_text(font_df)`: Identifies white text in font data.
- `analyze_white_text(white_text_df)`: Analyzes content of white text.
- `create_visual_heatmap(document, white_text_df, output_path=None)`: Generates heatmaps for white text.

**Technologies Used**:
- PyMuPDF for font extraction
- PIL for image processing
- Pytesseract for OCR

---

## DetectionReportGenerator
**Purpose**: Generates reports for whitefonting detection results.

**Key Features**:
- Creation of detailed reports in JSON and summary formats
- Integration of visualizations into reports

**Key Methods**:
- `generate_report(document, detection_results, include_visualizations=True)`: Generates a comprehensive report.
- `generate_json_report(document, detection_results)`: Creates a JSON report.
- `generate_summary_report(batch_results)`: Summarizes results for batch analysis.

**Technologies Used**:
- JSON for structured reporting
- PyMuPDF and PIL for visualizations

---

## Summary
The whitefonting detection modules provide a comprehensive solution for identifying and analyzing hidden text in resumes. By combining semantic analysis, font property extraction, and visualization, these modules ensure transparency and authenticity in resume content.
