# Data Processing Modules Documentation

## DocumentLoader
**Purpose**: Handles loading of documents in various formats and extracting metadata.

**Supported Formats**:
- PDF
- DOCX
- TXT

**Key Methods**:
- `load_document(file_path)`: Loads a single document based on its file type.
- `_load_pdf(file_path)`: Extracts metadata and initializes PDF documents.
- `_load_docx(file_path)`: Extracts text content from DOCX files.
- `_load_txt(file_path)`: Reads text content from TXT files.
- `batch_load(directory_path, file_types=None)`: Loads multiple documents from a directory.

**Technologies Used**:
- PyMuPDF (fitz) for PDF handling
- docx2txt for DOCX processing
- Standard Python file handling for TXT files

---

## FeatureExtractor
**Purpose**: Extracts structured features from text content for downstream analysis.

**Key Features**:
- Word and character count
- Skill extraction
- Education context extraction
- Experience section parsing
- Employment duration calculation

**Key Methods**:
- `extract_features(text)`: Extracts general features from text.
- `extract_skills(text)`: Identifies common skills from text.
- `extract_education(text)`: Extracts educational qualifications and context.
- `extract_experience(text)`: Parses experience sections from text.
- `extract_employment_durations(text)`: Identifies and calculates employment durations.

**Technologies Used**:
- NLTK for text tokenization and stopword filtering
- Regular expressions for pattern matching

---

## TextExtractor
**Purpose**: Extracts text content from documents, including OCR-based extraction for PDFs.

**Key Features**:
- Text extraction from PDF pages
- OCR-based extraction for non-textual PDF content
- Configurable confidence threshold for OCR

**Key Methods**:
- `extract_text(document)`: Extracts text content from a document.
- `_extract_from_pdf(document)`: Extracts text from PDF pages.
- `_ocr_page(page)`: Performs OCR on a single PDF page.

**Technologies Used**:
- PyMuPDF (fitz) for PDF parsing
- Pytesseract for OCR
- PIL for image preprocessing during OCR operations

---

## Summary
The data processing modules form the backbone of the Resume Analysis System, enabling efficient document handling, feature extraction, and text processing. Each module is designed to handle specific aspects of the pipeline, ensuring modularity and scalability for diverse use cases.
