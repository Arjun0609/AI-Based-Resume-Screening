# Resume Classification Modules Documentation

## CategoryPredictor
**Purpose**: Predicts job categories for resumes using machine learning models.

**Key Features**:
- Preprocessing of resume text
- Training and evaluation of classification models
- Batch prediction for multiple resumes
- Extraction of influential keywords for category analysis

**Key Methods**:
- `train_model(resume_texts, categories, test_size=0.2, optimize=False)`: Trains the classification model.
- `predict_category(resume_text)`: Predicts the job category for a single resume.
- `batch_predict(resume_texts)`: Predicts job categories for multiple resumes.
- `save_model(model_path)`: Saves the trained model to a file.
- `load_model(model_path)`: Loads a pre-trained model from a file.

**Technologies Used**:
- scikit-learn for machine learning
- Custom preprocessing pipeline for text normalization

---

## TextPreprocessor
**Purpose**: Prepares resume text for classification by cleaning and extracting features.

**Key Features**:
- Removal of stopwords
- Tokenization and normalization
- Feature extraction for classification

**Key Methods**:
- `preprocess(text)`: Cleans and tokenizes text.
- `extract_features(text)`: Extracts features for classification.

**Technologies Used**:
- NLTK for text processing

---

## ResumeClassifierModel
**Purpose**: Implements machine learning models for resume classification.

**Key Features**:
- Support for multiple model types (e.g., Random Forest, Logistic Regression)
- Training and evaluation of models
- Prediction of job categories

**Key Methods**:
- `train(X_train, y_train, optimize=False)`: Trains the model on labeled data.
- `evaluate(X_test, y_test)`: Evaluates the model's performance.
- `predict(resume_text)`: Predicts the job category for a given resume.
- `save_model(model_path)`: Saves the trained model to a file.
- `load_model(model_path)`: Loads a pre-trained model from a file.

**Technologies Used**:
- scikit-learn for machine learning
- Custom pipelines for feature extraction and model training

---

## Summary
The resume classification modules enable accurate categorization of resumes into predefined job categories. By combining preprocessing, feature extraction, and machine learning, these modules provide a robust solution for automated resume analysis.
