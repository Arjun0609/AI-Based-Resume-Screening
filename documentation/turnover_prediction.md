# Turnover Prediction Modules Documentation

## EmploymentAnalyzer
**Purpose**: Extracts and analyzes employment history from resumes.

**Key Features**:
- Extraction of employment records, including job titles and companies
- Parsing of date ranges to calculate employment durations
- Detection of employment gaps
- Analysis of employment patterns (e.g., average tenure, job-changing frequency)

**Key Methods**:
- `extract_employment_history(resume_text)`: Extracts employment records from resume text.
- `_parse_date_range(date_range)`: Parses date ranges and calculates durations.
- `_extract_job_title(context)`: Extracts job titles from context.
- `_extract_company(context)`: Extracts company names from context.
- `analyze_employment_patterns(employment_records)`: Analyzes employment patterns.
- `_detect_employment_gaps(employment_records)`: Detects gaps between employment periods.

**Technologies Used**:
- Regular expressions for pattern matching
- NumPy for statistical calculations
- Pandas for data manipulation
- Dateutil for date parsing

---

## TurnoverPredictionModel
**Purpose**: Implements machine learning models for turnover prediction.

**Key Features**:
- Preprocessing of features for training and prediction
- Training and evaluation of turnover prediction models
- Feature importance analysis

**Key Methods**:
- `train(X_train, y_train, optimize=False)`: Trains the turnover prediction model.
- `evaluate(X_test, y_test)`: Evaluates the model's performance.
- `predict(features)`: Predicts turnover likelihood based on features.
- `save_model(model_path)`: Saves the trained model to a file.
- `load_model(model_path)`: Loads a pre-trained model from a file.

**Technologies Used**:
- scikit-learn for machine learning
- Custom pipelines for feature preprocessing

---

## TurnoverPredictor
**Purpose**: Combines feature extraction and machine learning for turnover prediction.

**Key Features**:
- Extraction of features from resume text
- Prediction of turnover likelihood
- Contextual analysis of employment patterns

**Key Methods**:
- `extract_features(resume_text)`: Extracts features for turnover prediction.
- `predict_turnover(resume_text)`: Predicts turnover likelihood for a given resume.
- `train_model`: Trains the turnover prediction model.
- `save_model(model_path)`: Saves the trained model to a file.
- `load_model(model_path)`: Loads a pre-trained model from a file.

**Technologies Used**:
- scikit-learn for machine learning
- Custom feature extraction pipeline

---

## Summary
The turnover prediction modules enable the analysis of employment history and prediction of turnover likelihood. By combining feature extraction, employment pattern analysis, and machine learning, these modules provide valuable insights into candidate stability and risk factors.
