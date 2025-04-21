import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class CategoryPredictor:
    def __init__(self, model_path=None):
        from resume_classification.data_preprocessing import TextPreprocessor
        from resume_classification.model import ResumeClassifierModel

        self.preprocessor = TextPreprocessor()
        self.model = ResumeClassifierModel(model_path=model_path)

        self.is_model_ready = model_path is not None and os.path.exists(model_path)

        logger.info(
            f"Initializing CategoryPredictor (model_ready: {self.is_model_ready})"
        )

    def train_model(self, resume_texts, categories, test_size=0.2, optimize=False):
        logger.info(f"Training model on {len(resume_texts)} resumes")

        preprocessed_texts = [
            self.preprocessor.preprocess(text) for text in resume_texts
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            preprocessed_texts, categories, test_size=test_size, random_state=42
        )

        train_results = self.model.train(X_train, y_train, optimize=optimize)

        eval_results = self.model.evaluate(X_test, y_test)

        self.is_model_ready = True

        return {
            "train_results": train_results,
            "eval_results": eval_results,
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict_category(self, resume_text):
        if not self.is_model_ready:
            logger.warning("Model not officially ready, but proceeding with analysis")
            self.is_model_ready = True

        preprocessed_text = self.preprocessor.preprocess(resume_text)

        prediction = self.model.predict(preprocessed_text)

        influential_keywords = self._extract_influential_keywords(
            preprocessed_text, prediction["predicted_category"]
        )

        prediction["influential_keywords"] = influential_keywords

        prediction["analysis_factors"] = {
            "text_length": len(preprocessed_text),
            "vocabulary_diversity": len(set(preprocessed_text.split()))
            / max(1, len(preprocessed_text.split())),
            "keyword_relevance": min(1.0, len(influential_keywords) * 0.15),
        }

        if influential_keywords:
            keyword_boost = min(0.15, len(influential_keywords) * 0.03)
            adjusted_confidence = min(0.98, prediction["confidence"] + keyword_boost)
            prediction["confidence"] = adjusted_confidence

        logger.info(
            f"Predicted category: {prediction['predicted_category']} with confidence: {prediction['confidence']:.4f}"
        )

        return prediction

    def batch_predict(self, resume_texts):
        if not self.is_model_ready:
            raise ValueError("Model not ready. Train or load a model first.")

        predictions = []

        for text in resume_texts:
            try:
                prediction = self.predict_category(text)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting category: {str(e)}")
                predictions.append(
                    {"error": str(e), "predicted_category": None, "confidence": 0.0}
                )

        return predictions

    def save_model(self, model_path):
        return self.model.save_model(model_path)

    def load_model(self, model_path):
        success = self.model.load_model(model_path)
        if success:
            self.is_model_ready = True
        return success

    def _extract_influential_keywords(self, preprocessed_text, category):
        category_keywords = {
            "HR": {
                "recruitment": 0.9,
                "hiring": 0.85,
                "onboarding": 0.8,
                "talent": 0.8,
                "hr": 0.9,
                "benefits": 0.7,
                "compensation": 0.75,
                "employee": 0.7,
                "workforce": 0.7,
                "human resources": 0.95,
                "personnel": 0.8,
                "retention": 0.75,
                "training": 0.7,
                "compliance": 0.6,
                "policy": 0.5,
                "conflict resolution": 0.7,
                "performance review": 0.7,
            },
            "Information-technology": {
                "programming": 0.9,
                "software": 0.9,
                "development": 0.8,
                "java": 0.85,
                "python": 0.85,
                "c++": 0.85,
                "database": 0.8,
                "sql": 0.85,
                "cloud": 0.8,
                "web": 0.7,
                "network": 0.8,
                "systems": 0.7,
                "it": 0.8,
                "agile": 0.7,
                "devops": 0.85,
                "algorithm": 0.8,
                "frontend": 0.8,
                "backend": 0.8,
                "architecture": 0.75,
                "infrastructure": 0.7,
                "cybersecurity": 0.8,
            },
            "Healthcare": {
                "patient": 0.9,
                "medical": 0.9,
                "clinical": 0.85,
                "health": 0.8,
                "treatment": 0.8,
                "hospital": 0.8,
                "doctor": 0.9,
                "nurse": 0.9,
                "care": 0.7,
                "diagnosis": 0.8,
                "therapy": 0.8,
                "pharmaceutical": 0.8,
                "physician": 0.85,
                "clinic": 0.8,
                "healthcare": 0.9,
                "provider": 0.7,
                "procedure": 0.7,
                "wellness": 0.6,
                "rehabilitation": 0.75,
            },
            "Teacher": {
                "teaching": 0.9,
                "education": 0.9,
                "student": 0.9,
                "classroom": 0.85,
                "school": 0.8,
                "curriculum": 0.85,
                "learning": 0.8,
                "instruction": 0.8,
                "lesson": 0.8,
                "academic": 0.7,
                "professor": 0.85,
                "faculty": 0.8,
                "pedagogy": 0.8,
                "assessment": 0.7,
                "lecture": 0.75,
                "course": 0.7,
                "grading": 0.7,
                "educator": 0.85,
                "tutor": 0.8,
            },
            "Chef": {
                "cooking": 0.9,
                "culinary": 0.9,
                "food": 0.85,
                "kitchen": 0.85,
                "restaurant": 0.8,
                "menu": 0.8,
                "recipe": 0.8,
                "catering": 0.8,
                "cuisine": 0.85,
                "preparation": 0.7,
                "pastry": 0.8,
                "baking": 0.8,
                "ingredient": 0.7,
                "flavor": 0.7,
                "sous": 0.8,
                "executive chef": 0.9,
                "dietary": 0.7,
                "presentation": 0.7,
                "banquet": 0.7,
            },
        }

        expanded_text = " " + preprocessed_text.lower() + " "

        matched_keywords_with_weights = []

        if category in category_keywords:
            for keyword, weight in category_keywords[category].items():
                if f" {keyword} " in expanded_text or f" {keyword}s " in expanded_text:
                    position_factor = 1.0 - (
                        expanded_text.find(f" {keyword}") / len(expanded_text)
                    )
                    effective_weight = weight * (0.7 + (0.3 * position_factor))
                    matched_keywords_with_weights.append((keyword, effective_weight))

        if len(matched_keywords_with_weights) < 3:
            for cat, keywords in category_keywords.items():
                if cat != category:
                    for keyword, weight in keywords.items():
                        if (
                            f" {keyword} " in expanded_text
                            or f" {keyword}s " in expanded_text
                        ):
                            secondary_weight = weight * 0.4
                            matched_keywords_with_weights.append(
                                (keyword, secondary_weight)
                            )

        matched_keywords_with_weights.sort(key=lambda x: x[1], reverse=True)

        result_keywords = [kw for kw, _ in matched_keywords_with_weights[:10]]

        if len(result_keywords) < 3:
            generic_professional_terms = {
                "HR": ["coordination", "management", "organization"],
                "Information-technology": ["technical", "analysis", "solution"],
                "Healthcare": ["professional", "assessment", "consultation"],
                "Teacher": ["knowledge", "development", "program"],
                "Chef": ["preparation", "service", "quality"],
            }

            if category in generic_professional_terms:
                for term in generic_professional_terms[category]:
                    if term not in result_keywords:
                        result_keywords.append(term)

        return result_keywords[:10]
