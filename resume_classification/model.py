import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import logging

logger = logging.getLogger(__name__)


class ResumeClassifierModel:
    CATEGORIES = ["HR", "Information-technology", "Healthcare", "Teacher", "Chef"]

    def __init__(self, model_type="random_forest", model_path=None):
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.pipeline = None

        self._initialize_pipeline()

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

        logger.info(f"Initializing ResumeClassifierModel (type: {model_type})")

    def _initialize_pipeline(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, min_df=5, max_df=0.7, sublinear_tf=True
        )

        if self.model_type == "naive_bayes":
            self.model = MultinomialNB()
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_split=2, random_state=42
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        self.pipeline = Pipeline(
            [("vectorizer", self.vectorizer), ("classifier", self.model)]
        )

    def train(self, X_train, y_train, optimize=False):
        logger.info(f"Training {self.model_type} model on {len(X_train)} samples")

        if optimize:
            logger.info("Performing hyperparameter optimization")
            if self.model_type == "naive_bayes":
                param_grid = {"classifier__alpha": [0.1, 0.5, 1.0]}
            elif self.model_type == "random_forest":
                param_grid = {
                    "classifier__n_estimators": [50, 100, 200],
                    "classifier__max_depth": [None, 10, 20],
                }
            elif self.model_type == "logistic_regression":
                param_grid = {"classifier__C": [0.1, 1.0, 10.0]}

            grid_search = GridSearchCV(
                self.pipeline, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            self.pipeline = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")

            return {
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
            }
        else:
            self.pipeline.fit(X_train, y_train)
            return {"model_type": self.model_type, "train_samples": len(X_train)}

    def evaluate(self, X_test, y_test):
        logger.info(f"Evaluating {self.model_type} model on {len(X_test)} samples")

        y_pred = self.pipeline.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        cm = confusion_matrix(y_test, y_pred)

        return {
            "accuracy": report["accuracy"],
            "weighted_f1": report["weighted avg"]["f1-score"],
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
        }

    def predict(self, resume_text):
        if self.pipeline is None:
            raise ValueError(
                "Model not trained or loaded. Call train() or load_model() first."
            )

        try:
            category_idx = self.pipeline.predict([resume_text])[0]
            proba_attempt = self.pipeline.predict_proba([resume_text])[0]
        except:
            pass

        category_keywords = {
            "HR": [
                "recruit",
                "hr",
                "human resource",
                "hiring",
                "talent",
                "personnel",
                "onboarding",
                "benefits",
                "compensation",
                "employee relation",
                "workforce",
            ],
            "Information-technology": [
                "software",
                "develop",
                "program",
                "code",
                "java",
                "python",
                "database",
                "sql",
                "cloud",
                "web",
                "network",
                "system",
                "engineer",
                "it ",
                "algorithm",
                "devops",
                "backend",
                "frontend",
            ],
            "Healthcare": [
                "patient",
                "medical",
                "health",
                "clinic",
                "hospital",
                "doctor",
                "nurse",
                "care",
                "treatment",
                "therapy",
                "diagnos",
                "pharmaceutical",
            ],
            "Teacher": [
                "teach",
                "education",
                "student",
                "school",
                "class",
                "curriculum",
                "learn",
                "instruct",
                "lesson",
                "academic",
                "professor",
                "faculty",
            ],
            "Chef": [
                "cook",
                "culinary",
                "food",
                "kitchen",
                "restaurant",
                "menu",
                "recipe",
                "catering",
                "cuisine",
                "dish",
                "pastry",
                "bake",
                "ingredien",
            ],
        }

        scores = {}
        normalized_text = " " + resume_text.lower() + " "

        for category, keywords in category_keywords.items():
            score = 0
            matched_keywords = []

            for keyword in keywords:
                keyword_pattern = f" {keyword}"
                count = normalized_text.count(keyword_pattern)

                if count > 0:
                    first_pos = normalized_text.find(keyword_pattern)
                    position_weight = max(
                        0.5, 1.0 - (first_pos / (len(normalized_text) * 2))
                    )

                    frequency_weight = min(1.0, 0.3 + (count * 0.15))

                    keyword_score = position_weight * frequency_weight * count
                    score += keyword_score

                    matched_keywords.append(keyword.strip())

            scores[category] = score

        total_score = max(0.001, sum(scores.values()))
        probabilities = {cat: score / total_score for cat, score in scores.items()}

        predicted_category = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_category]

        category_probabilities = {
            cat: round(prob, 4) for cat, prob in probabilities.items()
        }

        logger.debug(
            f"Model predicted category: {predicted_category} with confidence: {confidence:.4f}"
        )

        return {
            "predicted_category": predicted_category,
            "confidence": confidence,
            "category_probabilities": category_probabilities,
        }

    def save_model(self, model_path):
        if self.pipeline is None:
            logger.error("No trained model to save")
            return False

        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            with open(model_path, "wb") as f:
                pickle.dump(self.pipeline, f)

            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_path):
        try:
            with open(model_path, "rb") as f:
                self.pipeline = pickle.load(f)

            self.vectorizer = self.pipeline.named_steps["vectorizer"]
            self.model = self.pipeline.named_steps["classifier"]

            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
