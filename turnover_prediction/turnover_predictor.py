import pandas as pd
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)
analysis_logger = logging.getLogger("analysis")

class TurnoverPredictor:
    def __init__(self, model_path=None):
        from turnover_prediction.employment_analyzer import EmploymentAnalyzer
        from turnover_prediction.model import TurnoverPredictionModel

        self.employment_analyzer = EmploymentAnalyzer()
        self.model = TurnoverPredictionModel(model_path=model_path)

        self.is_model_ready = model_path is not None and os.path.exists(model_path)

        logger.info(
            f"Initializing TurnoverPredictor (model_ready: {self.is_model_ready})"
        )

    def extract_features(self, resume_text):
        employment_records = self.employment_analyzer.extract_employment_history(
            resume_text
        )
                
        if len(employment_records) == 0:
            logger.warning("No employment records found, skipping pattern analysis.")
            pattern_analysis = {
                "average_tenure": 0,
                "average_tenure_years": 0,
                "total_experience": 0,
                "total_experience_years": 0,
                "job_count": 0,
                "job_changing_frequency": 0,
                "has_gaps": False,
                "gap_count": 0,
                "gap_details": [],
            }
        else:
            pattern_analysis = self.employment_analyzer.analyze_employment_patterns(
                employment_records
            )

        features = {
            "average_tenure": pattern_analysis["average_tenure"],
            "total_experience": pattern_analysis["total_experience"],
            "job_count": pattern_analysis["job_count"],
            "job_changing_frequency": pattern_analysis["job_changing_frequency"],
            "has_gaps": 1 if pattern_analysis["has_gaps"] else 0,
            "gap_count": pattern_analysis["gap_count"],
        }

        if pattern_analysis["gap_count"] > 0:
            gap_durations = [
                gap["duration_months"] for gap in pattern_analysis["gap_details"]
            ]
            features["avg_gap_duration"] = np.mean(gap_durations)
        else:
            features["avg_gap_duration"] = 0

        features_df = pd.DataFrame([features])

        return features_df, employment_records, pattern_analysis

    def predict_turnover(self, resume_text):
        if not self.is_model_ready:
            raise ValueError("Model not ready. Train or load a model first.")

        features, employment_records, pattern_analysis = self.extract_features(
            resume_text
        )
        print("\n", features, "\n")

        prediction = self.model.predict(features)

        feature_importance = self._analyze_feature_importance(features)

        contextual_analysis = self._generate_contextual_analysis(
            prediction, pattern_analysis
        )

        results = {
            "prediction": prediction,
            "employment_pattern": pattern_analysis,
            "feature_importance": feature_importance,
            "contextual_analysis": contextual_analysis,
        }
        analysis_logger.info(f"{results}")

        return results

    def train_model(
        self, employment_data, turnover_labels, test_size=0.2, optimize=False
    ):
        logger.info(f"Training model on {len(employment_data)} employment records")

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            employment_data, turnover_labels, test_size=test_size, random_state=42
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

    def save_model(self, model_path):
        return self.model.save_model(model_path)

    def load_model(self, model_path):
        success = self.model.load_model(model_path)
        if success:
            self.is_model_ready = True
        return success

    def _analyze_feature_importance(self, features):
        importance = {}

        for column in features.columns:
            if features[column].dtype == 'int64':
                value = int(features.iloc[0][column])
            elif features[column].dtype == 'float64':
                value = float(features.iloc[0][column])
            else:
                value = features.iloc[0][column]

            if column == "job_changing_frequency":
                importance[column] = {
                    "value": value,
                    "importance": (
                        "high" if value > 0.5 else "medium" if value > 0.2 else "low"
                    ),
                    "interpretation": (
                        "High job changing frequency"
                        if value > 0.5
                        else "Moderate job stability"
                    ),
                }
            elif column == "average_tenure":
                avg_tenure_years = value / 12
                importance[column] = {
                    "value": avg_tenure_years,
                    "unit": "years",
                    "importance": (
                        "high"
                        if avg_tenure_years < 1.5
                        else "medium" if avg_tenure_years < 3 else "low"
                    ),
                    "interpretation": (
                        "Short average tenure"
                        if avg_tenure_years < 1.5
                        else (
                            "Moderate tenure"
                            if avg_tenure_years < 3
                            else "Long average tenure"
                        )
                    ),
                }
            elif column == "has_gaps":
                # The value is a Python int at this point
                importance[column] = {
                    "value": bool(value),
                    "importance": "medium" if value else "low",
                    "interpretation": (
                        "Employment gaps present" if value else "No employment gaps"
                    ),
                }
            else:
                importance[column] = {
                    "value": value,
                    "importance": "medium",
                    "interpretation": f"{column}: {value}",
                }

        return importance

    def _generate_contextual_analysis(self, prediction, pattern_analysis):
        leave_probability = prediction["leave_probability"]
        avg_tenure = pattern_analysis["average_tenure_years"]
        job_count = pattern_analysis["job_count"]
        job_changing_frequency = pattern_analysis["job_changing_frequency"]

        if leave_probability > 0.7:
            risk_level = "High"
        elif leave_probability > 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        insights = []

        if avg_tenure < 1.5:
            insights.append("Short average job tenure suggests potential flight risk.")
        elif avg_tenure > 4:
            insights.append("Long average tenure indicates stability and commitment.")

        if job_changing_frequency > 0.5:
            insights.append(
                "Frequent job changes may indicate a pattern of short-term employment."
            )
        elif job_changing_frequency < 0.2 and job_count > 2:
            insights.append(
                "Infrequent job changes with multiple positions shows career stability."
            )

        if pattern_analysis["has_gaps"]:
            if pattern_analysis["gap_count"] > 1:
                insights.append(
                    f"Multiple employment gaps ({pattern_analysis['gap_count']}) may indicate career instability."
                )
            else:
                insights.append(
                    "Employment gap detected, consider exploring the reason during interview."
                )

        if risk_level == "High":
            recommendations = [
                "Consider discussing career goals and expectations clearly during interview.",
                "Explore commitment to long-term projects.",
                "Discuss specific retention strategies if hired.",
            ]
        elif risk_level == "Medium":
            recommendations = [
                "Assess fit with company culture and growth opportunities.",
                "Explore reasons for previous job changes during interview.",
                "Consider offering growth path to encourage retention.",
            ]
        else:
            recommendations = [
                "Candidate shows stable employment history.",
                "Focus on long-term career development during interview.",
                "Highlight company stability and growth opportunities.",
            ]

        return {
            "risk_level": risk_level,
            "insights": insights,
            "recommendations": recommendations,
        }
