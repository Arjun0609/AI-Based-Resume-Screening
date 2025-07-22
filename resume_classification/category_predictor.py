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
            "Accountant": {
                "accounting": 0.95,
                "financial": 0.9,
                "audit": 0.9,
                "tax": 0.9,
                "cpa": 0.9,
                "bookkeeping": 0.85,
                "reconciliation": 0.8,
                "gaap": 0.8,
                "forecasting": 0.75,
                "budgeting": 0.75,
                "ledger": 0.7,
                "invoicing": 0.6,
                "account": 0.9
            },
            "Advocate": {
                "legal": 0.95,
                "law": 0.9,
                "attorney": 0.9,
                "litigation": 0.85,
                "counsel": 0.8,
                "court": 0.8,
                "justice": 0.75,
                "jurisprudence": 0.75,
                "pleading": 0.7,
                "verdict": 0.7,
                "compliance": 0.65,
            },
            "Agriculture": {
                "farming": 0.95,
                "agriculture": 0.9,
                "crops": 0.85,
                "livestock": 0.8,
                "agronomy": 0.8,
                "harvest": 0.75,
                "soil": 0.75,
                "irrigation": 0.7,
                "fertilizer": 0.7,
                "sustainability": 0.65,
                "tractor": 0.65,
            },
            "Apparel": {
                "fashion": 0.9,
                "apparel": 0.9,
                "textile": 0.85,
                "clothing": 0.8,
                "design": 0.8,
                "merchandising": 0.75,
                "retail": 0.7,
                "sourcing": 0.7,
                "garment": 0.7,
                "style": 0.65,
            },
            "Arts": {
                "artist": 0.9,
                "creative": 0.85,
                "art": 0.8,
                "gallery": 0.8,
                "exhibition": 0.8,
                "studio": 0.75,
                "painting": 0.7,
                "sculpture": 0.7,
                "curator": 0.7,
                "portfolio": 0.65,
            },
            "Automobile": {
                "automotive": 0.9,
                "vehicle": 0.85,
                "mechanic": 0.85,
                "engine": 0.8,
                "repair": 0.8,
                "maintenance": 0.75,
                "dealership": 0.7,
                "parts": 0.7,
                "manufacturing": 0.65,
                "assembly": 0.65,
            },
            "Aviation": {
                "pilot": 0.9,
                "aviation": 0.9,
                "flight": 0.85,
                "aircraft": 0.8,
                "aerospace": 0.8,
                "airline": 0.75,
                "air traffic": 0.75,
                "airport": 0.7,
                "avionics": 0.7,
                "maintenance": 0.65,
            },
            "Banking": {
                "banking": 0.9,
                "loan": 0.85,
                "finance": 0.8,
                "mortgage": 0.8,
                "investment": 0.75,
                "credit": 0.75,
                "teller": 0.7,
                "transaction": 0.7,
                "risk management": 0.7,
                "financial services": 0.65,
            },
            "BPO": {
                "bpo": 0.95,
                "call center": 0.9,
                "customer service": 0.85,
                "inbound": 0.8,
                "outbound": 0.8,
                "outsourcing": 0.8,
                "telemarketing": 0.75,
                "technical support": 0.7,
                "data entry": 0.65,
                "client relations": 0.6,
            },
            "Business-development": {
                "business development": 0.95,
                "strategy": 0.85,
                "growth": 0.8,
                "partnership": 0.8,
                "sales": 0.75,
                "lead generation": 0.7,
                "revenue": 0.7,
                "negotiation": 0.7,
                "client acquisition": 0.65,
                "networking": 0.6,
            },
            "Construction": {
                "construction": 0.95,
                "building": 0.85,
                "project management": 0.8,
                "site supervision": 0.8,
                "contractor": 0.75,
                "blueprint": 0.75,
                "engineering": 0.7,
                "safety": 0.7,
                "masonry": 0.65,
                "estimating": 0.6,
            },
            "Consultant": {
                "consulting": 0.95,
                "advisory": 0.9,
                "strategy": 0.85,
                "analysis": 0.8,
                "solution": 0.8,
                "implementation": 0.75,
                "business process": 0.7,
                "problem-solving": 0.65,
                "stakeholder": 0.6,
                "change management": 0.65,
            },
            "Designer": {
                "design": 0.95,
                "graphic design": 0.9,
                "ui/ux": 0.85,
                "creative": 0.8,
                "adobe": 0.75,
                "typography": 0.7,
                "branding": 0.7,
                "visual identity": 0.7,
                "layout": 0.65,
                "prototyping": 0.65,
            },
            "Digital-media": {
                "digital marketing": 0.9,
                "social media": 0.85,
                "seo": 0.8,
                "content creation": 0.8,
                "analytics": 0.75,
                "campaign": 0.75,
                "web content": 0.7,
                "video production": 0.7,
                "online advertising": 0.65,
            },
            "Engineering": {
                "engineer": 0.95,
                "engineering": 0.9,
                "design": 0.85,
                "system": 0.8,
                "civil": 0.8,
                "mechanical": 0.8,
                "electrical": 0.8,
                "software": 0.8,
                "cad": 0.75,
                "prototype": 0.7,
            },
            "Finance": {
                "finance": 0.95,
                "financial analysis": 0.9,
                "investment": 0.85,
                "budgeting": 0.8,
                "corporate finance": 0.8,
                "portfolio": 0.75,
                "trading": 0.75,
                "risk assessment": 0.7,
                "mergers and acquisitions": 0.7,
            },
            "Fitness": {
                "fitness": 0.95,
                "personal trainer": 0.9,
                "wellness": 0.85,
                "exercise": 0.8,
                "nutrition": 0.75,
                "coaching": 0.75,
                "gym": 0.7,
                "group fitness": 0.7,
                "strength training": 0.7,
                "sports science": 0.65,
            },
            "Public-relations": {
                "public relations": 0.95,
                "pr": 0.9,
                "media relations": 0.85,
                "press release": 0.8,
                "crisis communication": 0.75,
                "brand reputation": 0.75,
                "corporate communications": 0.7,
                "storytelling": 0.7,
                "event planning": 0.65,
                "publicity": 0.65,
            },
            "Sales": {
                "sales": 0.95,
                "client relations": 0.85,
                "business development": 0.8,
                "account management": 0.8,
                "lead generation": 0.75,
                "revenue": 0.75,
                "quota": 0.7,
                "crm": 0.7,
                "negotiation": 0.7,
                "customer acquisition": 0.65,
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
                "Accountant": ["financial", "auditing", "taxation"],
                "Advocate": ["legal", "litigation", "counsel"],
                "Agriculture": ["crop", "harvest", "farming"],
                "Apparel": ["fashion", "design", "merchandising"],
                "Arts": ["creative", "exhibition", "portfolio"],
                "Automobile": ["vehicle", "mechanics", "assembly"],
                "Aviation": ["flight", "aircraft", "aerospace"],
                "Banking": ["finance", "loan", "customer-service"],
                "BPO": ["outbound", "inbound", "call-center"],
                "Business-development": ["strategy", "growth", "partnership"],
                "Chef": ["preparation", "service", "quality"],
                "Construction": ["site", "project", "building"],
                "Consultant": ["advisory", "solution", "strategy"],
                "Designer": ["visual", "creative", "prototyping"],
                "Digital-media": ["content", "platform", "marketing"],
                "Engineering": ["design", "system", "technical"],
                "Finance": ["investment", "budget", "analysis"],
                "Fitness": ["training", "wellness", "coaching"],
                "Healthcare": ["professional", "assessment", "consultation"],
                "HR": ["coordination", "management", "organization"],
                "Information-technology": ["technical", "analysis", "solution"],
                "Public-relations": ["communication", "media", "brand"],
                "Sales": ["client", "revenue", "negotiation"],
                "Teacher": ["knowledge", "development", "program"]
            }

            if category in generic_professional_terms:
                for term in generic_professional_terms[category]:
                    if term not in result_keywords:
                        result_keywords.append(term)

        return result_keywords[:10]
