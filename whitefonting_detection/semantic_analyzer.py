import logging
import re
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import spacy
from spacy.matcher import PhraseMatcher, Matcher
from spacy.tokens import Doc, Span
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from gensim.models import KeyedVectors
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import defaultdict


logger = logging.getLogger(__name__)


class SemanticAnalyzer:

    def __init__(self, config=None):

        self.config = {
            "spacy_model": "en_core_web_sm",
            "use_transformers": True,
            "transformer_model": "bert-base-uncased",
            "use_nltk": True,
            "use_industry_terms": True,
            "industry_terms_path": "data/industry_terms.json",
            "skill_terms_path": "data/skills.json",
            "confidence_threshold": 0.7,
            "ngram_range": (1, 3),
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

        if config:
            self.config.update(config)

        self._init_components()

        logger.info("SemanticAnalyzer initialized")

    def _init_components(self):

        try:
            self.nlp = spacy.load(self.config["spacy_model"])
            logger.info(f"Loaded spaCy model: {self.config['spacy_model']}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            self.nlp = None

        if self.config["use_nltk"]:
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)
                self.stop_words = set(stopwords.words("english"))
                self.lemmatizer = WordNetLemmatizer()
                logger.info("NLTK components initialized")
            except Exception as e:
                logger.error(f"Failed to initialize NLTK components: {str(e)}")
                self.config["use_nltk"] = False

        if self.config["use_transformers"]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config["transformer_model"]
                )
                self.model = AutoModel.from_pretrained(self.config["transformer_model"])
                self.model.to(self.config["device"])
                self.model.eval()

                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if self.config["device"] == "cuda" else -1,
                )
                logger.info(
                    f"Loaded transformer model: {self.config['transformer_model']}"
                )
            except Exception as e:
                logger.error(f"Failed to load transformer model: {str(e)}")
                self.config["use_transformers"] = False
                self.tokenizer = None
                self.model = None
                self.classifier = None

        if self.config["use_industry_terms"]:
            try:
                with open(self.config["industry_terms_path"], "r") as f:
                    self.industry_terms = json.load(f)
                with open(self.config["skill_terms_path"], "r") as f:
                    self.skill_terms = json.load(f)
                logger.info("Loaded industry terms and skills")

                self.industry_matcher = self._create_industry_matcher()
                self.skill_matcher = self._create_skill_matcher()
            except Exception as e:
                logger.error(f"Failed to load industry terms: {str(e)}")
                self.config["use_industry_terms"] = False
                self.industry_terms = {}
                self.skill_terms = {}

    def _create_industry_matcher(self):
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        for industry, terms in self.industry_terms.items():
            patterns = [self.nlp.make_doc(term) for term in terms]
            matcher.add(f"INDUSTRY_{industry}", patterns)

        return matcher

    def _create_skill_matcher(self):
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")

        for category, skills in self.skill_terms.items():
            patterns = [self.nlp.make_doc(skill) for skill in skills]
            matcher.add(f"SKILL_{category}", patterns)

        return matcher

    def preprocess_text(self, text):
        if not text or not isinstance(text, str):
            return ""

        text = re.sub(r"\s+", " ", text)
        text = text.strip()

        if self.config["use_nltk"]:

            tokens = word_tokenize(text.lower())

            filtered_tokens = [
                self.lemmatizer.lemmatize(token)
                for token in tokens
                if token.isalnum() and token not in self.stop_words
            ]

            return " ".join(filtered_tokens)

        return text

    def analyze_semantic_content(self, visible_text, hidden_text):
        if not hidden_text or not visible_text:
            return {"has_suspicious_content": False, "confidence": 0.0, "analysis": {}}

        processed_visible = self.preprocess_text(visible_text)
        processed_hidden = self.preprocess_text(hidden_text)

        results = {"has_suspicious_content": False, "confidence": 0.0, "analysis": {}}

        analysis_methods = [
            self._analyze_term_frequency,
            self._analyze_semantic_similarity,
            self._analyze_industry_terms,
            self._analyze_patterns,
            self._analyze_intent,
            self._analyze_context_relevance,
        ]

        for method in analysis_methods:
            try:
                method_results = method(
                    processed_visible, processed_hidden, visible_text, hidden_text
                )
                results["analysis"][
                    method.__name__.replace("_analyze_", "")
                ] = method_results

                if (
                    method_results.get("suspicious", False)
                    and method_results.get("confidence", 0)
                    >= self.config["confidence_threshold"]
                ):
                    results["has_suspicious_content"] = True

                    results["confidence"] = max(
                        results["confidence"], method_results.get("confidence", 0)
                    )
            except Exception as e:
                logger.error(f"Error in {method.__name__}: {str(e)}")
                results["analysis"][method.__name__.replace("_analyze_", "")] = {
                    "error": str(e)
                }

        results["summary"] = self._generate_summary(results)

        return results

    def _analyze_term_frequency(
        self, processed_visible, processed_hidden, raw_visible, raw_hidden
    ):

        vectorizer = TfidfVectorizer(ngram_range=self.config["ngram_range"])

        if len(processed_hidden.split()) < 3:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Hidden text too short for term frequency analysis",
            }

        combined_text = [processed_visible, processed_hidden]
        tfidf_matrix = vectorizer.fit_transform(combined_text)

        feature_names = vectorizer.get_feature_names_out()

        visible_scores = tfidf_matrix[0].toarray()[0]
        hidden_scores = tfidf_matrix[1].toarray()[0]

        significant_terms = []
        for i, (v_score, h_score) in enumerate(zip(visible_scores, hidden_scores)):

            if h_score > v_score * 2 and h_score > 0.1:
                term = feature_names[i]
                significant_terms.append(
                    {
                        "term": term,
                        "visible_score": v_score,
                        "hidden_score": h_score,
                        "ratio": h_score / max(v_score, 0.001),
                    }
                )

        significant_terms.sort(
            key=lambda x: x["hidden_score"] - x["visible_score"], reverse=True
        )

        is_suspicious = len(significant_terms) >= 3

        if significant_terms:

            confidence = min(
                1.0,
                sum(term["ratio"] for term in significant_terms[:5])
                / min(5, len(significant_terms)),
            )
        else:
            confidence = 0.0

        return {
            "suspicious": is_suspicious,
            "confidence": confidence,
            "significant_terms": significant_terms[:10],
            "term_count": len(significant_terms),
            "message": f"Found {len(significant_terms)} terms with significantly higher importance in hidden text",
        }

    def _analyze_semantic_similarity(
        self, processed_visible, processed_hidden, raw_visible, raw_hidden
    ):
        if not self.config["use_transformers"] or not self.model or not self.tokenizer:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Transformer model not available for semantic similarity analysis",
            }

        if len(processed_hidden.split()) < 3:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Hidden text too short for semantic similarity analysis",
            }

        # Make sure we're dealing with strings and using safe slicing
        if not isinstance(raw_visible, str):
            raw_visible = str(raw_visible)
        if not isinstance(raw_hidden, str):
            raw_hidden = str(raw_hidden)

        # Apply safe slicing - make sure we don't exceed string length
        max_visible_len = min(len(raw_visible), 512)
        max_hidden_len = min(len(raw_hidden), 512)

        visible_embedding = self._get_embedding(raw_visible[0:max_visible_len])
        hidden_embedding = self._get_embedding(raw_hidden[0:max_hidden_len])

        similarity = cosine_similarity([visible_embedding], [hidden_embedding])[0][0]

        is_suspicious = similarity < 0.3 or similarity > 0.95

        if similarity < 0.3:
            confidence = 1.0 - similarity / 0.3
        elif similarity > 0.95:
            confidence = (similarity - 0.95) * 20
        else:
            confidence = 0.0

        return {
            "suspicious": is_suspicious,
            "confidence": confidence,
            "similarity": similarity,
            "message": (
                "Hidden text is semantically very different from visible text"
                if similarity < 0.3
                else (
                    "Hidden text is suspiciously similar to visible text"
                    if similarity > 0.95
                    else "Semantic similarity within normal range"
                )
            ),
        }

    def _analyze_industry_terms(
        self, processed_visible, processed_hidden, raw_visible, raw_hidden
    ):
        if not self.config["use_industry_terms"] or not self.nlp:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Industry terms analysis not available",
            }

        # Make sure we're dealing with strings and using safe slicing
        if not isinstance(raw_visible, str):
            raw_visible = str(raw_visible)
        if not isinstance(raw_hidden, str):
            raw_hidden = str(raw_hidden)

        # Apply safe slicing - make sure we don't exceed string length
        max_visible_len = min(len(raw_visible), 100000)
        max_hidden_len = min(len(raw_hidden), 100000)

        visible_doc = self.nlp(raw_visible[0:max_visible_len])
        hidden_doc = self.nlp(raw_hidden[0:max_hidden_len])

        visible_industry_matches = self.industry_matcher(visible_doc)
        visible_skill_matches = self.skill_matcher(visible_doc)

        hidden_industry_matches = self.industry_matcher(hidden_doc)
        hidden_skill_matches = self.skill_matcher(hidden_doc)

        visible_industry_terms = self._extract_matches(
            visible_doc, visible_industry_matches
        )
        visible_skill_terms = self._extract_matches(visible_doc, visible_skill_matches)

        hidden_industry_terms = self._extract_matches(
            hidden_doc, hidden_industry_matches
        )
        hidden_skill_terms = self._extract_matches(hidden_doc, hidden_skill_matches)

        hidden_only_industry = {
            term for term in hidden_industry_terms if term not in visible_industry_terms
        }
        hidden_only_skills = {
            term for term in hidden_skill_terms if term not in visible_skill_terms
        }

        industry_term_categories = defaultdict(int)
        skill_term_categories = defaultdict(int)

        for match_id, _, _ in hidden_industry_matches:
            category = self.nlp.vocab.strings[match_id].replace("INDUSTRY_", "")
            industry_term_categories[category] += 1

        for match_id, _, _ in hidden_skill_matches:
            category = self.nlp.vocab.strings[match_id].replace("SKILL_", "")
            skill_term_categories[category] += 1

        is_suspicious = len(hidden_only_industry) + len(hidden_only_skills) >= 5

        term_count = len(hidden_only_industry) + len(hidden_only_skills)
        confidence = min(1.0, term_count / 10)

        return {
            "suspicious": is_suspicious,
            "confidence": confidence,
            "hidden_only_industry_terms": list(hidden_only_industry),
            "hidden_only_skill_terms": list(hidden_only_skills),
            "industry_term_categories": dict(industry_term_categories),
            "skill_term_categories": dict(skill_term_categories),
            "term_count": term_count,
            "message": f"Found {term_count} industry/skill terms only in hidden text",
        }

    def _analyze_patterns(
        self, processed_visible, processed_hidden, raw_visible, raw_hidden
    ):

        patterns = [
            (r"(\b\w+\b)(\s+\1){2,}", "Repeated keywords"),
            (
                r"\b(skills|expertise|proficient|proficiency|experience)s?\b.{0,30}?(?::|\-|\–).{0,200}?(\b\w+\b(,|\.|;).{0,10}?){10,}",
                "Excessive skill listing",
            ),
            (r"\b(keywords|tags|meta|seo)\b.{0,30}?(?::|\-|\–).{0,200}", "Meta tags"),
            (
                r"\b(job\s+title|position|role)\b.{0,30}?(?::|\-|\–).{0,100}",
                "Hidden job titles",
            ),
            (r"(\b\w+\b(,|\.|;)\s*){10,}", "Comma-separated keyword list"),
        ]

        detected_patterns = []

        for pattern, description in patterns:
            matches = re.finditer(pattern, raw_hidden, re.IGNORECASE)
            for match in matches:
                matched_text = match.group(0)
                if len(matched_text) > 10:
                    detected_patterns.append(
                        {
                            "pattern": description,
                            "text": matched_text[:100]
                            + ("..." if len(matched_text) > 100 else ""),
                        }
                    )

        is_suspicious = len(detected_patterns) > 0

        confidence = min(1.0, len(detected_patterns) / 3)

        return {
            "suspicious": is_suspicious,
            "confidence": confidence,
            "detected_patterns": detected_patterns,
            "message": f"Found {len(detected_patterns)} suspicious patterns in hidden text",
        }

    def _analyze_intent(
        self, processed_visible, processed_hidden, raw_visible, raw_hidden
    ):
        if not self.config["use_transformers"] or not self.classifier:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Transformer model not available for intent analysis",
            }

        if len(raw_hidden.split()) < 5:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Hidden text too short for intent analysis",
            }

        intent_categories = [
            "keyword stuffing",
            "relevant skills",
            "job qualifications",
            "personal information",
            "resume formatting",
            "legitimate content",
        ]

        try:

            result = self.classifier(raw_hidden[:512], intent_categories)

            top_intents = [
                {"label": result["labels"][i], "score": result["scores"][i]}
                for i in range(min(2, len(result["labels"])))
            ]

            suspicious_intents = ["keyword stuffing"]

            is_suspicious = top_intents[0]["label"] in suspicious_intents
            confidence = top_intents[0]["score"] if is_suspicious else 0.0

            return {
                "suspicious": is_suspicious,
                "confidence": confidence,
                "top_intents": top_intents,
                "message": f"Top intent detected: {top_intents[0]['label']} ({top_intents[0]['score']:.2f})",
            }

        except Exception as e:
            logger.error(f"Error in intent analysis: {str(e)}")
            return {
                "suspicious": False,
                "confidence": 0.0,
                "error": str(e),
                "message": "Failed to analyze intent",
            }

    def _create_tech_gazetteer(self):
        tech_terms = set()

        technical_skill_categories = [
            "programming_languages",
            "web_technologies",
            "databases",
            "cloud_platforms",
            "data_science",
            "devops",
            "mobile_development",
            "cybersecurity",
        ]

        for category in technical_skill_categories:
            if category in self.skill_terms:
                tech_terms.update([term.lower() for term in self.skill_terms[category]])

        if "technology" in self.industry_terms:
            tech_terms.update(
                [term.lower() for term in self.industry_terms["technology"]]
            )

        additional_tech_industries = [
            "telecommunications",
            "pharmaceutical",
            "healthcare",
        ]

        for industry in additional_tech_industries:
            if industry in self.industry_terms:
                tech_terms.update(
                    [term.lower() for term in self.industry_terms[industry]]
                )

        common_tech_abbreviations = [
            "ai",
            "ml",
            "nlp",
            "api",
            "aws",
            "gcp",
            "iot",
            "ar",
            "vr",
            "nosql",
            "sql",
            "cnn",
            "rnn",
            "lstm",
            "bert",
            "gpt",
            "devops",
            "cicd",
            "k-means",
            "hadoop",
            "spark",
            "tensorflow",
            "pytorch",
            "react",
            "node",
            "vue",
            "angular",
            "docker",
            "kubernetes",
            "linux",
            "windows",
            "ios",
            "android",
        ]

        tech_terms.update(common_tech_abbreviations)

        return tech_terms

    def _analyze_context_relevance(
        self, processed_visible, processed_hidden, raw_visible, raw_hidden
    ):
        if not self.nlp:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "NLP model not available for context relevance analysis",
            }

        if len(raw_hidden.split()) < 5:
            return {
                "suspicious": False,
                "confidence": 0.0,
                "message": "Hidden text too short for context relevance analysis",
            }

        try:
            if not isinstance(raw_visible, str):
                raw_visible = str(raw_visible)
            if not isinstance(raw_hidden, str):
                raw_hidden = str(raw_hidden)

            max_visible_len = min(len(raw_visible), 100000)
            max_hidden_len = min(len(raw_hidden), 100000)

            tech_gazetteer = self._create_tech_gazetteer()

            visible_doc = self.nlp(raw_visible[0:max_visible_len])

            visible_entities = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],
                "DATE": [],
                "PRODUCT": [],
            }

            for ent in visible_doc.ents:
                if ent.label_ in visible_entities:
                    visible_entities[ent.label_].append(ent.text.lower())

            hidden_doc = self.nlp(raw_hidden[0:max_hidden_len])

            hidden_entities = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],
                "DATE": [],
                "PRODUCT": [],
            }

            for ent in hidden_doc.ents:
                entity_lower = ent.text.lower()
                words = entity_lower.split()

                if entity_lower in tech_gazetteer or any(
                    word in tech_gazetteer for word in words
                ):
                    continue

                if ent.label_ in hidden_entities:
                    hidden_entities[ent.label_].append(entity_lower)

            mismatches = []

            for person in hidden_entities["PERSON"]:
                if not any(
                    self._fuzzy_match(person, visible_person)
                    for visible_person in visible_entities["PERSON"]
                ):
                    mismatches.append(
                        {
                            "type": "PERSON",
                            "entity": person,
                            "message": "Person mentioned in hidden text but not in visible text",
                        }
                    )

            for org in hidden_entities["ORG"]:
                if not any(
                    self._fuzzy_match(org, visible_org)
                    for visible_org in visible_entities["ORG"]
                ):
                    mismatches.append(
                        {
                            "type": "ORG",
                            "entity": org,
                            "message": "Organization mentioned in hidden text but not in visible text",
                        }
                    )

            for ent_type in ["GPE", "PRODUCT"]:
                for entity in hidden_entities[ent_type]:
                    if not any(
                        self._fuzzy_match(entity, visible_entity)
                        for visible_entity in visible_entities[ent_type]
                    ):
                        mismatches.append(
                            {
                                "type": ent_type,
                                "entity": entity,
                                "message": f"{ent_type} mentioned in hidden text but not in visible text",
                            }
                        )

            is_suspicious = len(mismatches) >= 3

            confidence = min(1.0, len(mismatches) / 5)

            return {
                "suspicious": is_suspicious,
                "confidence": confidence,
                "entity_mismatches": mismatches[:10],
                "mismatch_count": len(mismatches),
                "message": f"Found {len(mismatches)} entities in hidden text that don't match the resume context",
            }

        except Exception as e:
            logger.error(f"Error in context relevance analysis: {str(e)}")
            return {
                "suspicious": False,
                "confidence": 0.0,
                "error": str(e),
                "message": "Failed to analyze context relevance",
            }

    def _generate_summary(self, results):
        analysis = results["analysis"]
        suspicious_count = sum(
            1 for a in analysis.values() if a.get("suspicious", False)
        )

        if not results["has_suspicious_content"]:
            return "No significant semantic manipulation detected in hidden text."

        summary = f"Detected potential semantic manipulation with {results['confidence']:.2f} confidence. "

        details = []

        if "term_frequency" in analysis and analysis["term_frequency"].get(
            "suspicious", False
        ):
            term_count = analysis["term_frequency"].get("term_count", 0)
            details.append(
                f"Found {term_count} terms with unusually high frequency in hidden text"
            )

        if "industry_terms" in analysis and analysis["industry_terms"].get(
            "suspicious", False
        ):
            term_count = analysis["industry_terms"].get("term_count", 0)
            details.append(
                f"Found {term_count} industry/skill terms hidden from visible text"
            )

        if "patterns" in analysis and analysis["patterns"].get("suspicious", False):
            pattern_count = len(analysis["patterns"].get("detected_patterns", []))
            details.append(
                f"Detected {pattern_count} suspicious patterns in hidden text"
            )

        if "intent" in analysis and analysis["intent"].get("suspicious", False):
            top_intent = (
                analysis["intent"].get("top_intents", [{}])[0].get("label", "unknown")
            )
            details.append(f"Hidden text appears to be {top_intent}")

        if "context_relevance" in analysis and analysis["context_relevance"].get(
            "suspicious", False
        ):
            mismatch_count = analysis["context_relevance"].get("mismatch_count", 0)
            details.append(
                f"Found {mismatch_count} entities that don't match the resume context"
            )

        summary += " ".join(details)

        return summary

    def _get_embedding(self, text):

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = {key: val.to(self.config["device"]) for key, val in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding

    def _extract_matches(self, doc, matches):
        terms = set()
        for match_id, start, end in matches:
            span = doc[start:end].text.lower()
            terms.add(span)
        return terms

    def _fuzzy_match(self, text1, text2, threshold=0.8):

        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())

        if not tokens1 or not tokens2:
            return False

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        similarity = len(intersection) / len(union)
        return similarity >= threshold
