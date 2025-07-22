import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

# Download NLTK resources (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')


class FeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.category_keywords = self._load_category_keywords()
        logger.info("Initializing FeatureExtractor")

    def _load_category_keywords(self):
        """Load keywords for each job category."""
        return {
            "Accountant": [
                "accounting",
                "bookkeeping",
                "financial statements",
                "auditing",
                "reconciliation",
                "cpa",
                "tax",
                "budgeting",
                "forecasting",
                "gaap",
            ],
            "Advocate": [
                "law",
                "legal",
                "justice",
                "litigation",
                "attorney",
                "counsel",
                "jurisprudence",
                "pleading",
                "court",
                "verdict",
            ],
            "Agriculture": [
                "farming",
                "crops",
                "livestock",
                "agronomy",
                "soil",
                "irrigation",
                "tractor",
                "fertilizer",
                "harvest",
                "sustainability",
            ],
            "Apparel": [
                "fashion",
                "textiles",
                "clothing",
                "design",
                "production",
                "retail",
                "merchandising",
                "sourcing",
                "garment",
                "style",
            ],
            "Arts": [
                "artist",
                "creative",
                "gallery",
                "exhibition",
                "studio",
                "sculpture",
                "painting",
                "drawing",
                "performance",
                "curator",
            ],
            "Automobile": [
                "automotive",
                "vehicle",
                "mechanic",
                "engine",
                "service",
                "repair",
                "maintenance",
                "parts",
                "dealership",
                "manufacturing",
            ],
            "Aviation": [
                "pilot",
                "aerospace",
                "flight",
                "aircraft",
                "air traffic control",
                "faa",
                "maintenance",
                "avionics",
                "airline",
                "airport",
            ],
            "Banking": [
                "finance",
                "loan",
                "mortgage",
                "teller",
                "investment",
                "credit",
                "risk management",
                "financial services",
                "wealth management",
                "transactions",
            ],
            "BPO": [
                "business process outsourcing",
                "customer service",
                "call center",
                "inbound",
                "outbound",
                "telemarketing",
                "client relations",
                "technical support",
                "data entry",
                "quality assurance",
            ],
            "Business-development": [
                "business development",
                "sales",
                "strategy",
                "partnership",
                "market research",
                "lead generation",
                "negotiation",
                "revenue growth",
                "client acquisition",
                "networking",
            ],
            "Chef": [
                "cooking",
                "culinary",
                "food",
                "restaurant",
                "kitchen",
                "menu",
                "catering",
                "baking",
                "recipe",
                "cuisine",
                "hospitality",
                "gastronomy",
            ],
            "Construction": [
                "construction",
                "project management",
                "site supervision",
                "blueprint",
                "contractor",
                "engineering",
                "building",
                "safety",
                "masonry",
                "estimating",
            ],
            "Consultant": [
                "consulting",
                "advisory",
                "strategy",
                "analysis",
                "implementation",
                "client solutions",
                "business process",
                "problem-solving",
                "change management",
                "stakeholder",
            ],
            "Designer": [
                "design",
                "graphic design",
                "ui/ux",
                "creative suite",
                "adobe",
                "typography",
                "branding",
                "visual identity",
                "layout",
                "prototyping",
            ],
            "Digital-media": [
                "digital marketing",
                "social media",
                "seo",
                "content creation",
                "analytics",
                "campaign",
                "web content",
                "video production",
                "email marketing",
                "online advertising",
            ],
            "Engineering": [
                "engineer",
                "design",
                "cad",
                "electrical",
                "mechanical",
                "civil",
                "software",
                "chemical",
                "analysis",
                "prototype",
            ],
            "Finance": [
                "financial analysis",
                "investment",
                "portfolio",
                "budgeting",
                "corporate finance",
                "financial modeling",
                "mergers and acquisitions",
                "equities",
                "trading",
                "risk assessment",
            ],
            "Fitness": [
                "personal trainer",
                "fitness",
                "wellness",
                "exercise",
                "nutrition",
                "coaching",
                "gym",
                "group fitness",
                "strength training",
                "sports science",
            ],
            "Healthcare": [
                "patient care",
                "medical",
                "clinical",
                "physician",
                "nurse",
                "doctor",
                "healthcare",
                "hospital",
                "diagnosis",
                "treatment",
                "therapy",
                "pharmaceutical",
            ],
            "HR": [
                "recruitment",
                "hiring",
                "onboarding",
                "talent acquisition",
                "employee relations",
                "performance management",
                "benefits",
                "compensation",
                "hr management",
                "hris",
                "organizational development",
            ],
            "Information-technology": [
                "programming",
                "software",
                "development",
                "java",
                "python",
                "c++",
                "javascript",
                "web development",
                "database",
                "sql",
                "cloud",
                "aws",
                "azure",
                "devops",
                "agile",
                "scrum",
                "cybersecurity",
                "networking",
            ],
            "Public-relations": [
                "public relations",
                "media relations",
                "press release",
                "crisis communication",
                "brand reputation",
                "storytelling",
                "corporate communications",
                "event planning",
                "social media strategy",
                "publicity",
            ],
            "Sales": [
                "sales",
                "client relations",
                "business development",
                "account management",
                "lead generation",
                "revenue",
                "quota",
                "crm",
                "customer acquisition",
                "negotiation",
            ],
            "Teacher": [
                "education",
                "teaching",
                "curriculum",
                "instruction",
                "classroom",
                "school",
                "student",
                "learning",
                "assessment",
                "pedagogy",
                "lesson plan",
            ],
        }

    def extract_features(self, text):
        features = {}

        features["word_count"] = len(text.split())
        features["char_count"] = len(text)

        features["skills"] = self.extract_skills(text)

        features["education"] = self.extract_education(text)

        features["experience"] = self.extract_experience(text)

        return features

    def extract_skills(self, text):

        common_skills = [
            "python",
            "java",
            "c++",
            "javascript",
            "html",
            "css",
            "sql",
            "management",
            "leadership",
            "communication",
            "teamwork",
            "project management",
            "research",
            "analysis",
            "problem solving",
            "customer service",
            "sales",
            "marketing",
            "accounting",
            "microsoft office",
            "excel",
            "word",
            "powerpoint",
            "photoshop",
        ]

        found_skills = []
        for skill in common_skills:
            if re.search(r"\b" + re.escape(skill) + r"\b", text.lower()):
                found_skills.append(skill)

        return found_skills

    def extract_education(self, text):

        education = []

        degree_patterns = [
            r"\b(Bachelor|Master|PhD|Doctorate|BSc|MSc|BA|MA|MD|JD|MBA)\b",
            r"\b(Bachelor\'s|Master\'s|Doctoral|Graduate|Undergraduate)\b",
            r"\b(B\.\s?S\.|M\.\s?S\.|B\.\s?A\.|M\.\s?A\.|Ph\.\s?D\.)\b",
        ]

        for pattern in degree_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                education.append(context)

        return education

    def extract_experience(self, text):
        experience_headers = [
            "experience",
            "work experience",
            "employment",
            "work history",
            "professional experience",
            "career history",
        ]

        experience_sections = []

        for header in experience_headers:
            pattern = re.compile(r"\b" + re.escape(header) + r"\b", re.IGNORECASE)
            matches = pattern.finditer(text)

            for match in matches:
                # Extract the section following the header
                start = match.start()
                # Assume the section ends at the next major header or end of text
                next_section = text.find("\n\n", start + len(header))
                if next_section == -1:
                    next_section = len(text)

                section = text[start:next_section].strip()
                if section:
                    experience_sections.append(section)

        return experience_sections

    def extract_employment_durations(self, text):
        date_ranges = re.findall(
            r"(\b\d{4}\s*[-–—]\s*\d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–—]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\s*[-–—]\s*(?:Present|Current|Now)\b)",
            text,
        )

        # Process each date range to calculate duration
        durations = []
        for date_range in date_ranges:
            # TODO: Parse the date range and calculate duration in months
            # This would require more sophisticated date parsing
            durations.append(
                {
                    "range": date_range,
                    "parsed_duration": None,  # Placeholder for actual duration
                }
            )

        return durations
