import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Download required NLTK resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

logger = logging.getLogger(__name__)


class TextPreprocessor:
    def __init__(self, custom_stopwords=None):
        self.stop_words = set(stopwords.words("english"))
        if custom_stopwords:
            self.stop_words.update(custom_stopwords)

        self.lemmatizer = WordNetLemmatizer()
        logger.info("Initializing TextPreprocessor")

    def preprocess(self, text):
        text = text.lower()

        text = re.sub(r"\S*@\S*\s?", "", text)

        text = re.sub(r"http\S+", "", text)

        text = re.sub(
            r"\b(?:\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "", text
        )

        text = text.translate(str.maketrans("", "", string.punctuation))

        text = re.sub(r"\d+", "", text)

        tokens = word_tokenize(text)

        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words
        ]

        preprocessed_text = " ".join(tokens)

        return preprocessed_text

    def extract_features(self, text):
        tokens = text.split()
        word_count = len(tokens)

        avg_word_length = (
            sum(len(token) for token in tokens) / word_count if word_count > 0 else 0
        )

        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]

        return {
            "text": text,
            "word_count": word_count,
            "avg_word_length": avg_word_length,
            "bigrams": bigrams,
        }
