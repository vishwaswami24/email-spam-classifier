"""Text preprocessing pipeline."""

import re
import string
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from email_spam_classifier.models import CleanedEmail, EmailContent

_stemmer = PorterStemmer()
_stop_words = set(stopwords.words('english'))


def normalize(text: str) -> str:
    """Lowercase, strip HTML tags and punctuation."""
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return text


def tokenize(text: str) -> List[str]:
    """Split text into word tokens."""
    return [token for token in text.split() if token]


def remove_stop_words(tokens: List[str]) -> List[str]:
    """Remove standard English stop words from token list."""
    return [token for token in tokens if token not in _stop_words]


def stem(token: str) -> str:
    """Apply Porter Stemmer to a single token."""
    return _stemmer.stem(token)


def preprocess_email(content: EmailContent) -> CleanedEmail:
    """Normalize, tokenize, remove stop words, and stem email text."""
    combined = content.subject + ' ' + content.body
    normalized = normalize(combined)
    tokens = tokenize(normalized)
    tokens = remove_stop_words(tokens)
    stemmed = [stem(token) for token in tokens]
    final_tokens = [t for t in stemmed if t]
    return CleanedEmail(tokens=final_tokens)
