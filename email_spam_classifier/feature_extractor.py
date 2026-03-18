"""TF-IDF feature extraction and vocabulary building."""

import math
from collections import Counter
from typing import Dict, List

import numpy as np

from email_spam_classifier.models import CleanedEmail, FeatureVector

Vocabulary = Dict[str, int]

MAX_VOCAB_SIZE = 10_000


def build_vocabulary(corpus: List[CleanedEmail]) -> Vocabulary:
    """Build a vocabulary capped at 10,000 most frequent tokens.

    Returns a plain dict mapping word -> 0-based integer index,
    ordered by descending frequency.
    """
    freq: Counter = Counter()
    for email in corpus:
        freq.update(email.tokens)

    most_common = freq.most_common(MAX_VOCAB_SIZE)
    return {word: idx for idx, (word, _) in enumerate(most_common)}


def build_doc_freq(corpus: List[CleanedEmail]) -> Dict[str, int]:
    """Count how many documents each token appears in (not total occurrences)."""
    doc_freq: Counter = Counter()
    for email in corpus:
        doc_freq.update(set(email.tokens))
    return dict(doc_freq)


def compute_tfidf(
    tokens: List[str],
    vocab: Vocabulary,
    corpus_size: int,
    doc_freq: Dict[str, int],
) -> FeatureVector:
    """Compute TF-IDF vector.

    TF(w)  = count(w) / total_tokens
    IDF(w) = log((N+1) / (df(w)+1)) + 1
    OOV tokens contribute 0.0.
    """
    dim = len(vocab)
    values = np.zeros(dim, dtype=np.float64)

    if not tokens:
        return FeatureVector(values=values.tolist(), dim=dim)

    total_tokens = len(tokens)
    token_counts: Counter = Counter(tokens)

    for word, count in token_counts.items():
        if word not in vocab:
            continue
        idx = vocab[word]
        tf = count / total_tokens
        df = doc_freq.get(word, 0)
        idf = math.log((corpus_size + 1) / (df + 1)) + 1
        values[idx] = tf * idf

    return FeatureVector(values=values.tolist(), dim=dim)


def extract_features(email: CleanedEmail, vocab: Vocabulary) -> FeatureVector:
    """Compute a TF-IDF FeatureVector for a single CleanedEmail.

    Uses the email itself as a 1-document corpus for IDF computation.
    For training, prefer calling compute_tfidf directly with full corpus context.
    """
    doc_freq = {word: 1 for word in set(email.tokens)}
    return compute_tfidf(
        tokens=email.tokens,
        vocab=vocab,
        corpus_size=1,
        doc_freq=doc_freq,
    )
