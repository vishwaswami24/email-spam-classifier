"""Unit tests for the feature extractor module."""

import math

import pytest

from email_spam_classifier.models import CleanedEmail, FeatureVector
from email_spam_classifier.feature_extractor import (
    build_vocabulary,
    build_doc_freq,
    compute_tfidf,
    extract_features,
    MAX_VOCAB_SIZE,
)


def make_email(*tokens):
    return CleanedEmail(tokens=list(tokens))


# ---------------------------------------------------------------------------
# build_vocabulary
# ---------------------------------------------------------------------------

class TestBuildVocabulary:
    def test_returns_dict(self):
        corpus = [make_email("hello", "world")]
        vocab = build_vocabulary(corpus)
        assert isinstance(vocab, dict)

    def test_all_tokens_present(self):
        corpus = [make_email("spam", "free", "offer")]
        vocab = build_vocabulary(corpus)
        assert "spam" in vocab
        assert "free" in vocab
        assert "offer" in vocab

    def test_indices_are_unique(self):
        corpus = [make_email("a", "b", "c", "d")]
        vocab = build_vocabulary(corpus)
        indices = list(vocab.values())
        assert len(indices) == len(set(indices))

    def test_indices_are_zero_based(self):
        corpus = [make_email("hello", "world")]
        vocab = build_vocabulary(corpus)
        assert min(vocab.values()) == 0

    def test_capped_at_max_vocab_size(self):
        # Create a corpus with more than MAX_VOCAB_SIZE unique tokens
        tokens = [f"word{i}" for i in range(MAX_VOCAB_SIZE + 500)]
        corpus = [make_email(*tokens)]
        vocab = build_vocabulary(corpus)
        assert len(vocab) <= MAX_VOCAB_SIZE

    def test_most_frequent_tokens_kept(self):
        # "spam" appears 5 times, "rare" appears 1 time
        corpus = [make_email("spam", "spam", "spam", "spam", "spam", "rare")]
        vocab = build_vocabulary(corpus)
        assert "spam" in vocab
        assert "rare" in vocab

    def test_empty_corpus_returns_empty_vocab(self):
        vocab = build_vocabulary([])
        assert vocab == {}

    def test_empty_email_tokens(self):
        corpus = [make_email()]
        vocab = build_vocabulary(corpus)
        assert vocab == {}

    def test_frequency_ordering(self):
        # "a" appears 3x, "b" 2x, "c" 1x — "a" should get index 0
        corpus = [make_email("a", "a", "a", "b", "b", "c")]
        vocab = build_vocabulary(corpus)
        assert vocab["a"] == 0
        assert vocab["b"] == 1
        assert vocab["c"] == 2


# ---------------------------------------------------------------------------
# build_doc_freq
# ---------------------------------------------------------------------------

class TestBuildDocFreq:
    def test_counts_documents_not_occurrences(self):
        # "spam" appears twice in one doc — should count as 1
        corpus = [make_email("spam", "spam"), make_email("spam")]
        df = build_doc_freq(corpus)
        assert df["spam"] == 2

    def test_token_in_single_doc(self):
        corpus = [make_email("unique"), make_email("other")]
        df = build_doc_freq(corpus)
        assert df["unique"] == 1

    def test_empty_corpus(self):
        assert build_doc_freq([]) == {}


# ---------------------------------------------------------------------------
# compute_tfidf
# ---------------------------------------------------------------------------

class TestComputeTfidf:
    def test_returns_feature_vector(self):
        vocab = {"hello": 0, "world": 1}
        fv = compute_tfidf(["hello", "world"], vocab, corpus_size=1, doc_freq={"hello": 1, "world": 1})
        assert isinstance(fv, FeatureVector)

    def test_dimension_matches_vocab(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        fv = compute_tfidf(["a", "b"], vocab, corpus_size=2, doc_freq={"a": 1, "b": 1})
        assert fv.dim == 3
        assert len(fv.values) == 3

    def test_oov_tokens_contribute_zero(self):
        vocab = {"hello": 0}
        fv = compute_tfidf(["hello", "unknown"], vocab, corpus_size=1, doc_freq={"hello": 1})
        # "unknown" is not in vocab — only index 0 should be non-zero
        assert fv.values[0] > 0.0

    def test_empty_tokens_returns_zero_vector(self):
        vocab = {"hello": 0, "world": 1}
        fv = compute_tfidf([], vocab, corpus_size=1, doc_freq={})
        assert fv.values == [0.0, 0.0]
        assert fv.dim == 2

    def test_tfidf_formula(self):
        # Single token "spam" in a corpus of 2 docs, appearing in 1 doc
        vocab = {"spam": 0}
        tokens = ["spam", "spam"]  # count=2, total=2 → TF=1.0
        corpus_size = 2
        doc_freq = {"spam": 1}
        fv = compute_tfidf(tokens, vocab, corpus_size, doc_freq)
        expected_tf = 2 / 2  # 1.0
        expected_idf = math.log((2 + 1) / (1 + 1)) + 1
        expected = expected_tf * expected_idf
        assert abs(fv.values[0] - expected) < 1e-9

    def test_all_values_finite(self):
        vocab = {"a": 0, "b": 1}
        fv = compute_tfidf(["a", "b", "a"], vocab, corpus_size=5, doc_freq={"a": 3, "b": 2})
        for v in fv.values:
            assert math.isfinite(v), f"Non-finite value: {v}"

    def test_values_non_negative(self):
        vocab = {"hello": 0, "world": 1}
        fv = compute_tfidf(["hello"], vocab, corpus_size=3, doc_freq={"hello": 2, "world": 1})
        for v in fv.values:
            assert v >= 0.0


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_feature_vector(self):
        vocab = {"hello": 0, "world": 1}
        email = make_email("hello", "world")
        fv = extract_features(email, vocab)
        assert isinstance(fv, FeatureVector)

    def test_dimension_matches_vocab(self):
        vocab = {"a": 0, "b": 1, "c": 2}
        email = make_email("a")
        fv = extract_features(email, vocab)
        assert fv.dim == 3
        assert len(fv.values) == 3

    def test_oov_tokens_are_zero(self):
        vocab = {"known": 0}
        email = make_email("known", "unknown_word")
        fv = extract_features(email, vocab)
        assert fv.dim == 1
        assert len(fv.values) == 1
        # "unknown_word" not in vocab, so only index 0 matters
        assert fv.values[0] > 0.0

    def test_empty_tokens_zero_vector(self):
        vocab = {"hello": 0, "world": 1}
        email = make_email()
        fv = extract_features(email, vocab)
        assert fv.values == [0.0, 0.0]

    def test_all_values_finite(self):
        vocab = {"spam": 0, "free": 1, "offer": 2}
        email = make_email("spam", "free", "spam")
        fv = extract_features(email, vocab)
        for v in fv.values:
            assert math.isfinite(v)

    def test_empty_vocab_returns_empty_vector(self):
        vocab = {}
        email = make_email("hello")
        fv = extract_features(email, vocab)
        assert fv.dim == 0
        assert fv.values == []
