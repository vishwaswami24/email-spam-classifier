"""Unit tests for classifier.py — classify and train_model."""

import math

import pytest

from email_spam_classifier.classifier import classify, train_model
from email_spam_classifier.models import (
    ClassificationResult,
    FeatureVector,
    Label,
    TrainedModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_model(weights, bias=0.0):
    return TrainedModel(weights=list(weights), bias=bias, vocab_size=len(weights))


def make_fv(values):
    return FeatureVector(values=list(values), dim=len(values))


# ---------------------------------------------------------------------------
# classify — happy path
# ---------------------------------------------------------------------------

def test_classify_returns_result_and_none_on_success():
    model = make_model([1.0, 0.0], bias=0.0)
    fv = make_fv([10.0, 0.0])
    result, err = classify(model, fv)
    assert err is None
    assert isinstance(result, ClassificationResult)


def test_classify_spam_when_confidence_high():
    # Large positive z → confidence close to 1 → Spam
    model = make_model([1.0], bias=10.0)
    fv = make_fv([1.0])
    result, err = classify(model, fv)
    assert err is None
    assert result.label == Label.Spam
    assert result.confidence >= 0.5


def test_classify_ham_when_confidence_low():
    # Large negative z → confidence close to 0 → Ham
    model = make_model([1.0], bias=-10.0)
    fv = make_fv([1.0])
    result, err = classify(model, fv)
    assert err is None
    assert result.label == Label.Ham
    assert result.confidence < 0.5


def test_classify_boundary_exactly_half():
    # z = 0 → sigmoid(0) = 0.5 → Spam (>= 0.5)
    model = make_model([0.0], bias=0.0)
    fv = make_fv([0.0])
    result, err = classify(model, fv)
    assert err is None
    assert result.label == Label.Spam
    assert math.isclose(result.confidence, 0.5)


def test_classify_zero_vector_is_ham_near_half():
    # Empty token list → zero vector → z = bias = 0 → confidence ~0.5 → Spam
    # (bias=0 gives exactly 0.5 which is Spam; with negative bias it's Ham)
    model = make_model([0.0, 0.0], bias=-1.0)
    fv = make_fv([0.0, 0.0])
    result, err = classify(model, fv)
    assert err is None
    assert result.label == Label.Ham
    assert result.confidence < 0.5


def test_classify_confidence_in_range():
    model = make_model([0.5, -0.3], bias=0.1)
    fv = make_fv([2.0, 1.0])
    result, err = classify(model, fv)
    assert err is None
    assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# classify — dimension mismatch
# ---------------------------------------------------------------------------

def test_classify_dimension_mismatch_returns_error():
    model = make_model([1.0, 2.0], bias=0.0)  # vocab_size=2
    fv = make_fv([1.0])                        # dim=1
    result, err = classify(model, fv)
    assert result is None
    assert err is not None
    assert "mismatch" in err.lower() or "2" in err


# ---------------------------------------------------------------------------
# train_model — happy path
# ---------------------------------------------------------------------------

def test_train_model_returns_model_and_none_on_success():
    features = [make_fv([1.0, 0.0]), make_fv([0.0, 1.0])]
    labels = [Label.Spam, Label.Ham]
    model, err = train_model(features, labels)
    assert err is None
    assert isinstance(model, TrainedModel)


def test_train_model_weights_length_matches_vocab_size():
    features = [make_fv([1.0, 2.0, 3.0])] * 3
    labels = [Label.Spam, Label.Ham, Label.Spam]
    model, err = train_model(features, labels)
    assert err is None
    assert len(model.weights) == 3
    assert model.vocab_size == 3


def test_train_model_bias_is_finite():
    features = [make_fv([1.0]), make_fv([0.0])]
    labels = [Label.Spam, Label.Ham]
    model, err = train_model(features, labels)
    assert err is None
    assert math.isfinite(model.bias)


def test_train_model_learns_to_separate():
    # Clearly separable: spam has high feature, ham has low feature
    features = [make_fv([5.0])] * 5 + [make_fv([0.0])] * 5
    labels = [Label.Spam] * 5 + [Label.Ham] * 5
    model, err = train_model(features, labels, learning_rate=0.1, epochs=200)
    assert err is None
    spam_result, _ = classify(model, make_fv([5.0]))
    ham_result, _ = classify(model, make_fv([0.0]))
    assert spam_result.label == Label.Spam
    assert ham_result.label == Label.Ham


def test_train_model_vocabulary_is_empty_dict():
    features = [make_fv([1.0])]
    labels = [Label.Ham]
    model, err = train_model(features, labels)
    assert err is None
    assert model.vocabulary == {}


# ---------------------------------------------------------------------------
# train_model — error cases
# ---------------------------------------------------------------------------

def test_train_model_empty_features_returns_error():
    result, err = train_model([], [])
    assert result is None
    assert err is not None


def test_train_model_mismatched_lengths_returns_error():
    features = [make_fv([1.0]), make_fv([2.0])]
    labels = [Label.Spam]
    result, err = train_model(features, labels)
    assert result is None
    assert err is not None


def test_train_model_more_labels_than_features_returns_error():
    features = [make_fv([1.0])]
    labels = [Label.Spam, Label.Ham]
    result, err = train_model(features, labels)
    assert result is None
    assert err is not None
