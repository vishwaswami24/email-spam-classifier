"""Logistic regression classifier and model trainer."""

import math
from typing import List, Optional, Tuple

import numpy as np

from email_spam_classifier.models import (
    ClassificationResult,
    FeatureVector,
    Label,
    TrainedModel,
)


def classify(
    model: TrainedModel, features: FeatureVector
) -> Tuple[Optional[ClassificationResult], Optional[str]]:
    """Apply logistic regression to features and return a ClassificationResult.

    Returns (ClassificationResult, None) on success, or (None, error_string) on failure.
    """
    if features.dim != model.vocab_size:
        return (
            None,
            f"Dimension mismatch: expected {model.vocab_size}, got {features.dim}",
        )

    weights = np.array(model.weights)
    x = np.array(features.values)
    z = model.bias + float(np.dot(weights, x))
    confidence = 1.0 / (1.0 + math.exp(-z))
    label = Label.Spam if confidence >= 0.5 else Label.Ham
    return (ClassificationResult(label=label, confidence=confidence), None)


def train_model(
    features: List[FeatureVector],
    labels: List[Label],
    learning_rate: float = 0.1,
    epochs: int = 100,
) -> Tuple[Optional[TrainedModel], Optional[str]]:
    """Train a logistic regression model via gradient descent.

    Returns (TrainedModel, None) on success, or (None, error_string) on failure.
    """
    if len(features) != len(labels):
        return (
            None,
            f"Length mismatch: {len(features)} feature vectors but {len(labels)} labels",
        )
    if len(features) == 0:
        return (None, "Cannot train on an empty feature list")

    vocab_size = features[0].dim
    weights = np.zeros(vocab_size, dtype=float)
    bias = 0.0

    y_numeric = np.array(
        [1.0 if lbl == Label.Spam else 0.0 for lbl in labels], dtype=float
    )
    X = np.array([fv.values for fv in features], dtype=float)

    for _ in range(epochs):
        for i in range(len(features)):
            z = bias + float(np.dot(weights, X[i]))
            pred = 1.0 / (1.0 + math.exp(-z))
            delta = pred - y_numeric[i]
            weights -= learning_rate * delta * X[i]
            bias -= learning_rate * delta

    return (
        TrainedModel(
            weights=weights.tolist(),
            bias=float(bias),
            vocab_size=vocab_size,
            vocabulary={},
        ),
        None,
    )
