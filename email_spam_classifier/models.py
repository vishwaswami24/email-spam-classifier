"""Shared data models for the Email Spam Classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Label(Enum):
    """Binary classification label."""
    Spam = "spam"
    Ham = "ham"


@dataclass
class RawEmail:
    """Unstructured input email."""
    sender: str
    subject: str
    body: str
    headers: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class EmailContent:
    """Structured representation of a parsed email."""
    subject: str
    body: str
    sender: str


@dataclass
class CleanedEmail:
    """Preprocessed email with normalized tokens."""
    tokens: List[str]


@dataclass
class FeatureVector:
    """Numerical TF-IDF representation of an email."""
    values: List[float]
    dim: int


@dataclass
class TrainedModel:
    """Serializable logistic regression model artifact."""
    weights: List[float]
    bias: float
    vocab_size: int
    vocabulary: Dict[str, int] = field(default_factory=dict)


@dataclass
class ClassificationResult:
    """Output of the classifier for a single email."""
    label: Label
    confidence: float


@dataclass
class ClassificationOutput:
    """Final output with metadata attached by the result aggregator."""
    email_id: str
    label: Label
    confidence: float
    timestamp: str
    needs_review: bool = False


@dataclass
class LabeledEmail:
    """An email paired with a ground-truth label, used for training."""
    content: EmailContent
    label: Label


@dataclass
class ModelMetadata:
    """Metadata produced after a training run."""
    vocab_size: int
    num_training_samples: int
    model_path: str
