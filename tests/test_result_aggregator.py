"""Tests for result_aggregator.py — build_output function."""

import re

import pytest

from email_spam_classifier.models import ClassificationResult, Label
from email_spam_classifier.result_aggregator import build_output


ISO_8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$"
)


# ---------------------------------------------------------------------------
# Required output fields
# ---------------------------------------------------------------------------

def test_build_output_has_email_id():
    result = ClassificationResult(label=Label.Spam, confidence=0.9)
    output = build_output("msg-001", result)
    assert output.email_id == "msg-001"


def test_build_output_has_label():
    result = ClassificationResult(label=Label.Ham, confidence=0.8)
    output = build_output("msg-002", result)
    assert output.label == Label.Ham


def test_build_output_has_confidence():
    result = ClassificationResult(label=Label.Spam, confidence=0.75)
    output = build_output("msg-003", result)
    assert output.confidence == 0.75


def test_build_output_has_timestamp():
    result = ClassificationResult(label=Label.Spam, confidence=0.9)
    output = build_output("msg-004", result)
    assert output.timestamp is not None
    assert isinstance(output.timestamp, str)
    assert len(output.timestamp) > 0


def test_build_output_has_needs_review():
    result = ClassificationResult(label=Label.Spam, confidence=0.9)
    output = build_output("msg-005", result)
    assert hasattr(output, "needs_review")


# ---------------------------------------------------------------------------
# Timestamp is ISO 8601
# ---------------------------------------------------------------------------

def test_timestamp_is_iso8601_format():
    result = ClassificationResult(label=Label.Spam, confidence=0.9)
    output = build_output("msg-006", result)
    assert ISO_8601_RE.match(output.timestamp), f"Timestamp not ISO 8601: {output.timestamp!r}"


def test_timestamp_ends_with_z():
    result = ClassificationResult(label=Label.Spam, confidence=0.5)
    output = build_output("msg-007", result)
    assert output.timestamp.endswith("Z")


# ---------------------------------------------------------------------------
# needs_review flag based on confidence threshold
# ---------------------------------------------------------------------------

def test_needs_review_true_when_confidence_below_threshold():
    result = ClassificationResult(label=Label.Spam, confidence=0.2)
    output = build_output("msg-008", result, low_confidence_threshold=0.3)
    assert output.needs_review is True


def test_needs_review_false_when_confidence_above_threshold():
    result = ClassificationResult(label=Label.Spam, confidence=0.8)
    output = build_output("msg-009", result, low_confidence_threshold=0.3)
    assert output.needs_review is False


def test_needs_review_false_when_confidence_equals_threshold():
    result = ClassificationResult(label=Label.Ham, confidence=0.3)
    output = build_output("msg-010", result, low_confidence_threshold=0.3)
    assert output.needs_review is False


def test_needs_review_uses_default_threshold():
    # Default threshold is 0.3; confidence 0.1 should trigger review
    result = ClassificationResult(label=Label.Ham, confidence=0.1)
    output = build_output("msg-011", result)
    assert output.needs_review is True


def test_needs_review_false_with_high_confidence_default_threshold():
    result = ClassificationResult(label=Label.Spam, confidence=0.95)
    output = build_output("msg-012", result)
    assert output.needs_review is False


def test_needs_review_custom_threshold():
    result = ClassificationResult(label=Label.Ham, confidence=0.5)
    output = build_output("msg-013", result, low_confidence_threshold=0.6)
    assert output.needs_review is True
