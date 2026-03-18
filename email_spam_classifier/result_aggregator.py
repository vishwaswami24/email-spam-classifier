"""Result aggregation — attach metadata and format classification output."""

from datetime import datetime

from email_spam_classifier.models import ClassificationOutput, ClassificationResult


def build_output(
    email_id: str,
    result: ClassificationResult,
    low_confidence_threshold: float = 0.3,
) -> ClassificationOutput:
    """Attach email ID, ISO 8601 timestamp, and review flag to a ClassificationResult.

    Sets needs_review=True when confidence is below low_confidence_threshold.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"
    needs_review = result.confidence < low_confidence_threshold
    return ClassificationOutput(
        email_id=email_id,
        label=result.label,
        confidence=result.confidence,
        timestamp=timestamp,
        needs_review=needs_review,
    )
