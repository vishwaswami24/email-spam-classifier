"""Email ingestion and validation layer."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from email_spam_classifier.models import EmailContent, RawEmail

# RFC 5322-inspired email regex (practical subset)
_EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


def ingest_email(raw: RawEmail) -> Tuple[Optional[EmailContent], Optional[str]]:
    """Parse and validate a RawEmail.

    Returns:
        (EmailContent, None) on success.
        (None, error_string) on validation failure.
    """
    if not raw.body or not raw.body.strip():
        return None, "Invalid email: body is missing or empty"

    if not raw.sender or not _EMAIL_REGEX.match(raw.sender.strip()):
        return None, f"Invalid email: sender '{raw.sender}' is not a valid email address"

    content = EmailContent(
        subject=raw.subject,
        body=raw.body,
        sender=raw.sender.strip(),
    )
    return content, None
