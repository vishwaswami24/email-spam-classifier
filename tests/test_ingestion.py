"""Unit tests for the email ingestion module."""

import pytest
from email_spam_classifier.models import RawEmail
from email_spam_classifier.ingestion import ingest_email


def make_raw(sender="user@example.com", subject="Hello", body="This is the email body."):
    return RawEmail(sender=sender, subject=subject, body=body)


class TestIngestEmailValid:
    def test_valid_input_returns_content(self):
        raw = make_raw()
        content, error = ingest_email(raw)
        assert error is None
        assert content is not None

    def test_valid_input_preserves_fields(self):
        raw = make_raw(sender="alice@domain.org", subject="Test", body="Hello world")
        content, error = ingest_email(raw)
        assert content.sender == "alice@domain.org"
        assert content.subject == "Test"
        assert content.body == "Hello world"

    def test_sender_is_stripped(self):
        raw = make_raw(sender="  user@example.com  ")
        content, error = ingest_email(raw)
        assert error is None
        assert content.sender == "user@example.com"


class TestIngestEmailEmptyBody:
    def test_empty_body_returns_error(self):
        raw = make_raw(body="")
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None
        assert "body" in error.lower()

    def test_whitespace_only_body_returns_error(self):
        raw = make_raw(body="   \n\t  ")
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None

    def test_none_body_returns_error(self):
        raw = RawEmail(sender="user@example.com", subject="Hi", body=None)
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None


class TestIngestEmailInvalidSender:
    def test_missing_at_sign_returns_error(self):
        raw = make_raw(sender="notanemail")
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None
        assert "sender" in error.lower()

    def test_missing_domain_returns_error(self):
        raw = make_raw(sender="user@")
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None

    def test_empty_sender_returns_error(self):
        raw = make_raw(sender="")
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None

    def test_none_sender_returns_error(self):
        raw = RawEmail(sender=None, subject="Hi", body="Body text")
        content, error = ingest_email(raw)
        assert content is None
        assert error is not None
