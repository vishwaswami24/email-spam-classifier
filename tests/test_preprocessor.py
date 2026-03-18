"""Unit tests for the email preprocessor module."""

import pytest
from email_spam_classifier.models import EmailContent
from email_spam_classifier.preprocessor import preprocess_email


def make_content(subject="", body="Hello world"):
    return EmailContent(subject=subject, body=body, sender="user@example.com")


class TestPreprocessEmailLowercase:
    def test_tokens_are_lowercase(self):
        content = make_content(body="HELLO WORLD Spam")
        result = preprocess_email(content)
        for token in result.tokens:
            assert token == token.lower(), f"Token '{token}' is not lowercase"

    def test_mixed_case_subject_lowercased(self):
        content = make_content(subject="FREE MONEY", body="click here")
        result = preprocess_email(content)
        for token in result.tokens:
            assert token == token.lower()


class TestPreprocessEmailStopWordRemoval:
    def test_common_stop_words_removed(self):
        content = make_content(body="this is a test of the system")
        result = preprocess_email(content)
        stop_words = {"this", "is", "a", "of", "the"}
        for token in result.tokens:
            assert token not in stop_words, f"Stop word '{token}' was not removed"

    def test_non_stop_words_retained(self):
        content = make_content(body="buy cheap medication now")
        result = preprocess_email(content)
        # After stemming, meaningful words should still produce tokens
        assert len(result.tokens) > 0


class TestPreprocessEmailNoMutation:
    def test_original_content_not_mutated(self):
        original_subject = "Hello World"
        original_body = "This is a test email body."
        content = make_content(subject=original_subject, body=original_body)
        preprocess_email(content)
        assert content.subject == original_subject
        assert content.body == original_body
        assert content.sender == "user@example.com"

    def test_returns_new_cleaned_email_object(self):
        content = make_content(body="spam free offer")
        result = preprocess_email(content)
        assert result is not content
        assert hasattr(result, "tokens")
        assert isinstance(result.tokens, list)
