"""Unit tests for pipeline.py — classify_email, classify_batch, train_pipeline."""

import os
import tempfile

import pandas as pd
import pytest

from email_spam_classifier.models import (
    ClassificationOutput,
    Label,
    ModelMetadata,
    RawEmail,
)
from email_spam_classifier.pipeline import classify_batch, classify_email, train_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPAM_EMAIL = RawEmail(
    sender="offers@deals.example.com",
    subject="You won a prize!!!",
    body="Click here to claim your free gift. Limited time offer! Buy now win money.",
)

HAM_EMAIL = RawEmail(
    sender="alice@example.com",
    subject="Meeting tomorrow",
    body="Hi, just a reminder about our meeting tomorrow at 10am. See you then.",
)

INVALID_EMAIL = RawEmail(
    sender="not-an-email",
    subject="Test",
    body="Some body text",
)

EMPTY_BODY_EMAIL = RawEmail(
    sender="user@example.com",
    subject="Test",
    body="",
)


def make_dataset_csv(tmp_dir: str) -> str:
    """Create a small labeled CSV dataset for training tests."""
    data = {
        "sender": [
            "spam@deals.com",
            "spam@offers.com",
            "spam@promo.com",
            "spam@win.com",
            "spam@free.com",
            "alice@work.com",
            "bob@work.com",
            "carol@work.com",
            "dave@work.com",
            "eve@work.com",
        ],
        "subject": [
            "Win money now",
            "Free prize offer",
            "Click here buy",
            "Limited offer win",
            "Claim your prize",
            "Meeting tomorrow",
            "Project update",
            "Lunch plans",
            "Code review",
            "Weekly sync",
        ],
        "body": [
            "Click here to win money free prize offer buy now",
            "You have won a free prize click here to claim offer",
            "Buy now limited time offer click win free money",
            "Win big money free offer click here limited time",
            "Free prize claim now click here win money offer",
            "Hi, reminder about our meeting tomorrow at 10am",
            "Please review the latest project update and provide feedback",
            "Are you free for lunch today at noon?",
            "Could you review my pull request when you get a chance?",
            "Weekly sync is at 3pm on Friday, please attend",
        ],
        "label": ["spam", "spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham", "ham"],
    }
    path = os.path.join(tmp_dir, "dataset.csv")
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def make_dataset_json(tmp_dir: str) -> str:
    """Create a small labeled JSON dataset for training tests."""
    data = {
        "sender": ["spam@deals.com", "alice@work.com"],
        "subject": ["Win money", "Meeting"],
        "body": ["Click here win free money prize offer", "Reminder about our meeting tomorrow"],
        "label": ["spam", "ham"],
    }
    path = os.path.join(tmp_dir, "dataset.json")
    pd.DataFrame(data).to_json(path)
    return path


# ---------------------------------------------------------------------------
# train_pipeline — used to produce a model for classify tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_model_path():
    """Train a model once and return its path for use in classify tests."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_dataset_csv(tmp)
        model_path = os.path.join(tmp, "model.joblib")
        metadata, err = train_pipeline(csv_path, model_path, learning_rate=0.1, epochs=200)
        assert err is None, f"Training failed: {err}"
        assert os.path.exists(model_path)
        # Copy to a persistent temp file so the fixture outlives the context manager
        import shutil, tempfile as tf2
        persistent = tf2.mktemp(suffix=".joblib")
        shutil.copy(model_path, persistent)
        yield persistent
        if os.path.exists(persistent):
            os.remove(persistent)


# ---------------------------------------------------------------------------
# train_pipeline tests
# ---------------------------------------------------------------------------

def test_train_pipeline_returns_metadata_on_success():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_dataset_csv(tmp)
        model_path = os.path.join(tmp, "model.joblib")
        metadata, err = train_pipeline(csv_path, model_path)
        assert err is None
        assert isinstance(metadata, ModelMetadata)


def test_train_pipeline_metadata_fields():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_dataset_csv(tmp)
        model_path = os.path.join(tmp, "model.joblib")
        metadata, err = train_pipeline(csv_path, model_path)
        assert err is None
        assert metadata.num_training_samples == 10
        assert metadata.vocab_size > 0
        assert metadata.model_path == model_path


def test_train_pipeline_saves_model_to_disk():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = make_dataset_csv(tmp)
        model_path = os.path.join(tmp, "model.joblib")
        train_pipeline(csv_path, model_path)
        assert os.path.exists(model_path)


def test_train_pipeline_json_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        json_path = make_dataset_json(tmp)
        model_path = os.path.join(tmp, "model.joblib")
        metadata, err = train_pipeline(json_path, model_path)
        assert err is None
        assert isinstance(metadata, ModelMetadata)


def test_train_pipeline_invalid_path_returns_error():
    metadata, err = train_pipeline("/nonexistent/path/dataset.csv", "/tmp/model.joblib")
    assert metadata is None
    assert err is not None
    assert "component" in err
    assert "error_type" in err
    assert "message" in err


# ---------------------------------------------------------------------------
# classify_email tests
# ---------------------------------------------------------------------------

def test_classify_email_returns_output_and_none_on_success(trained_model_path):
    output, err = classify_email(SPAM_EMAIL, trained_model_path)
    assert err is None
    assert isinstance(output, ClassificationOutput)


def test_classify_email_output_has_required_fields(trained_model_path):
    output, err = classify_email(SPAM_EMAIL, trained_model_path)
    assert err is None
    assert output.email_id is not None
    assert output.label in (Label.Spam, Label.Ham)
    assert 0.0 <= output.confidence <= 1.0
    assert output.timestamp is not None


def test_classify_email_uses_provided_email_id(trained_model_path):
    output, err = classify_email(SPAM_EMAIL, trained_model_path, email_id="test-id-123")
    assert err is None
    assert output.email_id == "test-id-123"


def test_classify_email_generates_uuid_when_no_id(trained_model_path):
    output, err = classify_email(SPAM_EMAIL, trained_model_path)
    assert err is None
    # UUID4 is 36 chars with hyphens
    assert len(output.email_id) == 36


def test_classify_email_invalid_sender_returns_error(trained_model_path):
    output, err = classify_email(INVALID_EMAIL, trained_model_path)
    assert output is None
    assert err is not None
    assert err["component"] == "ingest_email"
    assert "error_type" in err
    assert "message" in err


def test_classify_email_empty_body_returns_error(trained_model_path):
    output, err = classify_email(EMPTY_BODY_EMAIL, trained_model_path)
    assert output is None
    assert err is not None
    assert err["component"] == "ingest_email"


def test_classify_email_missing_model_returns_error():
    output, err = classify_email(SPAM_EMAIL, "/nonexistent/model.joblib")
    assert output is None
    assert err is not None
    assert err["component"] == "load_model"
    assert "error_type" in err
    assert "message" in err


def test_classify_email_error_dict_has_all_keys(trained_model_path):
    output, err = classify_email(INVALID_EMAIL, trained_model_path)
    assert output is None
    assert set(err.keys()) == {"component", "error_type", "message"}


# ---------------------------------------------------------------------------
# classify_batch tests
# ---------------------------------------------------------------------------

def test_classify_batch_returns_list_of_same_length(trained_model_path):
    raws = [SPAM_EMAIL, HAM_EMAIL, SPAM_EMAIL]
    results = classify_batch(raws, trained_model_path)
    assert len(results) == 3


def test_classify_batch_each_element_is_tuple(trained_model_path):
    results = classify_batch([SPAM_EMAIL], trained_model_path)
    assert len(results) == 1
    output, err = results[0]
    assert isinstance(output, ClassificationOutput)
    assert err is None


def test_classify_batch_invalid_email_returns_error_in_tuple(trained_model_path):
    results = classify_batch([INVALID_EMAIL], trained_model_path)
    output, err = results[0]
    assert output is None
    assert err is not None


def test_classify_batch_mixed_valid_invalid(trained_model_path):
    results = classify_batch([SPAM_EMAIL, INVALID_EMAIL, HAM_EMAIL], trained_model_path)
    assert len(results) == 3
    assert results[0][1] is None   # valid → no error
    assert results[1][0] is None   # invalid → no output
    assert results[1][1] is not None  # invalid → has error
    assert results[2][1] is None   # valid → no error


def test_classify_batch_empty_list(trained_model_path):
    results = classify_batch([], trained_model_path)
    assert results == []
