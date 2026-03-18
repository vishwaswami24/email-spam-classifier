"""Unit tests for model_store.py — save_model and load_model."""

import os
import tempfile

import pytest

from email_spam_classifier.model_store import save_model, load_model
from email_spam_classifier.models import TrainedModel


def make_model():
    return TrainedModel(
        weights=[0.1, -0.2, 0.3],
        bias=0.05,
        vocab_size=3,
        vocabulary={"spam": 0, "free": 1, "offer": 2},
    )


# ---------------------------------------------------------------------------
# save_model + load_model round-trip
# ---------------------------------------------------------------------------

def test_save_then_load_returns_equivalent_model():
    model = make_model()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        _, err = save_model(model, path)
        assert err is None

        loaded, err = load_model(path)
        assert err is None
        assert loaded is not None
        assert loaded.weights == model.weights
        assert loaded.bias == model.bias
        assert loaded.vocab_size == model.vocab_size
        assert loaded.vocabulary == model.vocabulary
    finally:
        os.unlink(path)


def test_save_creates_parent_directories(tmp_path):
    model = make_model()
    nested_path = str(tmp_path / "a" / "b" / "model.joblib")
    _, err = save_model(model, nested_path)
    assert err is None
    assert os.path.exists(nested_path)


def test_save_returns_none_none_on_success():
    model = make_model()
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        path = f.name
    try:
        result, err = save_model(model, path)
        assert result is None
        assert err is None
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# load_model — error cases
# ---------------------------------------------------------------------------

def test_load_from_nonexistent_path_returns_error():
    loaded, err = load_model("/nonexistent/path/model.joblib")
    assert loaded is None
    assert err is not None
    assert "not found" in err.lower() or "nonexistent" in err.lower() or "no such" in err.lower()


def test_load_corrupt_file_returns_error(tmp_path):
    corrupt_path = str(tmp_path / "corrupt.joblib")
    with open(corrupt_path, "wb") as f:
        f.write(b"this is not a valid joblib file")

    loaded, err = load_model(corrupt_path)
    assert loaded is None
    assert err is not None


def test_load_wrong_type_returns_error(tmp_path):
    import joblib
    path = str(tmp_path / "wrong_type.joblib")
    joblib.dump({"not": "a model"}, path)

    loaded, err = load_model(path)
    assert loaded is None
    assert err is not None
    assert "TrainedModel" in err or "not a" in err.lower()
