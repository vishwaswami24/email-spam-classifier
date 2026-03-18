"""Model persistence — save and load TrainedModel artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import joblib

from email_spam_classifier.models import TrainedModel


def save_model(model: TrainedModel, path: str) -> Tuple[None, Optional[str]]:
    """Serialize a TrainedModel to disk using joblib.

    Returns (None, None) on success, (None, error_string) on failure.
    """
    try:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, dest)
        return (None, None)
    except Exception as exc:
        return (None, str(exc))


def load_model(path: str) -> Tuple[Optional[TrainedModel], Optional[str]]:
    """Deserialize a TrainedModel from disk using joblib.

    Returns (TrainedModel, None) on success, (None, error_string) on failure.
    """
    try:
        obj = joblib.load(path)
    except FileNotFoundError:
        return (None, f"Model file not found: {path}")
    except EOFError:
        return (None, f"Model file is corrupt or empty: {path}")
    except Exception as exc:
        return (None, f"Failed to load model from {path}: {exc}")

    if not isinstance(obj, TrainedModel):
        return (None, f"Loaded object is not a TrainedModel (got {type(obj).__name__})")

    return (obj, None)
