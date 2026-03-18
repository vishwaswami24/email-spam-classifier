"""End-to-end classification and training pipelines."""

import uuid
from typing import List, Optional, Tuple

import pandas as pd

from email_spam_classifier.classifier import classify, train_model
from email_spam_classifier.feature_extractor import (
    build_doc_freq,
    build_vocabulary,
    compute_tfidf,
    extract_features,
)
from email_spam_classifier.ingestion import ingest_email
from email_spam_classifier.model_store import load_model, save_model
from email_spam_classifier.models import (
    ClassificationOutput,
    EmailContent,
    Label,
    LabeledEmail,
    ModelMetadata,
    RawEmail,
)
from email_spam_classifier.preprocessor import preprocess_email
from email_spam_classifier.result_aggregator import build_output


def classify_email(
    raw: RawEmail,
    model_path: str,
    email_id: str = None,
) -> Tuple[Optional[ClassificationOutput], Optional[dict]]:
    """Run a single email through the full classification pipeline.

    Returns (ClassificationOutput, None) on success, or (None, error_dict) on failure.
    """
    if email_id is None:
        email_id = str(uuid.uuid4())

    content, err = ingest_email(raw)
    if err is not None:
        return None, {"component": "ingest_email", "error_type": "ValidationError", "message": err}

    try:
        cleaned = preprocess_email(content)
    except Exception as exc:
        return None, {"component": "preprocess_email", "error_type": type(exc).__name__, "message": str(exc)}

    model, err = load_model(model_path)
    if err is not None:
        return None, {"component": "load_model", "error_type": "ModelLoadError", "message": err}

    try:
        features = extract_features(cleaned, model.vocabulary)
    except Exception as exc:
        return None, {"component": "extract_features", "error_type": type(exc).__name__, "message": str(exc)}

    result, err = classify(model, features)
    if err is not None:
        return None, {"component": "classify", "error_type": "ClassificationError", "message": err}

    try:
        output = build_output(email_id, result)
    except Exception as exc:
        return None, {"component": "build_output", "error_type": type(exc).__name__, "message": str(exc)}

    return output, None


def classify_batch(
    raws: List[RawEmail],
    model_path: str,
) -> List[Tuple[Optional[ClassificationOutput], Optional[dict]]]:
    """Classify a list of emails, returning one (output, error) tuple per input."""
    return [classify_email(raw, model_path) for raw in raws]


def train_pipeline(
    dataset_path: str,
    model_path: str,
    learning_rate: float = 0.1,
    epochs: int = 100,
) -> Tuple[Optional[ModelMetadata], Optional[dict]]:
    """Load a labeled dataset, train a model, save it, and return ModelMetadata.

    Returns (ModelMetadata, None) on success, or (None, error_dict) on failure.
    """
    try:
        if dataset_path.endswith(".json"):
            df = pd.read_json(dataset_path)
        else:
            for encoding in ("utf-8", "latin-1", "windows-1252", "utf-8-sig"):
                try:
                    df = pd.read_csv(dataset_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return None, {
                    "component": "train_pipeline",
                    "error_type": "UnicodeDecodeError",
                    "message": "Could not decode the CSV file. Try saving it as UTF-8.",
                }
    except Exception as exc:
        return None, {"component": "train_pipeline", "error_type": type(exc).__name__, "message": str(exc)}

    # Normalize column names to lowercase to handle variations like "Label", "LABEL"
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"body", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        return None, {
            "component": "train_pipeline",
            "error_type": "ValueError",
            "message": (
                f"Dataset is missing required column(s): {', '.join(sorted(missing))}. "
                f"Found columns: {', '.join(df.columns.tolist())}. "
                "Required: body, label. Optional: sender, subject."
            ),
        }

    try:
        labeled_emails: List[LabeledEmail] = []
        for _, row in df.iterrows():
            sender = str(row["sender"]) if "sender" in df.columns and pd.notna(row.get("sender")) else "unknown@unknown.com"
            content = EmailContent(
                subject=str(row["subject"]) if "subject" in df.columns and pd.notna(row.get("subject")) else "",
                body=str(row["body"]),
                sender=sender,
            )
            label = Label.Spam if str(row["label"]).strip().lower() == "spam" else Label.Ham
            labeled_emails.append(LabeledEmail(content=content, label=label))
    except Exception as exc:
        return None, {"component": "train_pipeline", "error_type": type(exc).__name__, "message": str(exc)}

    try:
        cleaned_emails = [preprocess_email(le.content) for le in labeled_emails]
    except Exception as exc:
        return None, {"component": "preprocess_email", "error_type": type(exc).__name__, "message": str(exc)}

    try:
        vocab = build_vocabulary(cleaned_emails)
        doc_freq = build_doc_freq(cleaned_emails)
        corpus_size = len(cleaned_emails)
    except Exception as exc:
        return None, {"component": "build_vocabulary", "error_type": type(exc).__name__, "message": str(exc)}

    try:
        feature_vectors = [
            compute_tfidf(tokens=ce.tokens, vocab=vocab, corpus_size=corpus_size, doc_freq=doc_freq)
            for ce in cleaned_emails
        ]
    except Exception as exc:
        return None, {"component": "extract_features", "error_type": type(exc).__name__, "message": str(exc)}

    labels = [le.label for le in labeled_emails]
    model, err = train_model(feature_vectors, labels, learning_rate=learning_rate, epochs=epochs)
    if err is not None:
        return None, {"component": "train_model", "error_type": "TrainingError", "message": err}

    model.vocabulary = vocab

    _, err = save_model(model, model_path)
    if err is not None:
        return None, {"component": "save_model", "error_type": "ModelSaveError", "message": err}

    return ModelMetadata(
        vocab_size=len(vocab),
        num_training_samples=len(labeled_emails),
        model_path=model_path,
    ), None
