"""Streamlit UI for the Email Spam Classifier."""

import os
import tempfile

import pandas as pd
import streamlit as st

from email_spam_classifier.models import Label, RawEmail
from email_spam_classifier.pipeline import classify_batch, classify_email, train_pipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="wide",
)

MODEL_PATH = "models/spam_classifier.joblib"

# ── Helpers ───────────────────────────────────────────────────────────────────

def model_exists() -> bool:
    return os.path.exists(MODEL_PATH)


def label_badge(label: Label, confidence: float) -> str:
    if label == Label.Spam:
        return f"🔴 **SPAM** ({confidence:.1%})"
    return f"🟢 **HAM** ({confidence:.1%})"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📧 Spam Classifier")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Classify Email", "Batch Classify", "Train Model"])

model_status = "✅ Model loaded" if model_exists() else "⚠️ No model found — train one first"
st.sidebar.markdown(f"**Model status:** {model_status}")
st.sidebar.markdown(f"`{MODEL_PATH}`")

# ── Page: Classify Email ──────────────────────────────────────────────────────
if page == "Classify Email":
    st.title("Classify a Single Email")
    st.markdown("Enter email details below and check if it's spam or ham.")

    with st.form("classify_form"):
        sender = st.text_input("Sender", placeholder="sender@example.com")
        subject = st.text_input("Subject", placeholder="Email subject line")
        body = st.text_area("Body", placeholder="Email body text...", height=180)
        submitted = st.form_submit_button("Classify", type="primary")

    if submitted:
        if not model_exists():
            st.error("No trained model found. Go to **Train Model** first.")
        else:
            raw = RawEmail(sender=sender, subject=subject, body=body)
            output, err = classify_email(raw, MODEL_PATH)

            if err:
                st.error(f"**Error** in `{err['component']}`: {err['message']}")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Result", "SPAM" if output.label == Label.Spam else "HAM")
                col2.metric("Confidence", f"{output.confidence:.1%}")
                col3.metric("Needs Review", "Yes" if output.needs_review else "No")

                st.markdown(f"### {label_badge(output.label, output.confidence)}")

                if output.needs_review:
                    st.warning("Low confidence — consider manual review.")

                with st.expander("Details"):
                    st.json({
                        "email_id": output.email_id,
                        "label": output.label.value,
                        "confidence": round(output.confidence, 4),
                        "timestamp": output.timestamp,
                        "needs_review": output.needs_review,
                    })

# ── Page: Batch Classify ──────────────────────────────────────────────────────
elif page == "Batch Classify":
    st.title("Batch Classify Emails")
    st.markdown(
        "Upload a CSV file with columns: `sender`, `subject`, `body`. "
        "The classifier will label each row."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = None
        for encoding in ("utf-8", "latin-1", "windows-1252", "utf-8-sig"):
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                break
        if df is None:
            st.error("Could not decode the CSV file. Try re-saving it as UTF-8.")
        else:
            st.write(f"Loaded **{len(df)} emails**. Preview:")
            st.dataframe(df.head(5), use_container_width=True)

        if df is not None and st.button("Run Classification", type="primary"):
            if not model_exists():
                st.error("No trained model found. Go to **Train Model** first.")
            else:
                raws = [
                    RawEmail(
                        sender=str(row.get("sender", "")),
                        subject=str(row.get("subject", "")),
                        body=str(row.get("body", "")),
                    )
                    for _, row in df.iterrows()
                ]

                with st.spinner("Classifying..."):
                    results = classify_batch(raws, MODEL_PATH)

                rows = []
                for i, (output, err) in enumerate(results):
                    if err:
                        rows.append({
                            "sender": raws[i].sender,
                            "subject": raws[i].subject,
                            "label": "ERROR",
                            "confidence": None,
                            "needs_review": None,
                            "error": err["message"],
                        })
                    else:
                        rows.append({
                            "sender": raws[i].sender,
                            "subject": raws[i].subject,
                            "label": output.label.value.upper(),
                            "confidence": f"{output.confidence:.1%}",
                            "needs_review": output.needs_review,
                            "error": "",
                        })

                result_df = pd.DataFrame(rows)
                spam_count = sum(1 for r in rows if r["label"] == "SPAM")
                ham_count = sum(1 for r in rows if r["label"] == "HAM")

                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(rows))
                col2.metric("Spam", spam_count)
                col3.metric("Ham", ham_count)

                st.dataframe(result_df, use_container_width=True)

                csv_out = result_df.to_csv(index=False).encode()
                st.download_button("Download Results CSV", csv_out, "results.csv", "text/csv")

# ── Page: Train Model ─────────────────────────────────────────────────────────
elif page == "Train Model":
    st.title("Train the Classifier")
    st.markdown(
        "Upload a labeled CSV with columns: `body`, `label` "
        "(where `label` is `spam` or `ham`). `sender` and `subject` are optional."
    )

    with st.expander("Sample dataset format"):
        st.dataframe(pd.DataFrame({
            "body": ["Click here to claim your free prize!", "Hi, reminder about our 10am meeting."],
            "label": ["spam", "ham"],
        }))

    uploaded = st.file_uploader("Upload labeled CSV", type=["csv"])

    col1, col2 = st.columns(2)
    learning_rate = col1.slider("Learning rate", 0.001, 1.0, 0.1, step=0.001, format="%.3f")
    epochs = col2.slider("Epochs", 10, 500, 100, step=10)

    if uploaded and st.button("Train Model", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        try:
            with st.spinner("Training... this may take a moment."):
                metadata, err = train_pipeline(
                    tmp_path, MODEL_PATH,
                    learning_rate=learning_rate,
                    epochs=epochs,
                )
        finally:
            os.unlink(tmp_path)

        if err:
            st.error(f"**Training failed** in `{err['component']}`: {err['message']}")
            st.info("Make sure your CSV has columns: `body`, `label`. `sender` and `subject` are optional. The `label` column must contain `spam` or `ham`.")
        else:
            st.success("Model trained and saved successfully.")
            col1, col2, col3 = st.columns(3)
            col1.metric("Training Samples", metadata.num_training_samples)
            col2.metric("Vocabulary Size", metadata.vocab_size)
            col3.metric("Saved to", metadata.model_path)
            st.rerun()
