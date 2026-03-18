# Email Spam Classifier

A modular email spam classifier built with Python and standard ML tools. Classifies emails as **spam** or **ham** using TF-IDF feature extraction and logistic regression, with a Streamlit web UI for training and inference.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-NLP-green)
![pytest](https://img.shields.io/badge/pytest-0A9EDC?logo=pytest&logoColor=white)
![Hypothesis](https://img.shields.io/badge/Hypothesis-PBT-purple)
![joblib](https://img.shields.io/badge/joblib-serialization-orange)

---

## Features

- Modular NLP pipeline: ingestion → preprocessing → TF-IDF → logistic regression
- Train on any labeled CSV/JSON dataset
- Real-time single email classification
- Batch classification with downloadable results
- Streamlit web UI — no frontend build step needed
- 93 tests covering unit, integration, and property-based correctness

---

## Project Structure

```
email_spam_classifier/
├── models.py            # shared dataclasses (RawEmail, TrainedModel, Label, ...)
├── ingestion.py         # email validation and parsing
├── preprocessor.py      # normalize → tokenize → stop words → stem
├── feature_extractor.py # TF-IDF vectorizer and vocabulary builder
├── classifier.py        # logistic regression classify + train_model
├── model_store.py       # joblib model save/load
├── result_aggregator.py # attach metadata and review flag to results
└── pipeline.py          # classify_email, classify_batch, train_pipeline

tests/                   # 93 passing tests (pytest + hypothesis)
app.py                   # Streamlit web UI
setup_nltk.py            # downloads required NLTK data
requirements.txt
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download NLTK data

```bash
python setup_nltk.py
```

### 3. Run the web UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset Format

Training datasets must be CSV or JSON with these columns:

| Column    | Description                        |
|-----------|------------------------------------|
| `sender`  | Sender email address               |
| `subject` | Email subject line                 |
| `body`    | Email body text                    |
| `label`   | `spam` or `ham`                    |

Example:

```csv
sender,subject,body,label
spam@deals.com,Win money now!,Click here to claim your free prize!,spam
alice@work.com,Meeting tomorrow,Hi reminder about our 10am meeting.,ham
```

---

## Python API

```python
from email_spam_classifier.models import RawEmail
from email_spam_classifier.pipeline import train_pipeline, classify_email

# Train
metadata, err = train_pipeline("dataset.csv", "models/spam_classifier.joblib")

# Classify a single email
raw = RawEmail(
    sender="offers@deals.com",
    subject="You won a prize!",
    body="Click here to claim your free gift. Limited time offer!",
)
output, err = classify_email(raw, "models/spam_classifier.joblib")
print(output.label, output.confidence)  # Label.Spam  0.97
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## How It Works

1. **Ingestion** — validates sender format and body, returns structured `EmailContent`
2. **Preprocessing** — lowercases, strips HTML/punctuation, removes stop words, applies Porter stemming
3. **Feature Extraction** — builds a vocabulary (top 10,000 tokens by frequency), computes TF-IDF vectors
4. **Classification** — logistic regression: `confidence = sigmoid(bias + dot(weights, features))`
5. **Result Aggregation** — attaches email ID, ISO 8601 timestamp, and a low-confidence review flag

---

## License

[MIT](LICENSE)
