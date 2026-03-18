"""Download required NLTK data packages."""

import nltk


def setup():
    nltk.download("stopwords")
    nltk.download("punkt")
    nltk.download("punkt_tab")


if __name__ == "__main__":
    setup()
    print("NLTK data downloaded successfully.")
