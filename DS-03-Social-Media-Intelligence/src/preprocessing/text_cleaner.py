
"""Text normalization utilities."""
import re


def clean_text(text: str) -> str:
    """Remove URLs and normalize whitespace while preserving words and emoji."""
    text = str(text or "")
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Return a language code, or 'unknown' when detection is unavailable."""
    try:
        from langdetect import detect
        return detect(text) if text.strip() else "unknown"
    except Exception:
        return "unknown"
