import re

def clean_text(text: str) -> str:

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove repeated newlines
    text = re.sub(r"\n+", "\n", text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text
