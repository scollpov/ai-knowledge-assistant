from pathlib import Path
from pypdf import PdfReader
from src.text_cleaner import clean_text

def load_text_document(file_path: Path) -> str:

    with open(file_path, "r") as file:
        return file.read()


def load_pdf_document(file_path: Path) -> str:

    reader = PdfReader(str(file_path))

    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    return text


def load_document(file_path: Path) -> str:

    if file_path.suffix == ".txt":
        return clean_text(load_text_document(file_path))

    if file_path.suffix == ".pdf":
        return clean_text(load_pdf_document(file_path))

    raise ValueError(f"Unsupported file type: {file_path}")
