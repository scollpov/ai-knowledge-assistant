from src.ai_utils import get_embedding
from src.text_processing import chunk_text
from pathlib import Path
import json

DOCUMENTS_DIR = Path("data/documents")
EMBEDDINGS_FILE = "data/embeddings.json"

stored_data = []

for document_path in DOCUMENTS_DIR.glob("*.txt"):
    print(f"\nProcessing document: {document_path}")

    with open(document_path, "r") as file:
        text = file.read()

    chunks = chunk_text(text, 
        chunk_size=300,
        overlap_sentences=1)

    for chunk in chunks:
        print(f"Embedding chunk: {chunk[:80]}...")

        embedding = get_embedding(chunk)

        stored_data.append({
            "id": len(stored_data) + 1,
            "source": str(document_path),
            "text": chunk,
            "embedding": embedding
        })

with open(EMBEDDINGS_FILE, "w") as file:
    json.dump(stored_data, file)

print("\nIngestion complete.")
print(f"Saved {len(stored_data)} chunks to {EMBEDDINGS_FILE}.")
