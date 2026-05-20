from pathlib import Path

from src.ai_utils import get_embedding
from src.text_processing import chunk_text
from src.vector_store import collection

DOCUMENTS_DIR = Path("data/documents")

existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])

chunk_id = 1

for document_path in DOCUMENTS_DIR.glob("*.txt"):

    print(f"\nProcessing document: {document_path}")

    with open(document_path, "r") as file:
        text = file.read()

    chunks = chunk_text(
        text, 
        chunk_size=300,
        overlap_sentences=1)

    for chunk in chunks:
        
        print(f"Embedding chunk: {chunk[:80]}...")

        embedding = get_embedding(chunk)

        collection.add(
            ids=[str(chunk_id)],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": str(document_path),
                "document_name": document_path.stem
            }]	
        )

        chunk_id += 1

print("\nIngestion complete.")
