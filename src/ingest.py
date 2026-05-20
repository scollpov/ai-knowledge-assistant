from pathlib import Path

from src.ai_utils import get_embedding
from src.text_processing import chunk_text
from src.vector_store import collection
from src.document_loader import load_document

DOCUMENTS_DIR = Path("data/documents")

existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])

chunk_id = 1

for document_path in DOCUMENTS_DIR.iterdir():

    print(f"\nProcessing document: {document_path}")

    if not document_path.is_file():
        continue

    text = load_document(document_path)

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
