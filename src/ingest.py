from pathlib import Path

from src.ai_utils import get_embedding
from src.text_processing import chunk_text
from src.vector_store import collection
from src.document_loader import load_document
from src.hash_utils import generate_content_hash
from src.logger import logger


DOCUMENTS_DIR = Path("data/documents")


for document_path in DOCUMENTS_DIR.iterdir():

    logger.info(f"\nProcessing document: {document_path}")

    if not document_path.is_file():
        continue

    text = load_document(document_path)

    document_hash = generate_content_hash(text)

    existing_by_source = collection.get(
        where={
            "source": str(document_path)
        }
    )

    if existing_by_source["ids"]:
        existing_metadata = existing_by_source["metadatas"][0]        

        if existing_metadata["document_hash"] == document_hash:
            logger.info(f"Skipping unchanged document: {document_path}")
            continue

        logger.info(f"Updating changed document: {document_path}")
        collection.delete(ids=existing_by_source["ids"])

    chunks = chunk_text(
        text, 
        chunk_size=300,
        overlap_sentences=1)

    for index, chunk in enumerate(chunks):
        
        logger.info(f"Embedding chunk: {chunk[:80]}...")

        embedding = get_embedding(chunk)

        collection.add(
            ids=[f"{document_path}-{index}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": str(document_path),
                "document_name": document_path.stem,
                "document_hash": document_hash
            }]	
        )

current_sources = {
    str(path)
    for path in DOCUMENTS_DIR.iterdir()
    if path.is_file()
}


existing = collection.get()


for metadata, chunk_id in zip(existing["metadatas"], existing["ids"]):
    source = metadata["source"]

    if source not in current_sources:
        logger.info(f"Removing stale chunk from deleted document: {source}")
        collection.delete(ids=[chunk_id])

logger.info("\nIngestion complete.")
