from ai_utils import get_embedding
from text_processing import chunk_text
import json

KNOWLEDGE_FILE = "data/knowledge.txt"
EMBEDDINGS_FILE = "data/embeddings.json"

with open(KNOWLEDGE_FILE, "r") as file:
    text = file.read()

knowledge_chunks = chunk_text(text)

print("\nGenerated chunks:\n")

for i, chunk in enumerate(knowledge_chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print()

stored_data = []

for chunk in knowledge_chunks:
    print(f"Embedding: {chunk}")
    embedding = get_embedding(chunk)

    stored_data.append({
        "text": chunk,
        "embedding": embedding
    })

with open(EMBEDDINGS_FILE, "w") as file:
    json.dump(stored_data, file)

print("\nIngestion complete.")
print(f"Saved {len(stored_data)} chunks to {EMBEDDINGS_FILE}.")
