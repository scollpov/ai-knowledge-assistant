from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

knowledge_chunks = [
    "Python is widely used in AI engineering and backend systems.",
    "FastAPI is a lightweight Python framework for building APIs.",
    "RAG reduces hallucinations by retrieving relevant context before generation.",
    "Docker containers help developers deploy applications consistently."
]

EMBEDDINGS_FILE = "data/embeddings.json"

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

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
