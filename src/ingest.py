from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

KNOWLEDGE_FILE = "data/knowledge.txt"
EMBEDDINGS_FILE = "data/embeddings.json"

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

with open(KNOWLEDGE_FILE, "r") as file:
    text = file.read()

knowledge_chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

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
