from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import json
import os

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

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Load or create embeddings
if os.path.exists(EMBEDDINGS_FILE):
    print("Loading embeddings from file...\n")

    with open(EMBEDDINGS_FILE, "r") as file:
        stored_data = json.load(file)

else:
    print("Creating embeddings...\n")

    stored_data = []

    for chunk in knowledge_chunks:
        embedding = get_embedding(chunk)

        stored_data.append({
            "text": chunk,
            "embedding": embedding
        })

    with open(EMBEDDINGS_FILE, "w") as file:
        json.dump(stored_data, file)

    print("Embeddings saved.\n")

question = input("Ask a question: ")

question_embedding = get_embedding(question)

best_score = -1
best_chunk = ""

for item in stored_data:
    score = cosine_similarity(
        question_embedding,
        item["embedding"]
    )

    print(f"\nChunk: {item['text']}")
    print(f"Similarity: {score:.4f}")

    if score > best_score:
        best_score = score
        best_chunk = item["text"]

print("\nBest retrieved chunk:\n")
print(best_chunk)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": f"""
            Answer the question ONLY using this context:

            {best_chunk}
            """
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

print("\nFinal Answer:\n")
print(response.choices[0].message.content)
