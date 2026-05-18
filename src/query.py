from openai import OpenAI
from dotenv import load_dotenv
from ai_utils import get_embedding, cosine_similarity
import json

load_dotenv()
client = OpenAI()

EMBEDDINGS_FILE = "data/embeddings.json"

TOP_K = 3
MIN_SIMILARITY_SCORE = 0.35

with open(EMBEDDINGS_FILE, "r") as file:
    stored_data = json.load(file)

question = input("Ask a question: ")

question_embedding = get_embedding(question)

scored_chunks = []

for item in stored_data:
    score = cosine_similarity(
        question_embedding,
        item["embedding"]
    )

    scored_chunks.append({
        "id": item["id"],
        "source": item["source"],
        "text": item["text"],
        "score": score
    })

scored_chunks.sort(
    key=lambda x: x["score"],
    reverse=True
)

print("\nAll retrieval scores:\n")

for chunk in scored_chunks:
    print(f"{chunk['score']:.4f} -> {chunk['text'][:80]}")

top_chunks = scored_chunks[:TOP_K]

if not top_chunks or top_chunks[0]["score"] < MIN_SIMILARITY_SCORE:
    print("\nNo sufficiently relevant context found.")
    print("Try improving the knowledge base or asking a more specific question.")
    exit()

print("\nTop retrieved chunks:\n")

context = ""

for i, chunk in enumerate(top_chunks):
    print(f"Rank {i+1}")
    print(f"Score: {chunk['score']:.4f}")
    print(f"Source ID: {chunk['id']}")
    print(f"Source File: {chunk['source']}")
    print(chunk["text"])
    print()

    context += chunk["text"] + "\n\n"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": f"Answer ONLY using this context:\n\n{context}"
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

print("\nFinal answer:")
print(response.choices[0].message.content)
