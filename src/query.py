from openai import OpenAI
from dotenv import load_dotenv
from ai_utils import get_embedding, cosine_similarity
import json

load_dotenv()
client = OpenAI()

EMBEDDINGS_FILE = "data/embeddings.json"

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
        "text": item["text"],
        "score": score
    })

scored_chunks.sort(
    key=lambda x: x["score"],
    reverse=True
)

top_chunks = scored_chunks[:3]

print("\nTop retrieved chunks:\n")

context = ""

for i, chunk in enumerate(top_chunks):
    print(f"Rank {i+1}")
    print(f"Score: {chunk['score']:.4f}")
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
