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

best_score = -1
best_chunk = ""

for item in stored_data:
    score = cosine_similarity(question_embedding, item["embedding"])

    if score > best_score:
        best_score = score
        best_chunk = item["text"]

print("\nRetrieved context:")
print(best_chunk)
print(f"\nSimilarity score: {best_score:.4f}")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": f"Answer ONLY using this context:\n\n{best_chunk}"
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

print("\nFinal answer:")
print(response.choices[0].message.content)
