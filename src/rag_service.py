from openai import OpenAI
from dotenv import load_dotenv
from src.ai_utils import get_embedding, cosine_similarity
import json

load_dotenv()
client = OpenAI()

EMBEDDINGS_FILE = "data/embeddings.json"
TOP_K = 3
MIN_SIMILARITY_SCORE = 0.25


def answer_question(question: str) -> dict:
    with open(EMBEDDINGS_FILE, "r") as file:
        stored_data = json.load(file)

    question_embedding = get_embedding(question)

    scored_chunks = []

    for item in stored_data:
        score = cosine_similarity(question_embedding, item["embedding"])

        scored_chunks.append({
            "id": item["id"],
            "source": item["source"],
            "text": item["text"],
            "score": score
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)

    top_chunks = scored_chunks[:TOP_K]

    if not top_chunks or top_chunks[0]["score"] < MIN_SIMILARITY_SCORE:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    context = ""

    for chunk in top_chunks:
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

    return {
        "answer": response.choices[0].message.content,
        "sources": top_chunks
    }
