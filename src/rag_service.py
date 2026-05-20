from openai import OpenAI
from dotenv import load_dotenv

from src.ai_utils import get_embedding
from src.vector_store import collection
from src.config import (
    TOP_K,
    MAX_DISTANCE,
    CHAT_MODEL
)

load_dotenv()

client = OpenAI()

def answer_question(question: str) -> dict:

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    if not documents:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    sources = []

    context = ""

    for doc, metadata, distance, doc_id in zip(
        documents,
        metadatas,
        distances,
        ids
    ):

        score = distance
	
        sources.append({
            "id": int(doc_id),
            "source": metadata["source"],
            "text": doc,
            "score": score
        })

        context += doc + "\n\n"

    if sources[0]["score"] > MAX_DISTANCE:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    response = client.chat.completions.create(
        model=CHAT_MODEL,
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
        "sources": sources
    }
