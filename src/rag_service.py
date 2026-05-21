from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional

from src.ai_utils import get_embedding
from src.vector_store import collection
from src.reranker import keyword_overlap_score
from src.config import (
    RETRIEVAL_K,
    FINAL_K,
    MAX_DISTANCE,
    CHAT_MODEL
)

load_dotenv()

client = OpenAI()

def answer_question(
    question: str,
    filter_metadata: Optional[dict] = None
) -> dict:

    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=RETRIEVAL_K,
        where=filter_metadata if filter_metadata else None
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
        rerank_score = keyword_overlap_score(question,doc)

        sources.append({
            "id": doc_id,
            "source": metadata["source"],
            "text": doc,
            "score": score,
            "rerank_score": rerank_score
        })

    sources.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )

    if sources[0]["score"] > MAX_DISTANCE:
        return {
            "answer": "No relevant context found.",
            "sources": []
        }

    sources = sources[:FINAL_K]

    for source in sources:
        context += source["text"] + "\n\n"

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
