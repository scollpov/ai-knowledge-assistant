from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional

from src.ai_utils import get_embedding
from src.vector_store import collection
from src.reranker import keyword_overlap_score
from src.query_rewriter import rewrite_query
from src.conversation_summary import get_summary
from src.config import (
    RETRIEVAL_K,
    FINAL_K,
    MAX_DISTANCE,
    CHAT_MODEL
)
from src.conversation_memory import (
    add_message,
    get_history
)
from src.memory_extractor import extract_fact
from src.long_term_memory import (
    add_fact,
    get_facts
)

load_dotenv()

client = OpenAI()

def answer_question(
    question: str,
    filter_metadata: Optional[dict] = None
) -> dict:

    fact = extract_fact(question)

    if fact:
        print(f"\nRemembered fact: {fact}")
        add_fact(fact)

    rewritten_query = rewrite_query(
        get_history(),
        question
    )

    print(f"\nRewritten query: {rewritten_query}")

    question_embedding = get_embedding(
        rewritten_query
    )

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
        messages = [
            {
                "role": "system",
                "content":
                    f"Conversation summary:\n{get_summary()}\n\n"
                    f"Known user facts:\n{get_facts()}\n\n"  
                    "Answer using the conversation history."
            }
        ]

        messages.extend(get_history())

        messages.append({
            "role": "user",
            "content": question
        })

        response = client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0,
            messages=messages
        )

        answer = response.choices[0].message.content

        add_message("user", question)
        add_message("assistant", answer)

        return {
            "answer": answer,
            "sources": []
        }

    sources = sources[:FINAL_K]

    for source in sources:
        context += source["text"] + "\n\n"

    messages = [
        {
            "role": "system",
            "content":
                f"Conversation summary:\n{get_summary()}\n\n"
                f"Known user facts:\n{get_facts()}\n\n" 
                f"Answer ONLY using this context:\n\n{context}"
        }
    ]

    messages.extend(get_history())

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=messages
    )

    answer = response.choices[0].message.content

    add_message("user", question)
    add_message("assistant", answer)

    return {
        "answer": answer,
        "sources": sources
    }
