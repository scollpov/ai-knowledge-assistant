from typing import Dict, List, Optional

from src.ai_utils import get_embedding
from src.config import RETRIEVAL_K, FINAL_K, MAX_DISTANCE
from src.reranker import keyword_overlap_score
from src.vector_store import collection


def retrieve_sources(
    rewritten_query: str,
    filter_metadata: Optional[dict] = None
) -> List[Dict]:
    question_embedding = get_embedding(rewritten_query)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=RETRIEVAL_K,
        where=filter_metadata if filter_metadata else None
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    if not documents: return []

    if distances[0] > MAX_DISTANCE: return []

    sources = []

    for doc, metadata, distance, doc_id in zip(
        documents,
        metadatas,
        distances,
        ids
    ):

        score = distance

        sources.append({
            "id": doc_id,
            "source": metadata.get("source"),
            "text": doc,
            "score": distance,
            "rerank_score": keyword_overlap_score(rewritten_query, doc)
        })

    sources.sort(
        key=lambda source: source["rerank_score"],
        reverse=True
    )

    return sources[:FINAL_K]
