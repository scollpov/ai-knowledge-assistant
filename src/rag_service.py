import json
import time

from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional
from typing import Generator

from src.ai_utils import get_embedding
from src.vector_store import collection
from src.reranker import keyword_overlap_score
from src.query_rewriter import rewrite_query
from src.conversation_summary import get_summary
from src.config import (
    RETRIEVAL_K,
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
from src.retrieval_router import should_retrieve
from src.retrieval_service import retrieve_sources
from src.logger import logger
from src.chat_service import (
    stream_response,
    generate_response
)


load_dotenv()

client = OpenAI()


def sse_event(event: str, data: str) -> str:
    return f"event: {event}\ndata: {data}\n\n"


def answer_question(
    question: str,
    filter_metadata: Optional[dict] = None
) -> dict:

    rewrite_time = 0
    retrieval_time = 0
    generation_time = 0

    start_time = time.perf_counter()

    fact = extract_fact(question)

    if fact:
        logger.info(f"\nRemembered fact: {fact}")
        add_fact(fact)

    retrieve = should_retrieve(question)

    logger.info(f"\nShould retrieve: {retrieve}")

    rewrite_start = time.perf_counter()

    rewritten_query = rewrite_query(
        get_history(),
        question
    )

    rewrite_time = (
        time.perf_counter()
        - rewrite_start
    )

    logger.info(f"\nRewritten query: {rewritten_query}")

    retrieval_start = time.perf_counter()

    sources = retrieve_sources(
        rewritten_query=rewritten_query, 
        filter_metadata=filter_metadata
    )

    retrieval_time = (
        time.perf_counter()
        - retrieval_start
    )

    if not retrieve or not sources:
        answer = generate_response(question)

        add_message("user", question)
        add_message("assistant", answer)

        total_time = time.perf_counter() - start_time

        logger.info(
            f"Request latency - total: {total_time:.3f}s"
        )

        return {
            "answer": answer,
            "sources": []
        } 

    context = "\n\n".join(
        source["text"]
        for source in sources
    )

    generation_start = time.perf_counter()

    answer = generate_response(
        question,
        context
    )

    generation_time = (
        time.perf_counter()
        - generation_start
    )

    add_message("user", question)
    add_message("assistant", answer)

    total_time = time.perf_counter() - start_time

    logger.info(
        f"Latency metrics - "
        f"rewrite: {rewrite_time:.3f}s, "
        f"retrieval: {retrieval_time:.3f}s, "
        f"generation: {generation_time:.3f}s, "
        f"total: {total_time:.3f}s"
    )

    return {
        "answer": answer,
        "sources": sources
    }


def stream_answer_question(
    question: str,
    filter_metadata: Optional[dict] = None
) -> Generator[str, None, None]:

    fact = extract_fact(question)

    if fact:
        logger.info(f"\nRemembered fact: {fact}")
        add_fact(fact)

    retrieve = should_retrieve(question)

    logger.info(f"\nShould retrieve: {retrieve}")

    rewritten_query = rewrite_query(
        get_history(),
        question
    )

    yield sse_event(
        "retrieval",
        "rewriting query"
    )

    logger.info(f"\nRewritten query: {rewritten_query}")

    yield sse_event(
        "retrieval",
        "searching documents"
    )

    sources = retrieve_sources(
        rewritten_query=rewritten_query,
        filter_metadata=filter_metadata
    )

    if not retrieve or not sources:
        yield from stream_response(question)
        return

    yield sse_event(
        "retrieval",
        f"retrieved {len(sources)} sources"
    )

    context = "\n\n".join(
        source["text"]
        for source in sources
    )

    unique_sources = {}

    for source in sources:
        current = unique_sources.get(source["source"])

        if (
            current is None
            or source["score"] < current["score"]
        ):
            unique_sources[source["source"]] = {
                "source": source["source"],
                "score": source["score"]
            }

    yield sse_event(
        "sources",
        json.dumps(list(unique_sources.values()))
    )

    yield sse_event("status", "generating")

    full_answer = ""

    for token in stream_response(question, context):
        full_answer += token

        yield sse_event("token", token)

    add_message("user", question)
    add_message("assistant", full_answer)
        
    yield sse_event("done", "[DONE]")
