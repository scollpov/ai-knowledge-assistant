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


def answer_question(
    question: str,
    filter_metadata: Optional[dict] = None
) -> dict:

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

    logger.info(f"\nRewritten query: {rewritten_query}")

    results = retrieve_sources(
        rewritten_query=rewritten_query, 
        filter_metadata=filter_metadata
    )

    if not retrieve or not results:
        answer = generate_response(question)

        add_message("user", question)
        add_message("assistant", answer)

        return {
            "answer": answer,
            "sources": []
        } 
    
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    answer = generate_response(
        question,
        context
    )

    add_message("user", question)
    add_message("assistant", answer)

    return {
        "answer": answer,
        "sources": results
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

    logger.info(f"\nRewritten query: {rewritten_query}")

    results = retrieve_sources(
        rewritten_query=rewritten_query,
        filter_metadata=filter_metadata
    )

    if not retrieve or not results:
        yield from stream_response(question)
        return

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    yield from stream_response(question, context)
