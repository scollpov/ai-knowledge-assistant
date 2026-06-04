from openai import OpenAI

from src.config import (
    CHAT_MODEL,
    PROMPT_TOKEN_COST_PER_1M,
    COMPLETION_TOKEN_COST_PER_1M
)
from src.conversation_memory import get_history
from src.conversation_summary import get_summary
from src.long_term_memory import get_facts
from src.logger import logger
from src.usage_metrics import UsageMetrics

from typing import Generator


client = OpenAI()


def build_messages(
    question: str,
    context: str = ""
) -> list:

    system_prompt = (
        f"Conversation summary:\n{get_summary()}\n\n"
        f"Known user facts:\n{get_facts()}\n\n"
    )

    if context:
        system_prompt += (
            f"Answer ONLY using this context:\n\n{context}"
        )
    else:
        system_prompt += (
            "Answer using the conversation history, "
            "conversation summary, known user facts, "
            "and recent conversation messages."
        )

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(get_history())

    messages.append({
        "role": "user",
        "content": question
    })

    return messages


def generate_response(
    question: str,
    context: str = ""
) -> str:

    response = ""

    messages = build_messages(question, context)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=messages
    )

    estimated_cost_usd = (
        (response.usage.prompt_tokens / 1_000_000) * PROMPT_TOKEN_COST_PER_1M
            +
        (response.usage.completion_tokens / 1_000_000) * COMPLETION_TOKEN_COST_PER_1M
    )

    usage = UsageMetrics(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
        estimated_cost_usd=estimated_cost_usd
    )

    logger.info(
        f"Token usage - prompt: {usage.prompt_tokens}, "
        f"completion: {usage.completion_tokens}, "
        f"total: {usage.total_tokens}, "
        f"estimated_cost_usd: {usage.estimated_cost_usd:.6f}"
    )

    return response.choices[0].message.content


def stream_chat_response(messages: list) -> Generator[str, None, None]:
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=messages,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta

        if delta.content:
            yield delta.content


def stream_response(
    question: str,
    context: str = ""
) -> Generator[str, None, None]:

    messages = build_messages(question, context)

    yield from stream_chat_response(messages)
