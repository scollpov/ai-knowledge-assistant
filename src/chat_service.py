from openai import OpenAI

from src.config import CHAT_MODEL
from src.conversation_memory import get_history
from src.conversation_summary import get_summary
from src.long_term_memory import get_facts
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
