from openai import OpenAI

from src.config import CHAT_MODEL
from src.conversation_memory import get_history
from src.conversation_summary import get_summary
from src.long_term_memory import get_facts

client = OpenAI()


def generate_response(
    question: str,
    context: str = ""
) -> str:

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

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=messages
    )

    return response.choices[0].message.content
