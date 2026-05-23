from openai import OpenAI
from src.config import CHAT_MODEL

client = OpenAI()


def should_retrieve(question: str) -> bool:

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Determine whether the user question requires "
                    "document retrieval from a knowledge base. "
                    "Answer ONLY YES or NO."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    return answer == "YES"
