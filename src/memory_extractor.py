from openai import OpenAI
from typing import Optional

client = OpenAI()

def extract_fact(message: str) -> Optional[str]:

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Determine whether the user message contains a long-term personal fact worth remembering. "
                    "Preserve the COMPLETE information exactly as provided. "
                    "Do not shorten names or facts. "
                    "If the user provides a full name, preserve the full name. "
                    "If yes, return ONLY the extracted fact. "
                    "If no, return ONLY: NONE"
                )
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )

    result = response.choices[0].message.content.strip()

    if result == "NONE":
        return None

    return result
