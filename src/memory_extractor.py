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
                    "You extract long-term facts from user messages.\n\n"
                    "Return ONLY one of these:\n"
                    "- the exact fact worth remembering\n"
                    "- NONE\n\n"
                    "A fact is worth remembering if it is stable and useful for future conversations.\n\n"
                    "Always remember:\n"
                    "- the user's name or full name\n"
                    "- the user's ongoing projects\n"
                    "- the user's technical stack or preferences\n"
                    "- the user's professional goals\n\n"
                    "Never remember:\n"
                    "- reminders\n"
                    "- temporary plans\n"
                    "- casual conversation\n"
                    "- one-time tasks\n\n"
                    "Rules:\n"
                    "- Preserve the complete fact exactly as written.\n"
                    "- Do not shorten names.\n"
                    "- Do not rewrite the fact.\n"
                    "- Do not add explanations.\n\n"
                    "Examples:\n"
                    "User: My full name is Juan Roca Veloz\n"
                    "Output: My full name is Juan Roca Veloz\n\n"
                    "User: I am working on an AI engineering portfolio project\n"
                    "Output: I am working on an AI engineering portfolio project\n\n"
                    "User: Remind me tomorrow to buy milk\n"
                    "Output: NONE"
                )
            },
            {
                "role": "user",
                "content": message,
            },
        ],
    )

    result = response.choices[0].message.content.strip()

    if result.upper() == "NONE":
        return None

    return result
