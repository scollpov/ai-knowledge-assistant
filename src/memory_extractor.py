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
                    "You extract long-term personal facts from user messages.\n\n"
                    "Return ONLY one of these:\n"
                    "- the exact fact worth remembering\n"
                    "- NONE\n\n"
                    "Rules:\n"
                    "- Only extract stable facts likely to remain useful in future conversations.\n"
                    "- Preserve the complete fact exactly as provided.\n"
                    "- Do not shorten names, project names, titles, or technical details.\n"
                    "- Do not infer, summarize, or rewrite beyond minimal cleanup.\n"
                    "- If the user gives a full name, preserve the full name.\n"
                    "- If the message is temporary, conversational, or unclear, return NONE.\n\n"
                    "Examples:\n"
                    "User: My full name is Juan Roca Veloz\n"
                    "Output: My full name is Juan Roca Veloz\n\n"
                    "User: I am working on an AI engineering portfolio project\n"
                    "Output: I am working on an AI engineering portfolio project\n\n"
                    "User: thanks\n"
                    "Output: NONE\n\n"
                    "User: remind me tomorrow\n"
                    "Output: NONE"
                ),
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
