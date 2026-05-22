from typing import List, Dict
from openai import OpenAI
from src.config import CHAT_MODEL

client = OpenAI()

conversation_summary = ""

def update_summary(history: List[Dict]) -> None:

    global conversation_summary
    
    if not history:
        return

    previous_summary = conversation_summary or "No previous summary."

    messages_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in history
    )

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Summarize the conversation so far for future assistant context. "
                    "Preserve important user goals, preferences, decisions, and unresolved topics. "
                    "Be concise."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Existing summary:\n{previous_summary}\n\n"
                    f"New messages:\n{messages_text}"
                )
            }
        ]
    )

    conversation_summary = response.choices[0].message.content

def get_summary() -> str:

    return conversation_summary

def clear_summary():

    global conversation_summary 
    conversation_summary = ""
