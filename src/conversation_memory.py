from src.config import (MAX_HISTORY_MESSAGES)

conversation_history = []

def add_message(role: str, content: str):

    conversation_history.append({
        "role": role,
        "content": content
    })

def get_history():

    return conversation_history[-MAX_HISTORY_MESSAGES:]

def clear_history():

    conversation_history.clear()
