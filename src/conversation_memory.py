from src.config import MAX_HISTORY_MESSAGES
from src.conversation_summary import update_summary

conversation_history = []

def add_message(role: str, content: str):

    conversation_history.append({
        "role": role,
        "content": content
    })

    if len(conversation_history) > MAX_HISTORY_MESSAGES:

        old_messages = conversation_history[:-MAX_HISTORY_MESSAGES]

        update_summary(old_messages)

        del conversation_history[:-MAX_HISTORY_MESSAGES]

def get_history():

    return conversation_history

def clear_history():

    global conversation_history
    conversation_history = []
