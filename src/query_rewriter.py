from openai import OpenAI

client = OpenAI()

def rewrite_query(
    conversation_history,
    question: str
) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the user's latest question into a concise standalone "
                "search query for semantic retrieval. "
                "DO NOT answer the question. "
                "DO NOT explain anything. "
                "ONLY return the rewritten search query."
            )
        }
    ]

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=messages
    )

    return response.choices[0].message.content
