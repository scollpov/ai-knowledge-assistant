from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()

client = OpenAI()

knowledge_chunks = [
    "Python is widely used in AI engineering and backend systems.",
    "FastAPI is a lightweight Python framework for building APIs.",
    "RAG reduces hallucinations by retrieving relevant context before generation.",
    "Docker containers help developers deploy applications consistently."
]

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Creating knowledge embeddings...\n")

knowledge_embeddings = []

for chunk in knowledge_chunks:
    embedding = get_embedding(chunk)
    knowledge_embeddings.append(embedding)

question = input("Ask a question: ")

question_embedding = get_embedding(question)

best_score = -1
best_chunk = ""

for i, chunk_embedding in enumerate(knowledge_embeddings):
    score = cosine_similarity(question_embedding, chunk_embedding)

    print(f"\nChunk: {knowledge_chunks[i]}")
    print(f"Similarity: {score:.4f}")

    if score > best_score:
        best_score = score
        best_chunk = knowledge_chunks[i]

print("\nBest retrieved context:")
print(best_chunk)

print("\nGenerating final answer...\n")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": f"""
            Answer the user's question ONLY using this context:

            {best_chunk}
            """
        },
        {
            "role": "user",
            "content": question
        }
    ]
)

print(response.choices[0].message.content)
