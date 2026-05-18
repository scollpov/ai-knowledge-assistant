from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

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

text_1 = "Cars, trucks, and buses are road vehicles used to transport people and goods."
text_2 = "Vehicles such as automobiles and buses help humans travel between locations."
text_3 = "Bananas, apples, and oranges are fruits commonly eaten as snacks."

embedding_1 = get_embedding(text_1)
embedding_2 = get_embedding(text_2)
embedding_3 = get_embedding(text_3)

print("cars vs vehicles:", cosine_similarity(embedding_1, embedding_2))
print("cars vs bananas:", cosine_similarity(embedding_1, embedding_3))
print("vehicles vs bananas:", cosine_similarity(embedding_2, embedding_3))
