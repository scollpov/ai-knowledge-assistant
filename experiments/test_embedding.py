from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

print("Creating embedding...")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Cars and vehicles are used for transportation."
)

embedding = response.data[0].embedding

print("Embedding created.")
print(f"Vector size: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
