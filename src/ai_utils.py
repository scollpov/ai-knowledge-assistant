from openai import OpenAI
from dotenv import load_dotenv
from src.config import EMBEDDING_MODEL
import numpy as np

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding
