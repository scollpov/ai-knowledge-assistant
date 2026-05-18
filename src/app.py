from fastapi import FastAPI
from src.rag_service import answer_question

app = FastAPI()

@app.post("/ask")
def ask_question(payload: dict):

    question = payload.get("question", "")

    return answer_question(question)
