from fastapi import FastAPI
from src.rag_service import answer_question
from src.models import QuestionRequest

app = FastAPI()

@app.post("/ask")
def ask_question(payload: QuestionRequest):

    question = payload.question

    return answer_question(question)
