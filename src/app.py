from fastapi import FastAPI
from src.rag_service import answer_question
from src.models import QuestionRequest, QuestionResponse

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "AI Knowledge Assistant API"
    }

@app.post("/ask", response_model=QuestionResponse)
def ask_question(payload: QuestionRequest):

    question = payload.question

    return answer_question(question)
