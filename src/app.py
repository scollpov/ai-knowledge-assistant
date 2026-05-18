from fastapi import FastAPI, HTTPException
from src.rag_service import answer_question
from src.models import QuestionRequest, QuestionResponse

app = FastAPI(
    title="AI Knowledge Assistant API",
    description="A Retrieval-Augmented Generation API using embeddings, semantic search, and OpenAI.",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI Knowledge Assistant API"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(payload: QuestionRequest):

    if not payload.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:
        return answer_question(payload.question)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Embeddings file not found. Run ingestion first."
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )
