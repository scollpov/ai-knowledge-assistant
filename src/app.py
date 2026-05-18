from fastapi import FastAPI, HTTPException
from pathlib import Path
from src.rag_service import answer_question
from src.models import QuestionRequest, QuestionResponse
from src.config import EMBEDDINGS_FILE

app = FastAPI(
    title="AI Knowledge Assistant API",
    description="A Retrieval-Augmented Generation API using embeddings, semantic search, and OpenAI.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_check():
    embeddings_path = Path(EMBEDDINGS_FILE)

    if not embeddings_path.exists():
        print("WARNING: embeddings file not found. Run ingestion before using /ask.")
    else:
        print(f"Embeddings file found: {EMBEDDINGS_FILE}")

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
