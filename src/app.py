from fastapi import FastAPI, HTTPException
from pathlib import Path
from fastapi.responses import StreamingResponse

from src.rag_service import answer_question
from src.models import QuestionRequest, QuestionResponse
from src.config import EMBEDDINGS_FILE
from src.logger import logger
from src.rag_service import stream_answer_question


app = FastAPI(
    title="AI Knowledge Assistant API",
    description="A Retrieval-Augmented Generation API using embeddings, semantic search, and OpenAI.",
    version="0.1.0"
)


@app.on_event("startup")
async def startup_check():
    embeddings_path = Path(EMBEDDINGS_FILE)

    if not embeddings_path.exists():
        logger.warning("WARNING: embeddings file not found. Run ingestion before using /ask.")
    else:
        logger.info(f"Embeddings file found: {EMBEDDINGS_FILE}")


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
        return answer_question(
            payload.question,
            filter_metadata=payload.filter_metadata
        )
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


@app.post("/ask-stream")
def ask_stream(request: QuestionRequest):
    return StreamingResponse(
        stream_answer_question(
            question=request.question,
            filter_metadata=request.filter_metadata
        ),
        media_type="text/event-stream"
    )
