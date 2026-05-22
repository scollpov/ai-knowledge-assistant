# AI Knowledge Assistant

A conversational Retrieval-Augmented Generation (RAG) backend service built with FastAPI, OpenAI embeddings, ChromaDB, and semantic search.

---

## Features

- ChromaDB vector database
- Persistent vector storage
- Metadata-based retrieval filtering
- OpenAI embeddings
- Semantic similarity retrieval
- Multi-document ingestion
- PDF document ingestion
- Sentence-aware chunking
- Chunk overlap strategy
- Top-k retrieval
- Retrieval confidence thresholds
- Source attribution
- Query rewriting for conversational retrieval
- Conversational memory
- Long-term fact memory
- Summarized conversation history
- Reranking pipeline
- Incremental document indexing
- FastAPI backend
- Swagger/OpenAPI documentation
- Typed request/response validation
- Async API endpoints
- Error handling and validation
- Dockerized deployment support

---

## Project Structure

```txt
src/
    app.py
    rag_service.py
    ai_utils.py
    text_processing.py
    config.py
    models.py
    ingest.py
    query.py
    reranker.py
    conversation_memory.py
    conversation_summary.py
    long_term_memory.py
    memory_extractor.py
    query_rewriter.py

data/
    documents/
    chroma/

experiments/
```

---

## Technologies

- Python
- FastAPI
- OpenAI API
- ChromaDB
- NumPy
- Pydantic
- Uvicorn
- Docker

---

## Memory Architecture

The assistant includes multiple memory layers:

- Short-term memory: recent conversation messages
- Summarized memory: compressed older conversation history
- Long-term memory: extracted user facts
- Retrieval memory: document chunks stored in ChromaDB

This allows the assistant to support follow-up questions, remember important user facts during a session, and combine conversational context with document retrieval.

---

## Conversational Retrieval Architecture

The assistant supports conversational RAG workflows through:

- Query rewriting for standalone semantic retrieval
- Conversational memory integration
- Long-term fact extraction
- Retrieval reranking
- Context summarization
- Retrieval fallback to conversational answering

Pipeline overview:

```txt
User Question
    ↓
Query Rewriting
    ↓
Embedding Generation
    ↓
Vector Retrieval
    ↓
Reranking
    ↓
LLM Generation
```

---

## Setup

Create virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` file:

```env
OPENAI_API_KEY=your_api_key
```

---

## Ingest Documents

The ingestion pipeline supports incremental updates:

- unchanged documents are skipped
- changed documents are re-embedded
- document hashes are stored as metadata
- stale chunks are automatically removed
- ChromaDB is used for persistent vector storage

Place `.txt` or `.pdf` documents inside:

```txt
data/documents/
```

Run ingestion:

```bash
python -m src.ingest
```

Persistent vector data is stored in:

```txt
data/chroma/
```

---

## Run CLI Query

```bash
python -m src.query
```

Example question:

```txt
How do AI systems reduce hallucinations?
```

Clear conversation memory:

```txt
clear
```

Exit:

```txt
exit
```

---

## Run API

Start FastAPI server:

```bash
uvicorn src.app:app --reload
```

Open interactive API documentation:

```txt
http://127.0.0.1:8000/docs
```

---

## Docker Deployment

Build Docker image:

```bash
docker build -t ai-knowledge-assistant .
```

Run container:

```bash
docker run -p 8000:8000 --env-file .env ai-knowledge-assistant
```

The container automatically supports cloud deployment platforms using the `PORT` environment variable.

---

## Example API Request With Metadata Filter

```json
{
  "question": "What helps package applications?",
  "filter_metadata": {
    "document_name": "software_engineering"
  }
}
```

---

## Example API Response

```json
{
  "answer": "AI systems reduce hallucinations by using Retrieval-Augmented Generation (RAG)...",
  "sources": [
    {
      "id": 2,
      "source": "data/documents/ai_concepts.txt",
      "score": 0.42
    }
  ]
}
```

---

## Future Improvements

- Cloud deployment
- Persistent database-backed memory
- Multi-user memory isolation
- Streaming responses
- Agent/tool routing
- Hybrid search
- Evaluation dashboards
- Authentication and authorization
