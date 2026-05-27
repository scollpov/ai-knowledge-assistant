# AI Knowledge Assistant

A production-oriented conversational Retrieval-Augmented Generation (RAG) backend service built with FastAPI, OpenAI embeddings, ChromaDB, semantic retrieval, conversational memory, and retrieval evaluation infrastructure.

---

## Overview

AI Knowledge Assistant is an end-to-end AI backend project focused on:

- Retrieval-Augmented Generation (RAG)
- Conversational AI workflows
- Long-term memory systems
- Semantic retrieval and reranking
- Retrieval evaluation infrastructure
- AI system observability and regression detection

The project demonstrates real-world AI engineering patterns used in modern LLM applications.

---

## Features

### Retrieval & RAG

- ChromaDB vector database
- Persistent vector storage
- OpenAI embeddings
- Semantic similarity retrieval
- Metadata-based retrieval filtering
- Multi-document ingestion
- PDF document ingestion
- Incremental document indexing
- Sentence-aware chunking
- Chunk overlap strategy
- Retrieval confidence thresholds
- Top-k retrieval
- Retrieval reranking pipeline
- Source attribution
- Conversational query rewriting

### Memory Architecture

- Conversational short-term memory
- Summarized conversation memory
- Long-term fact memory
- User fact extraction
- Retrieval memory stored in ChromaDB

### Evaluation Infrastructure

- Retrieval evaluation framework
- Positive retrieval tests
- Negative retrieval tests
- False positive tracking
- False negative tracking
- Retrieval accuracy metrics
- Distance metric tracking
- Rerank metric tracking
- Persisted evaluation reports
- Regression detection
- Retrieval diagnostic metadata
- Source tracking in reports

### Backend & API

- FastAPI backend
- Async API endpoints
- Typed request/response validation
- Swagger/OpenAPI documentation
- Error handling and validation
- Dockerized deployment support

---

## Architecture

### Conversational Retrieval Pipeline

```txt
User Question
    ↓
Should Retrieve?
    ↓
Query Rewriting
    ↓
Embedding Generation
    ↓
Vector Retrieval
    ↓
Keyword Reranking
    ↓
Context Assembly
    ↓
LLM Response Generation
    ↓
Long-Term Memory Extraction
```

---

## Memory Architecture

The assistant includes multiple memory layers:

- Short-term memory: recent conversation messages
- Summarized memory: compressed older conversation history
- Long-term memory: extracted user facts
- Retrieval memory: document chunks stored in ChromaDB

This architecture allows the assistant to:

- support follow-up questions
- maintain conversational context
- remember important user facts
- combine memory with semantic retrieval
- reduce hallucinations using retrieved context

---

## Retrieval Evaluation Framework

The project includes a dedicated retrieval evaluation system.

### Supported Evaluation Capabilities

- Positive retrieval evaluation
- Negative retrieval evaluation
- Retrieval regression detection
- Accuracy tracking
- False positive tracking
- False negative tracking
- Retrieval score metrics
- Rerank score metrics
- Source tracking
- Persisted JSON evaluation reports

### Example Evaluation Report

```txt
{
  "accuracy": 100.0,
  "false_positives": 0,
  "false_negatives": 0
}
```

Evaluation reports are persisted in:

```txt
evaluation/results/
```

---

## Project Structure

```txt
src/
    app.py
    rag_service.py
    retrieval_service.py
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


evaluation/
    evaluate_retrieval.py
    retrieval_cases.json

    results/
        latest_retrieval_report.json


tests/
    test_memory_extractor.py


data/
    documents/
    chroma/
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
- Pytest

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

## Document Ingestion

The ingestion pipeline supports incremental indexing:

- unchanged documents are skipped
- changed documents are re-embedded
- document hashes are stored as metadata
- stale chunks are automatically removed
- vector embeddings are persisted in ChromaDB

Place `.txt` or `.pdf` documents inside:

```txt
data/documents/
```

Run ingestion:

```bash
python -m src.ingest
```

Persistent vector storage:

```txt
data/chroma/
```

---

## Run Conversational CLI

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

## Run Retrieval Evaluation

Run retrieval benchmark suite:

```bash
python -m evaluation.evaluate_retrieval
```

Evaluation cases are stored in:

```txt
evaluation/retrieval_cases.json
```

Generated evaluation reports:

```txt
evaluation/results/latest_retrieval_report.json
```

---

## Run Tests

```bash
pytest
```

---

## Run API

Start FastAPI server:

```bash
uvicorn src.app:app --reload
```

Interactive API documentation:

```txt
http://127.0.0.1:8000/docs
```

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

## Docker Deployment

Build Docker image:

```bash
docker build -t ai-knowledge-assistant .
```

Run container:

```bash
docker run -p 8000:8000 --env-file .env ai-knowledge-assistant
```

The container supports cloud deployment platforms using the `PORT` environment variable.

---

## Future Improvements

- Hybrid search
- Streaming responses
- Multi-user memory isolation
- Persistent database-backed memory
- Agent/tool routing
- Authentication and authorization
- CI/CD evaluation automation
- Evaluation dashboards
- Retrieval observability dashboards
- Cloud-native deployment
- Semantic caching
- Multi-modal retrieval

