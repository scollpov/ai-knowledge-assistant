# AI Knowledge Assistant

A Retrieval-Augmented Generation (RAG) backend service built with FastAPI, OpenAI embeddings, and semantic search.

---

## Features

- ChromaDB vector database
- Persistent vector storage
- Metadata-based retrieval filtering
- OpenAI embeddings
- Semantic similarity retrieval
- Multi-document ingestion
- Sentence-aware chunking
- Chunk overlap strategy
- Top-k retrieval
- Retrieval confidence thresholds
- Source attribution
- FastAPI backend
- Swagger/OpenAPI documentation
- Typed request/response validation
- Async API endpoints
- Error handling and validation

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

data/
    documents/
    chroma

experiments/
```

---

## Technologies

- Python
- FastAPI
- OpenAI API
- NumPy
- Pydantic
- Uvicorn

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
- ChromaDB is used for persistent vector storage

Place `.txt` or `.pdf` documents inside:

```txt
data/documents/
```

Run ingestion:

```bash
python -m src.ingest
```

This generates:

```txt
data/embeddings.json
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
