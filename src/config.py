from pathlib import Path

RETRIEVAL_K = 10

FINAL_K = 3

MAX_DISTANCE = 1.6

EMBEDDINGS_FILE = "data/embeddings.json"

CHAT_MODEL = "gpt-4o-mini"

EMBEDDING_MODEL = "text-embedding-3-small"

MAX_HISTORY_MESSAGES = 6

MEMORY_FILE = Path("data/memory.json")

REPORT_FILE = "evaluation/results/latest_retrieval_report.json"
