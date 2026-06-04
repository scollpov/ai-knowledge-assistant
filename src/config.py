from pathlib import Path


RETRIEVAL_K = 10
FINAL_K = 3


MAX_DISTANCE = 1.6
MAX_HISTORY_MESSAGES = 6


CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


MEMORY_FILE = Path("data/memory.json")
REPORT_FILE = "evaluation/results/latest_retrieval_report.json"
EMBEDDINGS_FILE = "data/embeddings.json"


PROMPT_TOKEN_COST_PER_1M = 0.40
COMPLETION_TOKEN_COST_PER_1M = 1.60
