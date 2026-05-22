import json
from src.config import MEMORY_FILE


def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {
            "facts": [],
            "summary": ""
        }

    with open(MEMORY_FILE, "r") as file:
        return json.load(file)


def save_memory(memory: dict) -> None:
    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=2)
