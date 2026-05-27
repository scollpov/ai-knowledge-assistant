from src.memory_storage import (
    load_memory,
    save_memory
)
from src.logger import logger


memory = load_memory()

long_term_memory  = memory["facts"]


def add_fact(fact: str):

    if fact not in long_term_memory:
        long_term_memory.append(fact)

        memory["facts"] = long_term_memory

        save_memory(memory)

        logger.info(f"Fact stored: {memory['facts']}")


def get_facts():

    return long_term_memory


def clear_facts():

    long_term_memory.clear()

    memory["facts"] = long_term_memory

    save_memory(memory)
