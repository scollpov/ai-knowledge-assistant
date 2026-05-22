long_term_memory = []


def add_fact(fact: str):

    if fact not in long_term_memory:
        long_term_memory.append(fact)


def get_facts():

    return long_term_memory
