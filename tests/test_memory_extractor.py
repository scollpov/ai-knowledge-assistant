from src.memory_extractor import extract_fact


def test_preserves_full_name():
    fact = extract_fact("My full name is Juan Roca Veloz")
    assert fact == "My full name is Juan Roca Veloz"


def test_ignores_temporary_message():
    fact = extract_fact("Remind me tomorrow to buy milk")
    assert fact is None


def test_preserves_project_fact():
    fact = extract_fact("I am working on an AI engineering portfolio project")
    assert fact == "I am working on an AI engineering portfolio project"
