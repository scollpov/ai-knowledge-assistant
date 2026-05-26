import json

from src.vector_store import collection
from src.ai_utils import get_embedding
from src.config import RETRIEVAL_K
from src.retrieval_service import retrieve_sources


def load_cases(path: str):
    with open(path, "r") as file:
        return json.load(file)


def evaluate():
    cases = load_cases("evaluation/retrieval_cases.json")

    for case in cases:
        question = case["question"]
        expected_keywords = case["expected_keywords"]

        filter_metadata = case.get("filter_metadata")

        results = retrieve_sources(
            rewritten_query=question,
            filter_metadata=filter_metadata
        )

        retrieved_text = " ".join(
            result["text"]
            for result in results
        ).lower()

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in retrieved_text
        ]

        print("\nQuestion:", question)
        print("Expected:", expected_keywords)
        print("Matched:", matched_keywords)
        
        required_matches = case.get("required_matches", 2)
        should_retrieve = case.get("should_retrieve", True)

        if not should_retrieve:
            passed = len(results) == 0

            print("Result (negative retrieval):", "PASS" if passed else "FAIL")
        
            continue

        passed = len(matched_keywords) >= required_matches

        print("Result:(retrieval)", "PASS" if passed else "FAIL")
        
if __name__ == "__main__":
    evaluate()
