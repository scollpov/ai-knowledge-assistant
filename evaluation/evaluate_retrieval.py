import os
import json

from src.vector_store import collection
from src.ai_utils import get_embedding
from src.config import RETRIEVAL_K
from src.retrieval_service import retrieve_sources


def load_cases(path: str):
    with open(path, "r") as file:
        return json.load(file)


def evaluate():

    total_cases = 0
    passed_cases = 0

    false_positives = 0
    false_negatives = 0

    report_cases = []

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

        scores = [
            result["score"]
            for result in results
        ]

        average_score = (
            sum(scores) / len(scores)
            if scores else None
        )

        rerank_scores = [
            result["rerank_score"]
            for result in results
        ]

        average_rerank_score = (
            sum(rerank_scores) / len(rerank_scores)
            if rerank_scores else None
        )

        source_ids = [
            result["id"]
            for result in results
        ]

        source_files = [
            result["source"]
            for result in results
        ]

        matched_keywords = [
            keyword
            for keyword in expected_keywords
            if keyword.lower() in retrieved_text
        ]

        print("\nQuestion:", question)
        print("Expected:", expected_keywords)
        print("Matched:", matched_keywords)
        print("Average distance:", average_score)
        print("Average rerank score:", average_rerank_score)
        print("Source IDs:", source_ids)
        print("Source files:", source_files)

        required_matches = case.get("required_matches", 2)
        should_retrieve = case.get("should_retrieve", True)

        total_cases += 1

        passed = (
            (should_retrieve and len(matched_keywords) >= required_matches) or
            (not should_retrieve and len(results) == 0)
        ) 

        if passed: 
            passed_cases += 1 

        else:
            false_positives += 1 if should_retrieve else 0
            false_negatives += 0 if should_retrieve else 1

        print(
            "Result ", 
            "(positive retrieval):" if should_retrieve else "(negative retrieval):", 
            "PASS" if passed else "FAIL"
        )

        report_cases.append({
            "question": question,
            "expected_keywords": expected_keywords,
            "matched_keywords": matched_keywords,
            "should_retrieve": should_retrieve,
            "average_score": average_score,
            "average_rerank_score": average_rerank_score,
            "source_ids": source_ids,
            "source_files": source_files,
            "passed": passed
        })
        
    accuracy = (passed_cases / total_cases) * 100 if total_cases else 0

    print("\n====================")
    print("Evaluation Summary")
    print("====================")
    print(f"Passed: {passed_cases}/{total_cases}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")

    report = {
        "total_cases": total_cases,
        "passed_cases": passed_cases,
        "accuracy": accuracy,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "cases": report_cases
    }

    os.makedirs("evaluation/results", exist_ok=True)

    with open("evaluation/results/latest_retrieval_report.json", "w") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    evaluate()
