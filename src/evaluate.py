from src.rag_service import answer_question

TEST_CASES = [
    {
        "question": "What helps package applications?",
        "expected_keywords": [
            "Docker",
            "containers"
        ]
    },
    {
        "question": "How does RAG reduce hallucinations?",
        "expected_keywords": [
            "retrieving",
            "context"
        ]
    },
    {
        "question": "What are GPT-4 limitations?",
        "expected_keywords": [
            "limitations",
            "hallucinate",
            "learn"
        ]
    }
]

def evaluate():

    passed = 0

    for test in TEST_CASES:

        print("\n" + "=" * 80)

        question = test["question"]

        print(f"Question: {question}")

        result = answer_question(question)

        answer = result["answer"]

        print(f"\nAnswer:\n{answer}")

        success = True

        for keyword in test["expected_keywords"]:

            if keyword.lower() not in answer.lower():
                success = False

        if success:
            passed += 1
            print("\n✅ PASS")
        else:
            print("\n❌ FAIL")

    print("\n" + "=" * 80)
    print(f"Passed: {passed}/{len(TEST_CASES)}")


if __name__ == "__main__":
    evaluate()
