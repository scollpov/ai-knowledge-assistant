from src.rag_service import answer_question

question = input("Ask a question: ")

result = answer_question(question)

print("\nFinal answer:")
print(result["answer"])

print("\nSources:")
for source in result["sources"]:
    print(f"- {source['source']} | score={source['score']:.4f}")
    print(f"  {source['text']}")
