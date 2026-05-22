from src.rag_service import answer_question
from src.conversation_memory import clear_history
from src.long_term_memory import clear_facts

while True:
    question = input("\nAsk a question: ")

    if question.lower() in ["exit", "quit"]:
        break;
    
    if question.lower() == "clear":

        clear_history()

        print("Conversation history cleared.")

        continue

    if question.lower() == "clear facts":

        clear_facts()

        print("Long-term facts cleared.")

        continue

    result = answer_question(question)

    print("\nFinal answer:")
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source['source']} | distance={source['score']:.4f} | rerank={source['rerank_score']}")
        print(f"  {source['text'][:300]}...")
