def keyword_overlap_score(
    question: str,
    text: str
) -> int:

    question_words = set(
        question.lower().split()
    )

    text_words = set(
        text.lower().split()
    )

    return len(
        question_words.intersection(text_words)
    )
