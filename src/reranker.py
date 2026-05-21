import re

def tokenize(text: str) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())

    stopwords = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on",
        "for", "with", "is", "are", "was", "were", "what", "how",
        "does", "do", "did", "about"
    }

    return {
        word
        for word in words
        if word not in stopwords
    }


def keyword_overlap_score(question: str,text: str) -> int:

    question_words = tokenize(question)
    text_words = tokenize(text)

    return len(question_words.intersection(text_words))
