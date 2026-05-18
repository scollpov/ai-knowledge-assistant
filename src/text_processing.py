def chunk_text(text: str, chunk_size: int = 300, overlap_sentences: int = 1) -> list[str]:

    sentences = [
        sentence.strip()
        for sentence in text.split(".")
        if sentence.strip()
    ]

    chunks = []	
    current_chunk = []

    for sentence in sentences:
        sentence = sentence + "."

        candidate = " ".join(current_chunk + [sentence])

        if len(candidate) <= chunk_size:
            current_chunk.append(sentence)
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            overlap = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap + [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
